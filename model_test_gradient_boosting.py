# model_test_gradient_boosting.py

import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data

# thresholds we will test on the *validation* part of the TRAIN set
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

FEATURE_COLS = [
    'ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5',
    'ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10',
    'ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15',
    'ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20',
    'ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24',
    'mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5',
    'mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10',
    'mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15',
    'mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20',
    'mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24',
    'th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5',
    'th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10',
    'th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15',
    'th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20',
    'th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24'
]

def normalise_per_object(df, feature_cols):
    df_norm = df.copy()
    for obj_id, grp in df.groupby("object"):
        X = grp[feature_cols].values.astype(np.float32)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - mean) / std
        df_norm.loc[grp.index, feature_cols] = Xn
    return df_norm

def _load_xy():
    df_tr = load_data("bts3v2_palm_down")
    df_te = load_data("bts3v2_palm_down_test")

    df_tr[FEATURE_COLS] = df_tr[FEATURE_COLS].astype(np.float32)
    df_te[FEATURE_COLS] = df_te[FEATURE_COLS].astype(np.float32)

    df_tr = normalise_per_object(df_tr, FEATURE_COLS)
    df_te = normalise_per_object(df_te, FEATURE_COLS)

    X_train = df_tr[FEATURE_COLS].values.astype(np.float32)
    y_train = df_tr["slipped"].values.astype(int)
    X_test = df_te[FEATURE_COLS].values.astype(np.float32)
    y_test = df_te["slipped"].values.astype(int)

    return X_train, y_train, X_test, y_test


def _eval_at_threshold(y_true, probs, t):
    preds = (probs >= t).astype(int)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds)
    f1   = f1_score(y_true, preds)
    return acc, prec, rec, f1


def run(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = _load_xy()

    # -----------------------------
    # 1. Train/val split on TRAIN
    # -----------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # -----------------------------
    # 2. LightGBM with early stopping
    # -----------------------------
    clf = lgb.LGBMClassifier(
        n_estimators=2000,          # large, early stopping will cut it back
        learning_rate=0.02,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50)],
    )

    # -----------------------------
    # 3. Pick threshold on VAL
    # -----------------------------
    val_probs = clf.predict_proba(X_val)[:, 1]

    best_t = 0.5
    best_f1 = -1.0

    print("\n[LightGBM] Validation thresholds:")
    for t in THRESHOLDS:
        acc, prec, rec, f1 = _eval_at_threshold(y_val, val_probs, t)
        print(
            f"  t={t:5.2f}  acc={acc:.3f}  prec={prec:.3f}  "
            f"rec={rec:.3f}  f1={f1:.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\n[LightGBM] chosen threshold on validation: t={best_t:.2f} (F1={best_f1:.3f})")

    # -----------------------------
    # 4. Retrain on full TRAIN and test
    # -----------------------------
    # Use the best_iteration found during early stopping
    best_iters = clf.best_iteration_ if clf.best_iteration_ is not None else 300

    final_clf = lgb.LGBMClassifier(
        n_estimators=best_iters,
        learning_rate=0.02,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
    )
    final_clf.fit(X_train, y_train)

    test_probs = final_clf.predict_proba(X_test)[:, 1]
    acc, prec, rec, f1 = _eval_at_threshold(y_test, test_probs, best_t)

    model_path = os.path.join(model_dir, "slip_lgbm.txt")
    final_clf.booster_.save_model(model_path)

    metrics = {
        "model": "lightgbm",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": best_t,
        "model_path": model_path,
    }
    print("\n[LightGBM] FINAL test metrics:", metrics)
    return metrics


if __name__ == "__main__":
    run("model_test_models")
