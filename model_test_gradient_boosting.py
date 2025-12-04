# model_test_gradient_boosting.py

import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data

THRESHOLD = 0.5

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

def _load_xy():
    df_tr = load_data("bts3v2_palm_down")
    df_te = load_data("bts3v2_palm_down_test")

    X_train = df_tr[FEATURE_COLS].values.astype(np.float32)
    y_train = df_tr["slipped"].values.astype(int)
    X_test = df_te[FEATURE_COLS].values.astype(np.float32)
    y_test = df_te["slipped"].values.astype(int)

    return X_train, y_train, X_test, y_test


def run(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = _load_xy()

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    probs_test = clf.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test)
    f1 = f1_score(y_test, preds_test)

    model_path = os.path.join(model_dir, "slip_lgbm.txt")
    clf.booster_.save_model(model_path)

    metrics = {
        "model": "lightgbm",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": THRESHOLD,
        "model_path": model_path,
    }
    print("[LightGBM] metrics:", metrics)
    return metrics


if __name__ == "__main__":
    run("model_test_models")
