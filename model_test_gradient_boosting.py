# model_test_gradient_boosting.py

import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data

THRESHOLD = 0.5

FEATURE_COLS = [...]  # same list


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
