# train_random_forest.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
)

from load_data import load_data

RNG = 42


def train_rf(X_train, y_train) -> RandomForestClassifier:
    """Train a RandomForest with fixed hyperparameters."""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RNG,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    return clf


def plot_precision_recall(y_true, y_scores, output_path: str, title: str):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP = {ap:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_train_vs_val_accuracy(train_acc, val_acc, output_path: str, title: str):
    plt.figure()
    plt.bar(["Train", "Validation"], [train_acc, val_acc])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, output_path: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no slip", "slip"],
    )
    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(title)
    print(cm)


def try_thresholds(y_true, y_proba, thresholds):
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print(f"\n=== Threshold {t:.2f} ===")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f} (for slip=1)")
        print(f"Recall   : {rec:.3f} (for slip=1)")
        print("Confusion matrix [rows:true, cols:pred]:")
        print(cm)


def main():
    output_dir = "rf_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load TRAIN data (used for GroupKFold and final training)
    # ------------------------------------------------------------
    print("Loading training data...")
    df_train = load_data("bts3v2_palm_down")

    feature_cols = ['ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5','ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10','ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15','ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20','ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24','mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5','mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10','mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15','mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20','mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24','th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5','th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10','th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15','th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20','th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24']

    X_all = df_train[feature_cols].values.astype(float)
    y_all = df_train["slipped"].values.astype(int)
    groups = df_train["object"].values

    print("Train label counts:\n", df_train["slipped"].value_counts())

    # ------------------------------------------------------------
    # 2. GroupKFold: evaluate generalisation to unseen objects
    # ------------------------------------------------------------
    unique_objects = np.unique(groups)
    n_splits = min(5, len(unique_objects))  
    gkf = GroupKFold(n_splits=n_splits)

    fold_train_acc = []
    fold_val_acc = []

    print(f"\nRunning GroupKFold with {n_splits} folds, grouping by object...\n")

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]

        clf = train_rf(X_tr, y_tr)
        
        y_tr_proba = clf.predict_proba(X_tr)[:, 1]
        y_tr_pred = (y_tr_proba >= 0.18).astype(int)
        y_val_proba = (clf.predict_proba(X_val)[:, 1])
        y_val_pred = (y_val_proba >= 0.18).astype(int)

        tr_acc = accuracy_score(y_tr, y_tr_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        fold_train_acc.append(tr_acc)
        fold_val_acc.append(val_acc)

        print(f"Fold {fold_idx}: train acc = {tr_acc:.3f}, val acc = {val_acc:.3f}")

        # For a single illustrative fold (say the first one), make plots
        if fold_idx == 1:
            plot_train_vs_val_accuracy(
                tr_acc,
                val_acc,
                os.path.join(output_dir, f"train_vs_val_accuracy_fold{fold_idx}.png"),
                title=f"Train vs Validation Accuracy (fold {fold_idx})",
            )

            plot_precision_recall(
                y_val,
                y_val_proba,
                os.path.join(output_dir, f"precision_recall_fold{fold_idx}.png"),
                title=f"Precision–Recall curve (fold {fold_idx})",
            )

            plot_confusion_matrix(
                y_val,
                y_val_pred,
                os.path.join(output_dir, f"confusion_matrix_fold{fold_idx}.png"),
                title=f"Confusion Matrix (fold {fold_idx})",
            )

    print("\nGroupKFold summary:")
    print("Mean train accuracy:", np.mean(fold_train_acc))
    print("Mean val accuracy:  ", np.mean(fold_val_acc))

    # ------------------------------------------------------------
    # 3. Retrain on ALL training objects, then evaluate on TEST set
    # ------------------------------------------------------------
    print("\nRetraining Random Forest on all training data...")
    final_clf = train_rf(X_all, y_all)

    # Save model
    model_path = os.path.join(output_dir, "rf_slip_model.pkl")
    joblib.dump(final_clf, model_path)
    print(f"Saved final model to: {model_path}")

    # ---- Test set (11 unseen objects) ----
    print("\nLoading test data...")
    df_test = load_data("bts3v2_palm_down_test")
    print("Test label counts:\n", df_test["slipped"].value_counts())

    X_test = df_test[feature_cols].values.astype(float)
    y_test = df_test["slipped"].values.astype(int)

    print("Evaluating on test set (unseen objects)...")
    y_test_proba = final_clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.25).astype(int)

    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")
    print("Classification report (test):")
    print(classification_report(y_test, y_test_pred, digits=4))

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    try_thresholds(y_test, y_test_proba, thresholds)

    plot_precision_recall(
        y_test,
        y_test_proba,
        os.path.join(output_dir, "precision_recall_test.png"),
        title="Precision–Recall curve (test)",
    )

    plot_confusion_matrix(
        y_test,
        y_test_pred,
        os.path.join(output_dir, "confusion_matrix_test.png"),
        title="Confusion Matrix (test)",
    )


if __name__ == "__main__":
    main()
