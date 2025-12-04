# train_random_forest.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    log_loss,
)

from load_data import load_data 

RNG = 42

def train_rf(X_train, y_train) -> RandomForestClassifier:
    """
    Train a RandomForest and test its performance.
    Uses 100 trees and default hyperparameters.
    """
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


def plot_precision_recall(
    y_true, y_scores, output_dir: str, name: str = "validation"
):
    """
    Precision/Recall curve to help tune the classification threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curve ({name}), AP = {ap:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"precision_recall_{name}.png"))
    plt.show()

def plot_train_vs_val_accuracy(
    clf: RandomForestClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir: str,
):
    """
    Single point comparison of training vs validation accuracy
    for the final model.
    """
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    plt.figure()
    plt.bar(["Train", "Validation"], [train_acc, val_acc])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_vs_val_accuracy.png"))
    plt.show()

    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

def plot_confusion_matrix(
    clf: RandomForestClassifier,
    X_val,
    y_val,
    output_dir: str,
    name: str = "validation",
):
    """
    Confusion matrix on validation set.
    """
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no slip", "slip"],
    )
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"))
    plt.show()
    print("Confusion matrix:\n", cm)

def main():
    output_dir = "rf_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading training data...")
    df_train = load_data("bts3v2_palm_down")
    y_train = df_train["slipped"].values
    feature_cols = ['ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5','ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10','ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15','ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20','ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24','mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5','mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10','mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15','mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20','mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24','th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5','th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10','th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15','th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20','th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24']
    X_train = df_train[feature_cols].values.astype(float)

    print("Loading testing data...")
    df_test = load_data("bts3v2_palm_down_test")
    y_val = df_test["slipped"].values
    feature_cols = ['ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5','ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10','ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15','ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20','ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24','mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5','mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10','mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15','mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20','mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24','th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5','th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10','th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15','th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20','th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24']
    X_val = df_test[feature_cols].values.astype(float)

    print("Training baseline Random Forest...")
    clf = train_rf(X_train, y_train)

    # Metrics on validation set
    y_val_pred = clf.predict(X_val)
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    print("Classification report (validation):")
    print(classification_report(y_val, y_val_pred, digits=4))

    # 1) Precision–Recall curve (validation)
    print("Plotting Precision–Recall curve...")
    plot_precision_recall(y_val, y_val_proba, output_dir, name="validation")

    # 2) Training Accuracy vs Validation Accuracy
    print("Plotting train vs validation accuracy...")
    plot_train_vs_val_accuracy(clf, X_train, y_train, X_val, y_val, output_dir)

    # 3) Confusion Matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(clf, X_val, y_val, output_dir, name="validation")

    model_path = os.path.join(output_dir, "rf_slip_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Saved final model to: {model_path}")


if __name__ == "__main__":
    main()
