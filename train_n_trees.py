# train_random_forest.py

import os

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    log_loss
)

from load_data import load_data 

RANDOM_STATE = 42

def plot_accuracy_error_loss_vs_trees(
    X_train, y_train, X_val, y_val, output_dir: str
):
    """
    Random forests dont have 'epochs' like neural networks.
    Instead we'll train forests with different numbers of trees to see
    how performance scales with number of trees.
    We'll plot:
        - train accuracy
        - validation accuracy
        - validation error rate
        - validation log-loss
    versus number of trees.
    """
    n_trees_list = [10, 50, 100, 150, 200, 300]

    train_accs = []
    val_accs = []
    val_errors = []
    val_losses = []

    for n_trees in n_trees_list:
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        y_val_proba = clf.predict_proba(X_val)[:, 1]

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_error = 1.0 - val_acc
        # log_loss needs probabilities, ensure they are in (0,1)
        val_loss = log_loss(y_val, y_val_proba, labels=[0, 1])

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_errors.append(val_error)
        val_losses.append(val_loss)

    # Plot accuracy and error vs number of trees
    plt.figure()
    plt.plot(n_trees_list, train_accs, marker="o", label="Train accuracy")
    plt.plot(n_trees_list, val_accs, marker="o", label="Validation accuracy")
    plt.plot(n_trees_list, val_errors, marker="o", label="Validation error")
    plt.xlabel("Number of trees")
    plt.ylabel("Score")
    plt.title("Accuracy / Error vs Number of Trees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_error_vs_trees.png"))
    plt.show()

    # Plot log-loss vs number of trees
    plt.figure()
    plt.plot(n_trees_list, val_losses, marker="o")
    plt.xlabel("Number of trees")
    plt.ylabel("Log-loss")
    plt.title("Validation Log-loss vs Number of Trees")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_vs_trees.png"))
    plt.show()

    return


def main():
    output_dir = "rf_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data and pull out features(X) & labels(y)
    print("Loading training data...")
    df = load_data("bts3v2_palm_down")
    y = df["slipped"].values
    feature_cols = ['ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5','ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10','ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15','ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20','ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24','mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5','mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10','mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15','mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20','mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24','th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5','th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10','th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15','th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20','th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24']
    X = df[feature_cols].values.astype(float)


    # Simple train/validation split (no GroupKFold as requested)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training Random Forests with 'n' trees...")
    # 1) Accuracy / Loss / Error vs 'epochs' (using number of trees as proxy)
    print("Plotting accuracy/error/loss vs number of trees...")
    plot_accuracy_error_loss_vs_trees(X_train, y_train, X_val, y_val, output_dir)

if __name__ == "__main__":
    main()
