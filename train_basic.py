# train_random_forest.py

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from load_data import load_data 

RNG = 42

def train_rf(X_train, y_train) -> RandomForestClassifier:
    """
    Train a simple RandomForest baseline.
    Uses 200 trees and default hyperparameters.
    Explanation of random forests works: 
        https://builtin.com/data-science/random-forest-algorithm
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RNG,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    return clf


def main():
    # Define output directory
    
    output_dir = "rf_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load training data and pull out features(X) & labels(y)
    
    print("Loading training data...")
    df = load_data("bts3v2_palm_down")

    y = df["slipped"].values

    feature_cols = ['ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5','ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10','ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15','ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20','ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24','mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5','mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10','mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15','mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20','mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24','th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5','th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10','th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15','th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20','th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24']
    X = df[feature_cols].values.astype(float)

    # Train/validation split 
    
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RNG,
        stratify=y,
    )

    # Train Random Forest

    print("Training baseline Random Forest...")
    clf = train_rf(X_train, y_train)

    # Metrics on validation set
    
    y_val_pred = clf.predict(X_val)
    
    # Validation accuracy

    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    print("Classification report (validation):")
    print(classification_report(y_val, y_val_pred, digits=4))
    
if __name__ == "__main__":
    main()
