import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import engineer_features

from joblib import load
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

from utils import (
    load_data, preprocess_data,
    check_feature_alignment
)

def main():

    # Load tuned threshold (if it exists)
    threshold_path = 'models/stacking_threshold.txt'
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"[+] Using tuned threshold: {threshold}")
    else:
        threshold_file = "logs/optimal_threshold_stacking.txt"
        try:
            with open(threshold_file, "r") as f:
                threshold = float(f.read().strip())
            print(f"[+] Using tuned threshold: {threshold}")
        except FileNotFoundError:
            threshold = 0.5
            print(f"[+] Using default threshold: {threshold}")


    model_path = 'models/ensemble_stacking_model.joblib'
    test_data_path = 'data/training/UNSW_NB15_testing-set.csv'
    feature_names_path = 'models/top25_features.txt'



    print("Loading Stacking model...")
    model = load(model_path)

    print("Loading test dataset...")
    df = load_data(test_data_path)
    df = engineer_features(df)
    X_test, y_test, test_features = preprocess_data(df, selected_features_file=feature_names_path)

    with open(feature_names_path, "r") as f:
        train_features = [line.strip() for line in f]


    print("\n[+] Checking feature alignment...")
    X_test, aligned_features = check_feature_alignment(X_test, test_features, train_features)

    print(f"[+] X_test shape: {X_test.shape}")
    print("Evaluating Stacking model on test set...")

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Test label distribution:\n", y_test.value_counts())

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
