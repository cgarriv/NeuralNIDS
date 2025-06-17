import matplotlib.pyplot as plt
from joblib import load
from utils import load_data, preprocess_data, evaluate_model, check_feature_alignment
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np


def main():
    model_path = 'models/rf_model_top25.joblib'  # Random Forest model
    test_data_path = 'data/training/UNSW_NB15_testing-set.csv'
    top25_path = 'models/top25_features.txt'

    print("Loading trained Random Forest model...")
    model = load(model_path)

    print("Loading test dataset...")
    df = load_data(test_data_path)
    X_test, y_test, test_features = preprocess_data(df, selected_features_file=top25_path)

    # Load training feature names
    with open("models/feature_names.txt", "r") as f:
        train_features = [line.strip() for line in f]

    # Align test features to training
    X_test, aligned_features = check_feature_alignment(X_test, test_features, train_features)

    print(f"[+] X_test shape: {X_test.shape}")

    print("Evaluating Random Forest model on test set...")
    evaluate_model(model, X_test, y_test)

    print("Test label distribution:")
    print(y_test.value_counts())

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # --- ROC Curve ---
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
