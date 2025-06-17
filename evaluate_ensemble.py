import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from utils import load_data, preprocess_data, check_feature_alignment, evaluate_model
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def main():
    print("Loading Ensemble model...")
    model = load("models/ensemble_model.joblib") # Ensemble Model

    print("Loading test data...")
    df = load_data("data/training/UNSW_NB15_testing-set.csv")
    X_test, y_test, test_features = preprocess_data(df, selected_features_file="models/top25_features.txt")

    with open("models/feature_names_rf_tuned.txt") as f:
        train_features = [line.strip() for line in f]

    X_test, aligned_features = check_feature_alignment(X_test, test_features, train_features)
    print(f"[+] X_test shape: {X_test.shape}")

    print("Evaluating Ensemble model on test set...")
    evaluate_model(model, X_test, y_test)

    print("Test label distribution:")
    print(y_test.value_counts())

    # Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Ensemble Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="orange", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Ensemble Model")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
