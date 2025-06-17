import matplotlib.pyplot as plt
from joblib import load
from utils import load_data, preprocess_data, evaluate_model, check_feature_alignment
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

def main():
    model_path = 'models/rf_model_tuned.joblib'
    test_data_path = 'data/training/UNSW_NB15_testing-set.csv'
    feature_names_path = 'models/feature_names_rf_tuned.txt'

    print("Loading tuned Random Forest model...")
    model = load(model_path)

    print("Loading test dataset...")
    df = load_data(test_data_path)
    X_test, y_test, test_features = preprocess_data(df, selected_features_file='models/top25_features.txt')

    # Load the feature names used during training
    with open(feature_names_path, 'r') as f:
        train_features = [line.strip() for line in f]

    # Align test features to match training features
    X_test, aligned_features = check_feature_alignment(X_test, test_features, train_features)

    print(f"[+] X_test shape: {X_test.shape}")
    print("Evaluating tuned Random Forest model on test set...")

    # Evaluate + Confusion Matrix + ROC
    evaluate_model(model, X_test, y_test)

    print("Test label distribution:")
    print(y_test.value_counts())

if __name__ == "__main__":
    main()
