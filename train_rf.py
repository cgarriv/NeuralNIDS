from utils import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from joblib import dump
import os


def save_feature_names(feature_names, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

def main():
    training_file = "data/training/UNSW_NB15_training-set.csv"
    top_features_path = "models/top25_features.txt"
    model_path = "models/rf_model_top25.joblib"
    feature_names_path = "models/feature_names_rf.txt"

    # Load and preprocess data using top 25 features
    df = load_data(training_file)
    X, y, selected_features = preprocess_data(df, selected_features_file=top_features_path)

    print("[+] Label distribution before SMOTE:")
    print(y.value_counts())

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("[+] After SMOTE:")
    print(y_resampled.value_counts())

    # Train RF classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_resampled, y_resampled)

    # Save model and feature names
    dump(model, model_path)
    save_feature_names(selected_features, feature_names_path)
    print(f"[+] RF model saved to '{model_path}'")
    print(f"[+] Features saved to '{feature_names_path}'")


if __name__ == "__main__":
    main()
