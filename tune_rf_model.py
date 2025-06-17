from utils import load_data, preprocess_data
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Save top features
def save_feature_names(feature_names, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

def main():
    file_path = 'data/training/UNSW_NB15_training-set.csv'
    top_features_file = 'models/top25_features.txt'
    model_path = 'models/rf_model_tuned.joblib'
    feature_names_out = 'models/feature_names_rf_tuned.txt'

    print("[+] Loading and preprocessing training data...")
    df = load_data(file_path)
    X, y, all_features = preprocess_data(df, selected_features_file=top_features_file)

    print("[+] Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("[+] Starting Random Forest hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='balanced_accuracy', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    best_model = grid_search.best_estimator_
    print("\n[+] Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Balanced Accuracy (CV avg): {grid_search.best_score_:.4f}")

    print(f"Saving tuned model to '{model_path}'")
    dump(best_model, model_path)

    print(f"Saving aligned feature names to '{feature_names_out}'")
    save_feature_names(all_features, feature_names_out)

if __name__ == "__main__":
    main()
