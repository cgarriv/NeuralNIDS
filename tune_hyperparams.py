import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from joblib import dump
from utils import load_data, preprocess_data
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE

def main():
    file_path = 'data/training/UNSW_NB15_training-set.csv'
    df = load_data(file_path)
    X, y, feature_names = preprocess_data(df)

    # SMOTE
    print("Applying SMOTE in tuning...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Parameter grid
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'min_child_weight': [1, 2],
        'scale_pos_weight': [0.8, 1.0, 1.2]
    }

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,  # fewer iterations
        scoring='balanced_accuracy',  # better balance
        cv=3,
        verbose=1,
        n_jobs=4,  # set for safety
        random_state=42
    )

    print("[+] Starting hyperparameter search...")
    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    print("\n[+] Best Parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"Balanced Accuracy (CV avg): {best_score:.4f}")

    # Save best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_model_tuned.joblib"
    dump(best_model, model_path)
    print(f" Best model saved to '{model_path}'")

    # Log to CSV
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/tuning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_data = pd.DataFrame([{
        **best_params,
        "balanced_accuracy": best_score
    }])
    log_data.to_csv(log_path, index=False)
    print(f"[+] Logged tuning results to '{log_path}'")

if __name__ == "__main__":
    main()
