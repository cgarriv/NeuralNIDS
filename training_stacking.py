from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from joblib import load, dump
from utils import load_data, preprocess_data
import os
from utils import engineer_features
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():
    train_path = 'data/training/UNSW_NB15_training-set.csv'
    model_path = 'models/ensemble_stacking_model.joblib'
    top25_path = 'models/top25_features.txt'
    threshold_log_path = 'logs/optimal_threshold_stacking.txt'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(threshold_log_path), exist_ok=True)

    print("[+] Loading training data...")
    df = load_data(train_path)

    df = engineer_features(df)

    X, y, feature_names = preprocess_data(df, selected_features_file=top25_path)

    print("[+] Training Stacking Classifier...")
    # Load tuned base models
    xgb_model = load('models/xgb_model_tuned.joblib')
    rf_model = load('models/rf_model_tuned.joblib')

    # Define meta-learner
    meta_learner = LogisticRegression(max_iter=1000)

    # Define stacking classifier
    stack_model = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model)
        ],
        final_estimator=meta_learner,
        passthrough=True,
        n_jobs=-1
    )

    stack_model.fit(X, y)
    dump(stack_model, model_path)
    print(f"[+] Stacking model saved to: {model_path}")

if __name__ == "__main__":
    main()

