from joblib import load, dump
from sklearn.ensemble import VotingClassifier
from utils import load_data, preprocess_data
from imblearn.over_sampling import SMOTE

print("[+] Loading base models...")
# Using both RF and XGBoost
xgb_model = load("models/xgb_model_tuned.joblib")
rf_model = load("models/rf_model_tuned.joblib")

# Training data with top 25 features
print("[+] Loading training data...")
df = load_data("data/training/UNSW_NB15_training-set.csv")
X, y, _ = preprocess_data(df, selected_features_file="models/top25_features.txt")

# Apply SMOTE
print("[+] Applying SMOTE for ensemble training...")
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Voting classifier with ensemble
print("[+] Training ensemble VotingClassifier...")
ensemble = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("rf", rf_model)
    ],
    voting="soft"
)

ensemble.fit(X_res, y_res)
dump(ensemble, "models/ensemble_model.joblib")
print("Ensemble model fitted and saved to 'models/ensemble_model.joblib'")
