import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Define paths
model_path = 'models/xgb_model_tuned.joblib'
feature_names_path = 'models/feature_names.txt'
top_features_path = 'models/top25_features.txt'

# Load model
model = load(model_path)

# Load feature names
with open(feature_names_path, 'r') as f:
    feature_names = [line.strip() for line in f]

# Get feature importance
booster = model.get_booster()
importance_scores = booster.get_score(importance_type='weight')

# Map to actual feature names
mapped_scores = {
    feature_names[int(k[1:])]: v
    for k, v in importance_scores.items()
    if k[1:].isdigit() and int(k[1:]) < len(feature_names)
}

# Sort and get top 25
sorted_features = sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)[:25]

# Save top features
with open(top_features_path, 'w') as f:
    for feat, _ in sorted_features:
        f.write(f"{feat}\n")

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=[score for _, score in sorted_features], y=[feat for feat, _ in sorted_features], palette="viridis")
plt.title("Top 25 Most Important Features")
plt.xlabel("Importance Score (Weight)")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()
