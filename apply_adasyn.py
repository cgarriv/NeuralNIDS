import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
import joblib

# Load training dataset
df = pd.read_csv('data/training/UNSW_NB15_training-set.csv')

# Separate features and labels
X = df.drop('label', axis=1) # 'label' is the target column
y = df['label']

# One-Hot Encode
X = pd.get_dummies(X)

# Apply ADASYN
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Create a new DataFrame for the resampled data
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['label'] = y_resampled

# Save resampled dataset
df_resampled.to_csv('data/training/training_data_adasyn.csv', index=False)

print(f"Original dataset size: {X.shape}, Resampled dataset size: {X_resampled.shape}")
print("ADASYN resample dataset saved successfully")
