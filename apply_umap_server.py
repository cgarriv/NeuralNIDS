import pandas as pd
import umap
import joblib

# Load ADASYN data
df = pd.read_csv('/home/cgarriv/neuralnids-backend/training_data_adasyn.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# One-Hot Encoding
X = pd.get_dummies(X)

# Apply UMAP
reducer = umap.UMAP(
    n_components=10,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
)

X_umap = reducer.fit_transform(X)


# Save UMAP model
joblib.dump(reducer, 'models/umap_model.joblib')
