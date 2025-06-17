import pandas as pd
import umap
import joblib

# Load raw feature vectors
df = pd.read_csv("logs/suricata_features_for_umap.csv")

# Useful columns
columns_to_keep = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur','sinpkt', 'smean']
df = df[columns_to_keep]

# Drop rows that are zero
df = df[(df.T != 0).any()]

print(f"[INFO] Training UMAP on {len(df)} rows, {len(columns_to_keep)} features...")

# Train UMAP
reducer = umap.UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
)

X_umap = reducer.fit_transform(df)

# Save UMAP
joblib.dump(reducer, "models/umap_model.joblib")
print("[X] UMAP retrained. Model saved in models/umap_model.joblib")