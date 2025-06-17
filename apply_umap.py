import pandas as pd
import umap
import joblib
import matplotlib.pyplot as plt

# Load ADASYN data
df = pd.read_csv('data/training/training_data_adasyn.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Apply UMAP
reducer = umap.UMAP(
    n_components=10,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
)

X_umap = reducer.fit_transform(X)

# Create new DataFrame for the reduced dataset
df_umap = pd.DataFrame(X_umap, columns=[f'UMAP_{i}' for i in range(1, 11)])
df_umap['label'] = y.reset_index(drop=True)

# Save dataset and UMAP model
df_umap.to_csv('data/training/training_data_adaysn_umap.csv', index=False)
joblib.dump(reducer, 'models/umap_model.joblib')

print(f"Original feature space: {X.shape[1]} features")
print(f"Reduced feature space: {X_umap.shape[1]} features")
print(f"UMAP reduced dataset and model saved successfully")

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=5)
plt.title('UMAP projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='label')
plt.grid(True)
plt.show()
