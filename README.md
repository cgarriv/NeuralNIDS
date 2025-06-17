# NeuralNIDS
A Machine Learning-enhanced Network Intrusion Detection System

# NeuralNIDS – ML Pipeline

This module implements the machine learning pipeline for NeuralNIDS, a Network Intrusion Detection System enhanced with ML-based threat detection. It handles the preprocessing, feature engineering, dimensionality reduction, and model training required to classify live network traffic as malicious or benign.

---

## Directory Structure

```
ml_pipeline/
├── utils.py                  # Preprocessing, feature engineering, training/evaluation helpers
├── apply_adasyn.py           # Applies ADASYN to rebalance the dataset
├── apply_umap.py             # Reduces feature dimensions using UMAP
├── train_dnn.py              # Trains the PyTorch-based DNN
├── dnn_predictor.py          # Loads DNN model and runs predictions
├── ml_predictor.py           # Runs predictions using an ensemble model
├── ml_alert_watcher.py       # Monitors live Suricata logs and triggers predictions
├── suricata_feature_adapter.py # Maps Suricata alert fields to ML feature vectors
```
---

## 1. Preprocessing & Feature Engineering

`utils.py`

- `preprocess_data()`: Cleans dataset, scales features, encodes categories, aligns features to a selected subset.
- `engineer_features()`: Adds advanced features (packet ratios, byte ratios, flag encodings).
- `train_model()`: Trains XGBoost (default), applies SMOTE for imbalance correction.
- `evaluate_model()`: Generates confusion matrix, AUC, and top feature plots.
- `check_feature_alignment()`: Ensures feature consistency between training and prediction phases.

---

##  2. Data Balancing

`apply_adasyn.py`

- Loads raw training data, one-hot encodes categorical variables.
- Applies ADASYN to balance minority attack classes.
- Saves: `training_data_adasyn.csv`

---

## 3. Dimensionality Reduction

`apply_umap.py`

- Trains a UMAP model to reduce feature space from hundreds to 10 components.
- Saves:
  - `training_data_adaysn_umap.csv`
  - `models/umap_model.joblib`

---

## 4. DNN Training

`train_dnn.py`

- Loads UMAP-reduced data.
- Defines a 3-layer PyTorch neural network.
- Trains for 50 epochs with `rich` progress bar.
- Outputs:
  - Classification report
  - AUC/ROC
  - Accuracy
  - `models/dnn_model.pth`

---

## 5. DNN Inference

`dnn_predictor.py`

- Loads and runs the trained PyTorch DNN model.
- Accepts UMAP-reduced `DataFrame` or `ndarray`.
- Returns:
  - Probability score
  - Label (`NORMAL` or `ATTACK`)

---

## 6. ML Inference

`ml_predictor.py`

- Loads ensemble model (`choosen model of user's choice`).
- Applies full preprocessing and feature engineering.
- Accepts raw alert-like JSON or DataFrame input.
- Applies threshold tuning from `optimal_threshold_stacking.txt`.

---

## 7. Real-Time Alert Prediction

`ml_alert_watcher.py`

- Monitors Suricata’s `eve.json` log.
- Extracts `event_type: alert` packets.
- Uses `suricata_feature_adapter.py` to convert alerts into model features.
- Applies UMAP + DNN prediction.
- Logs results to `logs/ml_alerts.jsonl`.

---

## 8. Feature Adapter

`suricata_feature_adapter.py`

- Maps Suricata alert fields to top 25 UNSW-NB15 features.
- Handles missing data gracefully.
- Can export raw vectors for retraining UMAP.

---

## Workflow Summary

```bash
# Step 1: Balance data
python apply_adasyn.py

# Step 2: Reduce dimensions
python apply_umap.py

# Step 3: Train DNN
python train_dnn.py

# Step 5: Run live predictions
python ml_alert_watcher.py
```

---

## Notes

- `top25_features.txt` must match between UMAP model, ensemble predictor, and feature adapter.
- `training_data_adaysn_umap.csv` is used specifically for DNN training.
- `suricata_feature_adapter.py` must remain synchronized with evolving Suricata schema if fields change.

---

Developed as part of the Senior Capstone Project – NMSU Spring 2025.
