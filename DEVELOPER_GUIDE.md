# 1. Preprocessing and Feature Engineering (utils.py)

### Main Functions:
- preprocess_data(): Cleans and Scales the dataset, handles encoding and feature selection.
- engineer_features(): Adds features like byte_ratio, packet_ration, flag encodings
- train_model(), evaluate_model(): Trains and evaluates XGBoost models
- parse_pcap(), load_data(): Load and label PCAP or CSV datasets

# 2. DNN Training (train_dnn.py)
### Workflow:
- Loads preprocessed UMAP-reduced training data (training_data_adaysn_umap.csv).
- Trains a custom PyTorch neural network with 3 fully connected layers and dropout.
- Tracks loss with rich live progress bars and console display.
- Evaluates using accuracy, classification report, AUC-ROC.
- Saves the model to models/dnn_model.pth.

# 3. DNN Inference (dnn_predictor.py)
### Role:
- Loads the saved DNN model.
- Exposes predict_event(input_data) to return:
  - Probability (0–1)
  - Prediction (0 or 1)
  - Label (ATTACK or NORMAL)

# 4. Ensemble Model Prediction (ml_predictor.py)

### Purpose:
- Loads a stacking ensemble model using joblib.
- Performs full preprocessing and feature engineering internally.
- Reads top25_features.txt for consistency.
- Returns a dictionary with prediction details.

# 5. Suricata Feature Adapter (suricata_feature_adapter.py)

### Purpose:
- Maps flattened eve.json alert keys to UNSW-NB15 features.
- Handles missing fields with default values.
- Loads top25_features.txt for compatibility.

# 6. Real-Time Alert Watcher (ml_alert_watcher.py)

### Core Functionality:
- Watches /var/log/suricata/eve.json in real time.
- Parses new alerts with event_type = alert.
- Extracts flow info and skips duplicates.
- Builds feature vector → applies UMAP → sends to predict_event().
- Logs labeled predictions to logs/ml_alerts.jsonl.

### To Run Real-Time Detection:
1. Ensure Suricata is logging alerts to `/var/log/suricata/eve.json`.
2. Run watcher:
   ```bash
   python ml_alert_watcher.py
   ```
   
## Notes

- Ensure the number and order of features match across training and prediction.
- Regenerate UMAP model when feature engineering changes.
- Keep top25_features.txt consistent project-wide.