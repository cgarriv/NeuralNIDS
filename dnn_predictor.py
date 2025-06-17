import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np


# ___ CONFIG ___
MODEL_PATH = "models/dnn_model.pth"
N_FEATURES = 10 # After UMAP

# Define the DNN model
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

# Load DNN
model = NeuralNet(N_FEATURES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Predictions
def predict_event(input_data):
    try:
        if isinstance(input_data, pd.DataFrame):
            inputs = torch.tensor(input_data.values, dtype=torch.float32)
        elif isinstance(input_data, np.ndarray):
            inputs = torch.tensor(input_data, dtype=torch.float32)
        else:
            raise ValueError("Input must be a DataFrame or a ndarray")

        # Make prediction
        with (torch.no_grad()):
            outputs = model(inputs)
            prob = outputs.item()
            prediction = int(prob >= 0.5)

        return {
            "Probability": round(prob, 4),
            "Prediction": prediction,
            "Label": "ATTACK" if prediction == 1 else "NORMAL"
        }

    except Exception as e:
        print(f"[X] DNN Prediction Error: {e}")
        return {
            "Prediction": -1,
            "Confidence": 0.0,
            "Label": "Error",
            "Error": str(e)
        }
