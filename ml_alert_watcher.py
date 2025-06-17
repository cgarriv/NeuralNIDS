from dnn_predictor import predict_event
from suricata_feature_adapter import build_feature_vector
import json
import time
import os
import pandas as pd
import joblib

EVE_LOG = "/var/log/suricata/eve.json"
ML_ALERT_LOG = "logs/ml_alerts.jsonl"
TOP25_FEATURES_PATH = "models/top25_features.txt"
UMAP_MODEL_PATH = "models/umap_model.joblib"

# Load UMAP model once
umap_model = joblib.load(UMAP_MODEL_PATH)

def load_top_features(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def map_suricata_to_features(event):
    df = build_feature_vector(event)
    df_reduced = umap_model.transform(df)
    return df_reduced


def tail_eve_and_predict():
    print("[+] Watching eve.json for new alerts...")
    top_features = load_top_features(TOP25_FEATURES_PATH)
    seen_flows = set()

    with open(EVE_LOG, "r") as f:
        f.seek(0, 2) # Jump to end of file
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            try:
                event = json.loads(line.strip())

                if event.get("event_type") != "alert":
                    continue

                flow_id = event.get("flow_id")
                if flow_id in seen_flows:
                    continue
                seen_flows.add(flow_id)

                print(f"[ALERT] {event.get('src_ip')} → {event.get('dest_ip')} | {event.get('proto')}")

                df_reduced = map_suricata_to_features(event)

                print("[+] Sending to ML model for prediction...")
                result = predict_event(df_reduced)

                if "Probability" in result and "Label" in result:
                    print(f"[+] Prediction: {result['Label']} ({result['Probability']})")
                    with open(ML_ALERT_LOG, "a") as log:
                        log.write(json.dumps({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "label": result["Label"],
                            "confidence": result["Probability"],
                        }) + "\n")
                else:
                    print(f"[X] ML Prediction failed: {result.get('Error', 'Unknown error')}")

            except Exception as e:
                print(f"[X] ML Prediction failed: {e}")

if __name__ == "__main__":
    tail_eve_and_predict()