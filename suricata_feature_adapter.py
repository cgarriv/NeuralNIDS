import os.path

import pandas as pd
import numpy as np

# Map Suricata fields to UNSW-NB15 top 25 features
# Use None or a lambda for derived or unavailable fields
SURICATA_TO_UNSW_MAPPING = {
    "proto": "proto",
    "flow.pkts_toserver": "spkts",
    "flow.pkts_toclient": "dpkts",
    "flow.bytes_toserver": "sbytes",
    "flow.bytes_toclient": "dbytes",
    "flow_id": "id",
    "flow.duration": "dur",
    # Placeholder for unavailable features
    "n/a_rate": "rate",
    "n/a_sload": "sload",
    "n/a_sloss": "sloss",
    "n/a_smean": "smean",
    "n/a_dload": "dload",
    "n/a_ct_dst_src_ltm": "ct_dst_src_ltm",
    "n/a_ct_srv_dst": "ct_srv_dst",

}

# Load top features file
TOP_FEATURES_FILE = "models/top25_features.txt"


def load_top_features():
    with open(TOP_FEATURES_FILE) as f:
        return [line.strip() for line in f.readlines()]


def flatten_suricata_alert(alert):
    # Flattens nested dicts in Suricata alert JSON.
    flat = {}
    for k, v in alert.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{k}.{subk}"] = subv
        else:
            flat[k] = v
    return flat


def build_feature_vector(suricata_event):
    flat_event = flatten_suricata_alert(suricata_event)

    # print("[DEBUG] Flattened event keys:")
    # print(list(flat_event.keys()))

    feature_row = {}
    top_features = load_top_features()

    for feature in top_features:
        matched = None
        for key, mapped_feature in SURICATA_TO_UNSW_MAPPING.items():
            if mapped_feature == feature:
                matched = key
                break

        if matched and matched in flat_event:
            feature_row[feature] = flat_event[matched]
        else:
            feature_row[feature] = 0.0  # default for missing/unavailable

    print("[DEBUG] Built feature row:")
    print(feature_row)

    return pd.DataFrame([feature_row])

if __name__ == "__main__":
    import json

    with open("example_alert.json") as f:
        alert = json.load(f)

    df = build_feature_vector(alert)
    # Dump feature vectors to CSV for UMAP retraining
    df.to_csv("logs/suricata_features_for_umap.csv", mode = "a", header = not os.path.exists("logs/suricata_features_for_umap.csv"), index=False)
    print(df.head())