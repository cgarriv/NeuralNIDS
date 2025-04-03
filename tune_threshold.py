import argparse
from joblib import load
from utils import load_data, preprocess_data
import numpy as np
from sklearn.metrics import f1_score

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

def tune_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"[+] Best threshold: {best_threshold:.4f} (F1 = {best_f1:.4f})")
    return best_threshold

def main():
    model_path = f"models/{args.model}_model_tuned.joblib" if args.model != "stacking" else "models/ensemble_stacking_model.joblib"
    df = load_data("data/training/UNSW_NB15_training-set.csv")
    X, y, _ = preprocess_data(df, selected_features_file="models/top25_features.txt")

    print("[+] Loading model...")
    model = load(model_path)

    print("[+] Tuning threshold...")
    y_probs = model.predict_proba(X)[:, 1]
    best_threshold = tune_threshold(y, y_probs)

    # Save threshold
    threshold_log_path = f"logs/optimal_threshold_{args.model}.txt"
    with open(threshold_log_path, "w") as f:
        f.write(f"{best_threshold:.4f}")
    print(f"[+] Optimal threshold saved to '{threshold_log_path}'")

if __name__ == "__main__":
    main()
