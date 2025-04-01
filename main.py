from utils import load_data, preprocess_data, train_model, parse_pcap
import os
from joblib import dump, load

def main():
    # Specify your data file path. PCAP or CSV file.
    file_path = 'data/training/UNSW_NB15_training-set.csv'
    label = "normal"  # or "attack" if applicable

    model_path = 'models/xgb_model_top25.joblib'
    feature_names_path = 'models/feature_names.txt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = load_data(file_path, label)
    print("Extracted Data Preview:")
    print(df.head())

    X, y, feature_names = preprocess_data(df, selected_features_file="models/top25_features.txt")

    with open(feature_names_path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    print("Label distribution before training:")
    print(y.value_counts())

    # Train model and save
    model = train_model(X, y)
    dump(model, model_path)
    print(f"Model saved to '{model_path}'")

if __name__ == "__main__":
    main()