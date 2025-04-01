
def engineer_features(df):
    """
    Perform domain-specific feature engineering on the dataset.

    Parameters:
        df (pd.DataFrame): The original UNSW-NB15 DataFrame.

    Returns:
        pd.DataFrame: Modified DataFrame with new or transformed features.
    """
    import numpy as np

    # --- Feature 1: Packet size ratio ---
    df["pkt_size_ratio"] = df["sbytes"] / (df["dbytes"] + 1)  # avoid division by zero

    # --- Feature 2: Bytes per packet ---
    df["bytes_per_packet"] = (df["sbytes"] + df["dbytes"]) / (df["spkts"] + df["dpkts"] + 1)

    # --- Feature 3: Total packets ---
    df["total_pkts"] = df["spkts"] + df["dpkts"]

    # --- Feature 4: TTL difference (if available) ---
    if "sttl" in df.columns and "dttl" in df.columns:
        df["ttl_diff"] = np.abs(df["sttl"] - df["dttl"])

    # --- Feature 5: Time ratio ---
    if "sload" in df.columns and "dload" in df.columns:
        df["load_ratio"] = df["sload"] / (df["dload"] + 1)

    # --- Feature 6: Flow complexity ---
    if "ct_src_ltm" in df.columns and "ct_srv_dst" in df.columns:
        df["flow_complexity"] = df["ct_src_ltm"] + df["ct_srv_dst"]

    # --- Feature 7: Duration normalized ---
    if "dur" in df.columns:
        df["log_duration"] = np.log1p(df["dur"])

    # --- Feature 8: Protocol-State Combo ---
    if "proto" in df.columns and "state" in df.columns:
        df["proto_state"] = df["proto"].astype(str) + "_" + df["state"].astype(str)

    return df