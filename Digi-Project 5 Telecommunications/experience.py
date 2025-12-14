# experience.py — Task 3: Customer Experience Analytics
# ------------------------------------------------------
# This script computes:
#  - Network experience metrics (RTT, Throughput, TCP Retransmission)
#  - Device information
#  - Experience Score
#  - KMeans clustering for customer experience
#  - Output CSV: experience_results.csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"✔ Loaded cleaned telecom dataset — shape: {df.shape}")
    return df


# ------------------------------------------------------
# 2. COMPUTE EXPERIENCE METRICS (Task 3.1)
# ------------------------------------------------------
def compute_experience_metrics(df):

    print("✔ Computing experience metrics...")

    agg = df.groupby("MSISDN/Number").agg({
        "TCP DL Retrans. Vol (Bytes)": "mean",
        "TCP UL Retrans. Vol (Bytes)": "mean",
        "Avg RTT DL (ms)": "mean",
        "Avg RTT UL (ms)": "mean",
        "Avg Bearer TP DL (kbps)": "mean",
        "Avg Bearer TP UL (kbps)": "mean",
        "Handset Type": "first"
    }).reset_index()

    # Combine UL + DL metrics
    agg["tcp_retrans"] = agg["TCP DL Retrans. Vol (Bytes)"] + agg["TCP UL Retrans. Vol (Bytes)"]
    agg["rtt"] = agg["Avg RTT DL (ms)"] + agg["Avg RTT UL (ms)"]
    agg["throughput"] = agg["Avg Bearer TP DL (kbps)"] + agg["Avg Bearer TP UL (kbps)"]

    # Drop raw columns
    agg = agg[[
        "MSISDN/Number",
        "tcp_retrans",
        "rtt",
        "throughput",
        "Handset Type"
    ]]

    print("✔ Experience metrics computed successfully")
    return agg


# ------------------------------------------------------
# 3. COMPUTE EXPERIENCE SCORE
# ------------------------------------------------------
def compute_experience_score(agg):
    print("✔ Computing experience_score...")

    scaler = MinMaxScaler()

    agg["throughput_norm"] = scaler.fit_transform(agg[["throughput"]])
    agg["rtt_norm"] = scaler.fit_transform(agg[["rtt"]])
    agg["tcp_norm"] = scaler.fit_transform(agg[["tcp_retrans"]])

    # Higher throughput = better
    # Lower RTT + retransmission = better
    agg["experience_score"] = (
        agg["throughput_norm"] * 0.6 -
        agg["rtt_norm"] * 0.2 -
        agg["tcp_norm"] * 0.2
    )

    print("✔ Added experience_score successfully")
    return agg


# ------------------------------------------------------
# 4. CLUSTER USERS (Task 3.2)
# ------------------------------------------------------
def cluster_experience(agg, k=3):

    print("✔ Performing KMeans clustering for Experience...")

    X = agg[["throughput", "rtt", "tcp_retrans"]]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["experience_cluster"] = kmeans.fit_predict(X_scaled)

    # Human-readable cluster labels
    cluster_labels = {
        0: "Good Experience",
        1: "Normal Experience",
        2: "Poor Experience"
    }

    agg["experience_label"] = agg["experience_cluster"].map(cluster_labels)

    print("✔ Experience clusters assigned")
    return agg


# ------------------------------------------------------
# 5. SAVE RESULTS
# ------------------------------------------------------
def save_results(agg, save_path):
    agg.to_csv(save_path, index=False)
    print(f"✔ experience_results.csv saved to:\n{save_path}")


# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------
def main():

    input_path = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\telcom_data_fully_cleaned.csv"
    save_path = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\experience_results.csv"

    df = load_data(input_path)
    agg = compute_experience_metrics(df)
    agg = compute_experience_score(agg)
    agg = cluster_experience(agg)

    save_results(agg, save_path)

    print("\n🎉 Task 3 — experience.py completed successfully!")


if __name__ == "__main__":
    main()
