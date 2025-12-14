#!/usr/bin/env python3
"""
engagement.py — Task 2 (Full version)
- Loads the TellCo xDR dataset (CSV)
- Computes top handsets & manufacturers
- Aggregates per-user per-app metrics
- Computes engagement score
- Writes engagement_results.csv
- Prints top / bottom engaged users for quick review
"""

import os
import sys
import pandas as pd
import numpy as np

# -------------------------
# CONFIG — update if needed
# -------------------------
INPUT_CSV = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\telcom_data_fully_cleaned.csv"
OUTPUT_CSV = os.path.join(os.path.dirname(INPUT_CSV), "engagement_results.csv")

# -------------------------
# Safe loader
# -------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"✔ Loaded {path} — shape: {df.shape}")
    return df

# -------------------------
# Helpers
# -------------------------
def ensure_columns_exist(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

# -------------------------
# Handset & manufacturer analysis (Task1 part)
# -------------------------
def top_handsets(df, top_n=10):
    col = "Handset Type"
    if col not in df.columns:
        print("⚠ Handset Type column not found.")
        return pd.Series(dtype=int)
    return df[col].value_counts().head(top_n)

def top_manufacturers_and_handsets(df, manu_n=3, handsets_per_manu=5):
    manu_col = "Handset Manufacturer"
    handset_col = "Handset Type"
    if manu_col not in df.columns or handset_col not in df.columns:
        print("⚠ Manufacturer/Handset columns not found.")
        return pd.DataFrame()
    top_manu = df[manu_col].value_counts().head(manu_n).index.tolist()
    out = {}
    for m in top_manu:
        top_h = df.loc[df[manu_col] == m, handset_col].value_counts().head(handsets_per_manu)
        out[m] = top_h
    return out

# -------------------------
# Per-user aggregation of app metrics (Task1.1 + Task2)
# -------------------------
def aggregate_user_app(df):
    # columns we expect for per-app bytes (these are from your dataset)
    app_cols = [
        "Social Media DL (Bytes)", "Social Media UL (Bytes)",
        "Google DL (Bytes)", "Google UL (Bytes)",
        "Email DL (Bytes)", "Email UL (Bytes)",
        "Youtube DL (Bytes)", "Youtube UL (Bytes)",
        "Netflix DL (Bytes)", "Netflix UL (Bytes)",
        "Gaming DL (Bytes)", "Gaming UL (Bytes)",
        "Other DL (Bytes)", "Other UL (Bytes)",
        "Total DL (Bytes)", "Total UL (Bytes)",
        "Activity Duration DL (ms)", "Activity Duration UL (ms)"
    ]

    # ensure columns exist (only warn for optional fields)
    present = [c for c in app_cols if c in df.columns]
    if "MSISDN/Number" not in df.columns:
        raise KeyError("Missing required column: 'MSISDN/Number'")

    # Compute per-row totals if needed
    # Session duration in seconds (safe fallback if columns missing)
    if "Activity Duration DL (ms)" in df.columns or "Activity Duration UL (ms)" in df.columns:
        df["session_duration_sec"] = 0
        if "Activity Duration DL (ms)" in df.columns:
            df["session_duration_sec"] += df["Activity Duration DL (ms)"].fillna(0)
        if "Activity Duration UL (ms)" in df.columns:
            df["session_duration_sec"] += df["Activity Duration UL (ms)"].fillna(0)
        df["session_duration_sec"] = df["session_duration_sec"] / 1000.0
    else:
        # if activity duration not present, try to use Dur. (ms)
        if "Dur. (ms)" in df.columns:
            df["session_duration_sec"] = df["Dur. (ms)"].fillna(0) / 1000.0
        else:
            df["session_duration_sec"] = 0.0

    # Per-row total traffic (bytes)
    if "Total DL (Bytes)" in df.columns or "Total UL (Bytes)" in df.columns:
        df["row_total_bytes"] = 0
        if "Total DL (Bytes)" in df.columns:
            df["row_total_bytes"] += df["Total DL (Bytes)"].fillna(0)
        if "Total UL (Bytes)" in df.columns:
            df["row_total_bytes"] += df["Total UL (Bytes)"].fillna(0)
    else:
        # fallback sum of app columns
        df["row_total_bytes"] = 0
        for c in ["Social Media DL (Bytes)", "Social Media UL (Bytes)",
                  "Google DL (Bytes)", "Google UL (Bytes)",
                  "Email DL (Bytes)", "Email UL (Bytes)",
                  "Youtube DL (Bytes)", "Youtube UL (Bytes)",
                  "Netflix DL (Bytes)", "Netflix UL (Bytes)",
                  "Gaming DL (Bytes)", "Gaming UL (Bytes)",
                  "Other DL (Bytes)", "Other UL (Bytes)"]:
            if c in df.columns:
                df["row_total_bytes"] += df[c].fillna(0)

    # Per-user aggregation
    agg_dict = {
        "row_total_bytes": "sum",
        "session_duration_sec": "sum",
    }

    # also aggregate per-app bytes if present
    per_app_cols = []
    for prefix in ["Social Media", "Google", "Email", "Youtube", "Netflix", "Gaming", "Other"]:
        dl = f"{prefix} DL (Bytes)"
        ul = f"{prefix} UL (Bytes)"
        if dl in df.columns:
            agg_dict[dl] = "sum"; per_app_cols.append(dl)
        if ul in df.columns:
            agg_dict[ul] = "sum"; per_app_cols.append(ul)

    # groupby
    user_agg = df.groupby("MSISDN/Number").agg(agg_dict).reset_index()

    # keep helpful names
    user_agg.rename(columns={
        "row_total_bytes": "total_bytes",
        "session_duration_sec": "total_duration_sec"
    }, inplace=True)

    # compute total traffic in MB and simple engagement score
    user_agg["total_traffic_MB"] = user_agg["total_bytes"] / (1024 * 1024)

    # engagement_score: weighted combination (tunable)
    # - traffic (MB): 50% weight
    # - duration (sec): 30% weight (scaled)
    # - app diversity/usage: 20% weight (sum of per-app MB if available)
    traffic_component = user_agg["total_traffic_MB"]
    duration_component = user_agg["total_duration_sec"] / 3600.0  # convert to hours to keep scale reasonable

    app_component = 0
    app_mb_cols = []
    for c in per_app_cols:
        # convert per-app bytes aggregated into MB
        app_mb = user_agg[c] / (1024 * 1024)
        app_component = app_component + app_mb
        app_mb_cols.append(c)

    # If no per-app columns found, app_component stays 0
    user_agg["app_usage_MB"] = app_component

    # Normalize contributions before combining to avoid extreme dominance
    # small helper to scale series to 0-1 safely
    def minmax(s):
        if s.max() == s.min():
            return s - s  # zeros
        return (s - s.min()) / (s.max() - s.min())

    t_norm = minmax(traffic_component)
    d_norm = minmax(duration_component)
    a_norm = minmax(user_agg["app_usage_MB"])

    user_agg["engagement_score"] = 0.5 * t_norm + 0.3 * d_norm + 0.2 * a_norm

    # also produce decile segmentation based on total_duration_sec
    user_agg["duration_decile"] = pd.qcut(user_agg["total_duration_sec"].replace(0, np.nan).fillna(0) + 1e-9, 10, labels=False, duplicates="drop") + 1

    return user_agg

# -------------------------
# Reporting helpers
# -------------------------
def top_n_by_column(df, col, n=10):
    if col not in df.columns:
        return pd.DataFrame()
    return df.nlargest(n, col)

def bottom_n_by_column(df, col, n=10):
    if col not in df.columns:
        return pd.DataFrame()
    return df.nsmallest(n, col)

# -------------------------
# CLI / main
# -------------------------
def main(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    print("\n📌 Loading dataset...")
    df = load_data(input_csv)

    # Print available columns (short)
    print("Columns (sample):", df.columns.tolist()[:20])

    # Handset & manufacturers quick analysis
    print("\n===== TOP 10 HANDSETS =====")
    try:
        print(top_handsets(df, 10))
    except Exception as e:
        print("Could not compute top handsets:", e)

    print("\n===== TOP 3 MANUFACTURERS + TOP 5 HANDSETS EACH =====")
    try:
        manu_handsets = top_manufacturers_and_handsets(df, manu_n=3, handsets_per_manu=5)
        for manu, series in manu_handsets.items():
            print(f"\nManufacturer: {manu}")
            print(series)
    except Exception as e:
        print("Could not compute manufacturers/handsets:", e)

    # Aggregate per user
    print("\n📌 Aggregating user app & engagement metrics...")
    user_agg = aggregate_user_app(df)

    # Show top/bottom
    print("\n===== TOP 10 MOST ENGAGED USERS (by engagement_score) =====")
    print(top_n_by_column(user_agg, "engagement_score", 10)[["MSISDN/Number", "engagement_score", "total_traffic_MB"]])

    print("\n===== BOTTOM 10 LEAST ENGAGED USERS (by engagement_score) =====")
    print(bottom_n_by_column(user_agg, "engagement_score", 10)[["MSISDN/Number", "engagement_score", "total_traffic_MB"]])

    # Save results
    user_agg.to_csv(output_csv, index=False)
    print(f"\n✅ Saved engagement results to: {output_csv}")

    return user_agg

# -------------------------
# run as script
# -------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
