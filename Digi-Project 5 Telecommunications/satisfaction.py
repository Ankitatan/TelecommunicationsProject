#!/usr/bin/env python3
"""
satisfaction.py - Task 4: Customer Satisfaction Analytics

- Loads engagement_results.csv and experience_results.csv
- Computes engagement & experience scores as Euclidean distance to low-quality cluster centroids
- Computes satisfaction score = average(engagement_distance_norm, experience_distance_norm)
- Trains a regression model to predict satisfaction (RandomForest)
- Clusters users on (engagement_norm, experience_score)
- Saves results to CSV and exports to MySQL table
- Writes a model run log CSV for tracking
"""

import os
import sys
import time
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sqlalchemy import create_engine
import joblib

# ------------------------
# CONFIG - set from user input
# ------------------------
ENG_PATH = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\engagement_results.csv"
EXP_PATH = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\experience_results.csv"

OUTPUT_DIR = os.path.dirname(ENG_PATH)
SAT_OUT_CSV = os.path.join(OUTPUT_DIR, "satisfaction_results.csv")
MODEL_LOG_CSV = os.path.join(OUTPUT_DIR, "model_runs.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "satisfaction_model.joblib")

# MySQL credentials provided by user
MYSQL_HOST = ""   # leave blank for localhost or add host
MYSQL_USER = "root"
MYSQL_PASSWORD = "Tiger@123"
MYSQL_DB = "DBtelecom"
MYSQL_TABLE = "telecom"

# If your MySQL server is not local, set MYSQL_HOST above. If local use 'localhost' or empty.
if MYSQL_HOST == "":
    MYSQL_HOST = "localhost"

# ------------------------
# Helpers
# ------------------------
def load_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"✔ Loaded {os.path.basename(path)} — shape: {df.shape}")
    return df

def choose_present_columns(df, candidates):
    present = [c for c in candidates if c in df.columns]
    return present

def safe_create_engine(user, password, host, db):
    # using pymysql
    conn_str = f"mysql+pymysql://{user}:{password}@{host}/{db}"
    return create_engine(conn_str)

# ------------------------
# Step 1: Load datasets
# ------------------------
def prepare_inputs(eng_path, exp_path):
    eng = load_csv_safe(eng_path)
    exp = load_csv_safe(exp_path)

    # show columns
    print("Engagement columns:", eng.columns.tolist())
    print("Experience columns:", exp.columns.tolist())

    return eng, exp

# ------------------------
# Step 2: Engagement clustering then distance calculation
# ------------------------
def compute_engagement_distances(eng_df, k=3, random_state=42):
    # candidate features for engagement clustering
    candidates = ["engagement_score", "total_traffic_MB", "total_duration_sec", "app_usage_MB", "app_usage"]
    feat_cols = choose_present_columns(eng_df, candidates)

    if not feat_cols:
        # fallback: use engagement_score only
        if "engagement_score" not in eng_df.columns:
            raise KeyError("engagement_score not found in engagement dataframe.")
        feat_cols = ["engagement_score"]

    X = eng_df[feat_cols].fillna(0).astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(Xs)
    eng_df["_eng_cluster"] = clusters

    # Identify the least-engaged cluster by mean engagement_score (if present) else by centroid sum
    if "engagement_score" in eng_df.columns:
        cluster_means = eng_df.groupby("_eng_cluster")["engagement_score"].mean()
        least_cluster = cluster_means.idxmin()
    else:
        # fallback: pick cluster with smallest centroid norm
        centroid_norms = np.linalg.norm(kmeans.cluster_centers_, axis=1)
        least_cluster = int(np.argmin(centroid_norms))

    centroid = kmeans.cluster_centers_[least_cluster]

    # compute Euclidean distance in scaled space
    dists = np.linalg.norm(Xs - centroid, axis=1)

    eng_df["engagement_distance_raw"] = dists
    # normalize distances to 0-1
    eng_df["engagement_distance"] = MinMaxScaler().fit_transform(dists.reshape(-1,1)).flatten()

    print(f"✔ Engagement: used features {feat_cols}; least-engaged cluster = {least_cluster}")
    return eng_df, {"scaler": scaler, "kmeans": kmeans, "feat_cols": feat_cols, "least_cluster": int(least_cluster)}

# ------------------------
# Step 3: Experience clustering then distance calculation
# ------------------------
def compute_experience_distances(exp_df, k=3, random_state=42):
    # candidate features for experience clustering
    candidates = ["experience_score", "throughput", "rtt", "tcp_retrans"]
    feat_cols = choose_present_columns(exp_df, candidates)

    if not feat_cols:
        if "experience_score" not in exp_df.columns:
            raise KeyError("experience_score not found in experience dataframe.")
        feat_cols = ["experience_score"]

    X = exp_df[feat_cols].fillna(0).astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(Xs)
    exp_df["_exp_cluster"] = clusters

    # identify worst-experience cluster (lowest mean experience_score)
    if "experience_score" in exp_df.columns:
        cluster_means = exp_df.groupby("_exp_cluster")["experience_score"].mean()
        worst_cluster = cluster_means.idxmin()
    else:
        centroid_norms = np.linalg.norm(kmeans.cluster_centers_, axis=1)
        worst_cluster = int(np.argmin(centroid_norms))

    centroid = kmeans.cluster_centers_[worst_cluster]
    dists = np.linalg.norm(Xs - centroid, axis=1)

    exp_df["experience_distance_raw"] = dists
    exp_df["experience_distance"] = MinMaxScaler().fit_transform(dists.reshape(-1,1)).flatten()

    print(f"✔ Experience: used features {feat_cols}; worst-experience cluster = {worst_cluster}")
    return exp_df, {"scaler": scaler, "kmeans": kmeans, "feat_cols": feat_cols, "worst_cluster": int(worst_cluster)}

# ------------------------
# Step 4: Merge & compute satisfaction
# ------------------------
def merge_and_compute_satisfaction(eng_df, exp_df):
    # merge on MSISDN/Number
    if "MSISDN/Number" not in eng_df.columns or "MSISDN/Number" not in exp_df.columns:
        raise KeyError("MSISDN/Number missing in one of the inputs.")
    merged = pd.merge(eng_df, exp_df[["MSISDN/Number", "experience_score", "experience_distance", "experience_distance_raw"]], on="MSISDN/Number", how="inner")

    # ensure engagement_distance exists
    if "engagement_distance" not in merged.columns:
        # try to bring from eng_df
        if "engagement_distance" in eng_df.columns:
            merged["engagement_distance"] = eng_df.set_index("MSISDN/Number")["engagement_distance"].reindex(merged["MSISDN/Number"]).values
        else:
            raise KeyError("engagement_distance missing; run engagement distance step first.")

    # compute satisfaction as average of normalized distances
    merged["satisfaction_score"] = (merged["engagement_distance"].fillna(0) + merged["experience_distance"].fillna(0)) / 2.0

    return merged

# ------------------------
# Step 5: Regression model to predict satisfaction
# ------------------------
def train_regression_model(df, target_col="satisfaction_score"):
    # choose features: use numeric columns that are likely useful
    candidate_features = ["engagement_score", "total_traffic_MB", "total_duration_sec", "app_usage_MB",
                          "experience_score", "throughput", "rtt", "tcp_retrans",
                          "engagement_distance", "experience_distance"]
    features = [c for c in candidate_features if c in df.columns]

    if not features:
        raise KeyError("No valid features found for regression.")

    X = df[features].fillna(0).astype(float)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)

    print(f"✔ Regression trained. R2={r2:.4f}, RMSE={rmse:.6f}")

    # persist model and scaler info if needed
    joblib.dump({"model": model, "features": features}, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

    return model, {"r2": r2, "rmse": rmse, "features": features}

# ------------------------
# Step 6: KMeans on engagement+experience scores (k=2)
# ------------------------
def cluster_on_scores(df, k=2):
    cols = ["engagement_distance", "experience_distance"]
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        raise KeyError("Need both engagement_distance and experience_distance to cluster.")
    X = df[present].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["sat_cluster"] = kmeans.fit_predict(Xs)
    return df, kmeans

# ------------------------
# Step 7: Save to MySQL
# ------------------------
def export_to_mysql(df, host, user, password, database, table):
    try:
        engine = safe_create_engine(user, password, host, database)
        # Write to SQL (replace)
        df_to_write = df[["MSISDN/Number", "engagement_distance", "experience_distance", "satisfaction_score"]].copy()
        df_to_write.to_sql(table, con=engine, if_exists="replace", index=False)
        print(f"✔ Exported {len(df_to_write)} rows to MySQL table {database}.{table}")
        return True
    except Exception as e:
        print("❌ MySQL export failed:", e)
        return False

# ------------------------
# Step 8: Run everything
# ------------------------
def main():
    start_time = datetime.utcnow()
    run_id = str(uuid.uuid4())

    eng_df, exp_df = prepare_inputs(ENG_PATH, EXP_PATH)

    eng_df, eng_meta = compute_engagement_distances(eng_df, k=3)
    exp_df, exp_meta = compute_experience_distances(exp_df, k=3)

    merged = merge_and_compute_satisfaction(eng_df, exp_df)

    model, metrics = train_regression_model(merged)

    merged, sat_km = cluster_on_scores(merged, k=2)

    # Save final CSV
    merged.to_csv(SAT_OUT_CSV, index=False)
    print(f"✔ Saved satisfaction CSV → {SAT_OUT_CSV}")

    # Export to MySQL
    exported = export_to_mysql(merged, MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_TABLE)

    # Model run log
    end_time = datetime.utcnow()
    run_record = {
        "run_id": run_id,
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "eng_meta": json.dumps(eng_meta),
        "exp_meta": json.dumps(exp_meta),
        "model_metrics": json.dumps(metrics),
        "exported_to_mysql": exported
    }

    # append to CSV
    df_log = pd.DataFrame([run_record])
    if os.path.exists(MODEL_LOG_CSV):
        df_log.to_csv(MODEL_LOG_CSV, mode='a', header=False, index=False)
    else:
        df_log.to_csv(MODEL_LOG_CSV, index=False)

    print(f"✔ Model run logged → {MODEL_LOG_CSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ ERROR:", e)
        sys.exit(1)
