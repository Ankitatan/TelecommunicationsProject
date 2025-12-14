import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="TellCo Telecom Dashboard", layout="wide")

# -----------------------------
# Default paths (change if needed)
# -----------------------------
DEFAULT_RAW = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\telcom_data_fully_cleaned.csv"
DEFAULT_ENG = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\engagement_results.csv"
DEFAULT_EXP = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\experience_results.csv"
DEFAULT_SAT = r"D:\Courses\DIGICROME\NextHikesINTERNSHIP\Digi-Project 5 Telecommunications\satisfaction_results.csv"
# Local uploaded python file (provided in this session):
SAMPLE_PY_FILE = "/mnt/data/b11ac1fa-a4b3-4c5e-8c30-780394e6305c.py"

# -----------------------------
# Sidebar: data selection
# -----------------------------
st.sidebar.title("Data sources")
use_upload = st.sidebar.checkbox("Upload CSVs manually (instead of default paths)", value=False)

if use_upload:
    raw_file = st.sidebar.file_uploader("Upload raw telco CSV (xDR) ", type=["csv"]) 
    eng_file = st.sidebar.file_uploader("Upload engagement_results.csv", type=["csv"]) 
    exp_file = st.sidebar.file_uploader("Upload experience_results.csv", type=["csv"]) 
    sat_file = st.sidebar.file_uploader("Upload satisfaction_results.csv", type=["csv"]) 
else:
    raw_file = None
    eng_file = None
    exp_file = None
    sat_file = None

# -----------------------------
# Load functions
# -----------------------------
@st.cache_data
def load_csv(path_or_buffer):
    if hasattr(path_or_buffer, "read"):
        return pd.read_csv(path_or_buffer)
    return pd.read_csv(path_or_buffer)

# Try to load datasets (prefer uploaded)
try:
    if eng_file is not None:
        eng_df = load_csv(eng_file)
    else:
        eng_df = load_csv(DEFAULT_ENG)
except Exception as e:
    st.sidebar.error(f"Couldn't load engagement file: {e}")
    eng_df = pd.DataFrame()

try:
    if exp_file is not None:
        exp_df = load_csv(exp_file)
    else:
        exp_df = load_csv(DEFAULT_EXP)
except Exception as e:
    st.sidebar.error(f"Couldn't load experience file: {e}")
    exp_df = pd.DataFrame()

try:
    if sat_file is not None:
        sat_df = load_csv(sat_file)
    else:
        sat_df = load_csv(DEFAULT_SAT)
except Exception as e:
    st.sidebar.info("Satisfaction file not found — you can compute it by running satisfaction.py")
    sat_df = pd.DataFrame()

try:
    if raw_file is not None:
        raw_df = load_csv(raw_file)
    else:
        raw_df = load_csv(DEFAULT_RAW)
except Exception as e:
    st.sidebar.info("Raw xDR file not loaded: " + str(e))
    raw_df = pd.DataFrame()

# -----------------------------
# Header
# -----------------------------
st.title("📶 TellCo — User Analytics Dashboard")
st.markdown("A compact dashboard for User Overview, Engagement, Experience, and Satisfaction analyses.")

col1, col2, col3 = st.columns(3)

# KPIs from engagement
with col1:
    if not eng_df.empty:
        total_users = eng_df.shape[0]
        avg_eng = eng_df['engagement_score'].mean() if 'engagement_score' in eng_df.columns else None
        st.metric("Total users (in engagement)", f"{total_users}")
        if avg_eng is not None:
            st.metric("Avg engagement score", f"{avg_eng:.4f}")
    else:
        st.metric("Total users (in engagement)", "-")

with col2:
    if not exp_df.empty:
        avg_exp = exp_df['experience_score'].mean() if 'experience_score' in exp_df.columns else None
        st.metric("Avg experience score", f"{avg_exp:.4f}" if avg_exp is not None else "-")
    else:
        st.metric("Avg experience score", "-")

with col3:
    if not sat_df.empty:
        avg_sat = sat_df['satisfaction_score'].mean() if 'satisfaction_score' in sat_df.columns else None
        st.metric("Avg satisfaction score", f"{avg_sat:.4f}" if avg_sat is not None else "-")
    else:
        st.metric("Avg satisfaction score", "-")

# -----------------------------
# User Overview Page
# -----------------------------
st.header("User Overview")
if not raw_df.empty:
    st.subheader("Top 10 Handsets")
    if 'Handset Type' in raw_df.columns:
        top_handsets = raw_df['Handset Type'].value_counts().head(10)
        fig = px.bar(x=top_handsets.values, y=top_handsets.index, orientation='h', labels={'x':'count','y':'Handset Type'}, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Handset Type column not found in raw data.")

    st.subheader("Top 3 Manufacturers and their Top-5 Handsets")
    if 'Handset Manufacturer' in raw_df.columns and 'Handset Type' in raw_df.columns:
        manu = raw_df['Handset Manufacturer'].value_counts().head(3).index.tolist()
        for m in manu:
            st.markdown(f"**{m}**")
            top5 = raw_df[raw_df['Handset Manufacturer']==m]['Handset Type'].value_counts().head(5)
            st.write(top5)
    else:
        st.info("Manufacturer/Handset columns not found.")
else:
    st.info("Raw xDR data not loaded — user overview limited.")

# -----------------------------
# Per-App Behaviour
# -----------------------------
st.header("Per-App Behaviour")
if not eng_df.empty:
    st.subheader("Top apps by total traffic")
    app_cols = [c for c in eng_df.columns if any(prefix in c for prefix in ['Social Media','Google','Email','Youtube','Netflix','Gaming','Other'])]
    if app_cols:
        app_sums = eng_df[app_cols].sum().sort_values(ascending=False)
        fig2 = px.bar(x=app_sums.values, y=app_sums.index, orientation='h', labels={'x':'bytes','y':'application'}, height=360)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No per-app columns found in engagement results.")
else:
    st.info("Engagement results not loaded.")

# -----------------------------
# Engagement Clusters
# -----------------------------
st.header("User Engagement Clusters")
if not eng_df.empty:
    # try to plot using first two numeric features
    numeric_cols = eng_df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in ['engagement_score','total_traffic_MB','total_duration_sec','app_usage_MB'] if c in numeric_cols]
    if len(candidates) >= 2:
        xcol, ycol = candidates[0], candidates[1]
        fig3 = px.scatter(eng_df, x=xcol, y=ycol, color=eng_df.get('cluster', None), hover_data=['MSISDN/Number'], height=500)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough numeric columns for cluster scatter plot (engagement).")
else:
    st.info("Engagement data not available for clustering view.")

# -----------------------------
# Experience Analytics
# -----------------------------
st.header("User Experience")
if not exp_df.empty:
    cols = [c for c in ['throughput','rtt','tcp_retrans','experience_score','experience_label'] if c in exp_df.columns]
    st.write(exp_df[cols].describe())

    if 'experience_label' in exp_df.columns:
        fig4 = px.histogram(exp_df, x='experience_label', title='Experience Label Distribution')
        st.plotly_chart(fig4, use_container_width=True)

    # Throughput by handset (top 10 handsets)
    if 'Handset Type' in exp_df.columns and 'throughput' in exp_df.columns:
        top_handset_thr = exp_df.groupby('Handset Type')['throughput'].mean().sort_values(ascending=False).head(10)
        fig5 = px.bar(x=top_handset_thr.values, y=top_handset_thr.index, orientation='h', labels={'x':'throughput (kbps)','y':'Handset'}, height=420)
        st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Experience results not available.")

# -----------------------------
# Satisfaction View
# -----------------------------
st.header("User Satisfaction")
if not sat_df.empty:
    st.subheader("Top 10 Satisfied Customers")
    if 'satisfaction_score' in sat_df.columns:
        st.write(sat_df[['MSISDN/Number','satisfaction_score']].nlargest(10,'satisfaction_score'))
        fig6 = px.histogram(sat_df, x='satisfaction_score', nbins=50, title='Satisfaction Score Distribution')
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("satisfaction_score column missing from satisfaction file.")
else:
    st.info("Satisfaction results not loaded. Run satisfaction.py to generate them.")

# -----------------------------
# PCA & Dimensionality Reduction quick view
# -----------------------------
st.header("Dimensionality Reduction (PCA) — Apps")
if not eng_df.empty:
    app_cols_all = [c for c in eng_df.columns if any(prefix in c for prefix in ['Social Media','Google','Email','Youtube','Netflix','Gaming','Other'])]
    if len(app_cols_all) >= 2:
        X = eng_df[app_cols_all].fillna(0).astype(float)
        scaler = MinMaxScaler(); Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(Xs)
        pca_df = pd.DataFrame(pcs, columns=['PC1','PC2'])
        pca_df['MSISDN/Number'] = eng_df['MSISDN/Number'].values
        fig7 = px.scatter(pca_df, x='PC1', y='PC2', hover_data=['MSISDN/Number'], height=500)
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Not enough per-app columns for PCA.")

# -----------------------------
# Download links + sample file info
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Session assets**")
st.sidebar.write(f"Sample Python file available at: {SAMPLE_PY_FILE}")

if not eng_df.empty:
    st.sidebar.download_button("Download engagement_results.csv", data=eng_df.to_csv(index=False), file_name="engagement_results.csv", mime="text/csv")
if not exp_df.empty:
    st.sidebar.download_button("Download experience_results.csv", data=exp_df.to_csv(index=False), file_name="experience_results.csv", mime="text/csv")
if not sat_df.empty:
    st.sidebar.download_button("Download satisfaction_results.csv", data=sat_df.to_csv(index=False), file_name="satisfaction_results.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("Data source defaults to your project folder. Change via uploads if needed.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built for TellCo — Nexthikes internship project. Scripts available: engagement.py, experience.py, satisfaction.py")
