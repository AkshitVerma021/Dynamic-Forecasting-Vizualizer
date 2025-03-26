import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import json
import os
import chardet
import hashlib
# import speech_recognition as sr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from auth import init_session_state, check_auth, sign_out
from chatbot import chatbot_section  # ✅ Import Chatbot Section

# 📌 Set Streamlit Page Config
st.set_page_config(layout="wide")

def load_css(styles):
    with open(styles, "r") as f:
        css_styles = f.read()
        st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

load_css("styles.css")


# 🔥 App Title
st.title("📊 AI-Powered Data Analysis & Forecasting with Voice Chat")

# 🗄 Define Storage Directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
USER_DATA_FILE = "users.json"
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# 🔑 Initialize Session State
init_session_state()

# 🔐 Check Authentication
check_auth()

# ✅ User is Authenticated
st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")
if st.sidebar.button("🚪 Sign Out"):
    sign_out()

# 🔍 AWS Bedrock Client Initialization
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_client = get_bedrock_client()

# 📥 Sidebar for Multiple File Uploads with Progress Bar
st.sidebar.header("📂 Upload Your Datasets")

uploaded_files = st.sidebar.file_uploader(
    "Upload Files (CSV, Excel, JSON, Parquet)",
    type=["csv", "xls", "xlsx", "json", "parquet"],
    accept_multiple_files=True
)

# Progress Bar for File Upload
progress_bar = st.sidebar.progress(0)

# 📚 Load Uploaded Files
dataframes = []
file_names = []
if uploaded_files:
    total_files = len(uploaded_files)
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        st.sidebar.write(f"✅ {file_name} uploaded successfully.")
        progress_bar.progress((i + 1) / total_files)
        # Load CSV/Excel/JSON/Parquet
        raw_data = uploaded_file.read()
        encoding_type = chardet.detect(raw_data)["encoding"]
        uploaded_file.seek(0)
        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding=encoding_type, encoding_errors="replace")
            elif file_name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            elif file_name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            elif file_name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                st.sidebar.error(f"❌ Unsupported file format: {file_name}")
                continue
            dataframes.append(df)
            file_names.append(file_name)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {file_name}: {str(e)}")
            continue

    # Reset Progress Bar
    progress_bar.progress(0)

# 📚 Show Each Uploaded File Separately
if dataframes:
    for i, df in enumerate(dataframes):
        st.write(f"### 📊 Preview of `{file_names[i]}`")
        st.dataframe(df.head(50))

        # ✅ Detect Date Column
        date_col = None
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() > 0:
                    date_col = col
                    break
            except Exception:
                continue

        if date_col is None:
            st.write("⚠ No valid date column found. Using index as time series.")
            df.reset_index(inplace=True)
            date_col = "index"
        else:
            st.write(f"📅 Detected Date Column: `{date_col}`")

        df.dropna(subset=[date_col], inplace=True)
        df = df.sort_values(by=date_col)

        # ✅ Filter columns that can be used for prediction (numeric or convertible to numeric)
        valid_columns = []
        for col in df.columns:
            if col == date_col:
                continue  # Skip the date column
            try:
                # Convert to numeric with errors ignored to identify possible valid columns
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notnull().sum() > 0:  # Check if any valid values remain
                    valid_columns.append(col)
            except Exception:
                st.write(f"⚠ Column `{col}` cannot be used for prediction (non-numeric and non-convertible).")

        # ✅ Further filter columns to exclude those causing overflow errors
        safe_columns = []
        for col in valid_columns:
            try:
                # Test if the column causes overflow during forecasting
                min_date = df[date_col].min()
                test_timestamps = (df[date_col] - min_date).dt.days.astype(float)
                test_X = test_timestamps.values.reshape(-1, 1)
                test_y = df[col].astype(float)

                # Test Random Forest model (skip if it fails)
                model = RandomForestRegressor(n_estimators=10, random_state=42)  # Use a smaller model for testing
                model.fit(test_X, test_y)
                safe_columns.append(col)
            except Exception as e:
                st.write(f"")

        # Show only safe columns in the dropdown
        if safe_columns:
            target_col = st.selectbox(f"Select Target Column for `{file_names[i]}`:", safe_columns, key=f"target_col_{i}")
        else:
            st.error(f"❌ No valid target columns found in `{file_names[i]}`. All columns caused errors.")
            continue

        # 📊 Forecasting Preparation
        forecast_index = pd.date_range(pd.to_datetime(df[date_col].iloc[-1]) + pd.Timedelta(days=1), periods=10, freq="D")

        # 🔥 Try ARIMA, Else Use Random Forest
        try:
            model = ARIMA(df[target_col], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)
            st.write("")
        except Exception:
            try:
                # Use relative timestamps (days since the first date)
                min_date = df[date_col].min()
                df["timestamp"] = (df[date_col] - min_date).dt.days
                X = df[["timestamp"]]
                y = df[target_col]

                # Convert to float to avoid overflow
                X = X.astype(float)
                y = y.astype(float)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                # Prepare future timestamps
                future_timestamps = (forecast_index - min_date).days.values.reshape(-1, 1)
                future_timestamps = future_timestamps.astype(float)

                forecast = model.predict(future_timestamps)
                st.write(f"")
            except Exception as e:
                st.write(f"❌ Forecasting failed for `{file_names[i]}`.")
                st.write("Error:", str(e))
                forecast = None

        # 📊 Forecast Visualization
        if forecast is not None:
            st.write(f"### 📊 Forecast vs Actual Data for `{file_names[i]}`")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df[date_col], df[target_col], label="📊 Actual Data", marker="o", linestyle="-")
            ax.plot(forecast_index, forecast, label="🔮 Forecast", marker="x", linestyle="--", color="red")
            ax.set_title(f"Forecast vs Actual Data for `{file_names[i]}`", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(target_col, fontsize=12)
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
            
st.write("")
chatbot_section(dataframes, file_names, bedrock_client)
