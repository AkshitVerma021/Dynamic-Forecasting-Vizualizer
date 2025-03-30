import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import json
import os
import hashlib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from auth import init_session_state, check_auth, sign_out
from chatbot import chatbot_section  # ✅ Import Chatbot Section

# 📌 Set Streamlit Page Config (MUST BE FIRST!)
st.set_page_config(page_title="AI-Powered Data Analysis & Forecasting", layout="centered")

# 📸 Add Company Logo at the Top
st.markdown(
    """
    <style>
    .center-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📸 Add Logo as Image (Centered Properly)
col1, col2, col3 = st.columns([1, 1, 1])  # Center the logo
with col2:
    st.image("logo.png", width=350)

# 🔥 App Title
st.title("📊 AI-Powered Data Analysis")

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

# 📍 AWS Bedrock Client Initialization
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

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

        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            elif file_name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            elif file_name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                st.sidebar.error(f"❌ Unsupported file format: {file_name}")
                continue

            if df.empty:
                st.sidebar.error(f"⚠️ {file_name} is empty or has invalid data.")
                continue

            dataframes.append(df)
            file_names.append(file_name)

        except Exception as e:
            st.sidebar.error(f"❌ Error loading {file_name}: {str(e)}")

    # Reset Progress Bar
    progress_bar.progress(0)

# 📂 Show Dropdown if Files are Uploaded
if dataframes:
    selected_file = st.selectbox(
        "📂 Select a file to view",
        file_names,
        index=0
    )

    # Get the corresponding dataframe
    selected_df = dataframes[file_names.index(selected_file)]

    # 📊 Choose Between Preview and Chart
    option = st.selectbox(
        f"📊 Select View for `{selected_file}`",
        ["📋 Preview", "📈 Chart"]
    )

    if option == "📋 Preview":
        st.write(f"### 📋 Preview of `{selected_file}`")
        st.dataframe(selected_df.head(50))

    # ✅ Detect Date Column for Forecasting
    date_col = None
    for col in selected_df.columns:
        try:
            selected_df[col] = pd.to_datetime(selected_df[col], errors="coerce")
            if selected_df[col].notna().sum() > 0:
                date_col = col
                break
        except Exception:
            continue

    if date_col is None:
        st.error("⚠️ No valid date column found. Using index as time series.")
        selected_df.reset_index(inplace=True)
        date_col = "index"
    else:
        selected_df.dropna(subset=[date_col], inplace=True)
        selected_df = selected_df.sort_values(by=date_col)

    # ✅ Filter Numeric Columns for Forecasting
    valid_columns = []
    for col in selected_df.columns:
        if col == date_col:
            continue
        try:
            selected_df[col] = pd.to_numeric(selected_df[col], errors="coerce")
            if selected_df[col].notnull().sum() > 0:
                valid_columns.append(col)
        except Exception:
            st.write(f"⚠️ Column `{col}` cannot be used for prediction (non-numeric).")

    # ✅ Safe Columns Check
    safe_columns = []
    for col in valid_columns:
        try:
            min_date = selected_df[date_col].min()
            test_timestamps = (selected_df[date_col] - min_date).dt.days.astype(float)
            test_X = test_timestamps.values.reshape(-1, 1)
            test_y = selected_df[col].astype(float)

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(test_X, test_y)
            safe_columns.append(col)
        except Exception:
            continue

    # Show Safe Columns Dropdown
    if safe_columns:
        target_col = st.selectbox(
            f"Select Target Column for `{selected_file}`:",
            safe_columns,
            key=f"target_col_{selected_file}"
        )
    else:
        st.error(f"❌ No valid target columns found in `{selected_file}`.")
        target_col = None

    # 📊 Forecasting Preparation
    forecast_index = pd.date_range(
        pd.to_datetime(selected_df[date_col].iloc[-1]) + pd.Timedelta(days=1),
        periods=10,
        freq="D"
    )

    # 🔥 Try ARIMA, Else Use Random Forest
    forecast = None
    if target_col:
        try:
            # Try ARIMA Model
            model = ARIMA(selected_df[target_col], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)
            st.success(f"✅ ARIMA Model Successfully Predicted!")
        except Exception:
            try:
                # Use Random Forest as Fallback
                selected_df["timestamp"] = (selected_df[date_col] - selected_df[date_col].min()).dt.days
                X = selected_df[["timestamp"]].astype(float)
                y = selected_df[target_col].astype(float)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                # Prepare Future Timestamps
                future_timestamps = (forecast_index - selected_df[date_col].min()).days.values.reshape(-1, 1).astype(float)
                forecast = model.predict(future_timestamps)

                st.success(f"✅ Random Forest Model Used as Fallback!")
            except Exception:
                st.error(f"❌ Both ARIMA and Random Forest models failed to forecast.")

    # 📈 Chart Section for Forecasting
    if option == "📈 Chart" and forecast is not None:
        st.write(f"### 📈 Forecast vs Actual Data for `{selected_file}`")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(selected_df[date_col], selected_df[target_col], label="📊 Actual Data", marker="o", linestyle="-")
        ax.plot(forecast_index, forecast, label="🔮 Forecast", marker="x", linestyle="--", color="red")
        ax.set_title(f"Forecast vs Actual Data for `{selected_file}`", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(target_col, fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

# 🤖 Chatbot Section
st.write("")
chatbot_section(dataframes, file_names, bedrock_client)
