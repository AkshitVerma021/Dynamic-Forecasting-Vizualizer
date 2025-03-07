import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import chardet  
import hashlib
import os
import pickle  
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# 📌 Set Streamlit to full-page width
st.set_page_config(layout="wide")

# 🔥 App Title
st.title("📊 Dynamic Forecasting Visualizer")

# 🗄 Define local storage directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔍 Function to generate a unique file hash
def get_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

# 📥 Sidebar for File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    
    # 🔍 Detect file encoding
    raw_data = uploaded_file.read()
    encoding_type = chardet.detect(raw_data)["encoding"]
    uploaded_file.seek(0)
    
    df = pd.read_csv(uploaded_file, encoding=encoding_type, encoding_errors="replace")
    df.dropna(inplace=True)
    
    st.sidebar.write(f"Detected Encoding: `{encoding_type}`")
    
    # 📝 Display Dataset Preview
    st.write("### 📌 Dataset Preview")
    st.dataframe(df.head())

    # 🔎 Detect Date Column
    date_col = None
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].notna().sum() > 0:
                date_col = col
                break
        except Exception:
            continue

    if date_col is None:
        st.write("⚠ No valid date column found. Using index as time series.")
        df.reset_index(inplace=True)
        date_col = 'index'
    else:
        st.write(f"📅 Detected Date Column: `{date_col}`")

    df.dropna(subset=[date_col], inplace=True)
    df = df.sort_values(by=date_col)

    # 🎯 Select Target Column
    target_col = st.sidebar.selectbox("Select Target Column (to forecast):", df.columns)

    if not np.issubdtype(df[target_col].dtype, np.number):
        st.sidebar.write("🔢 Target column is categorical. Encoding it numerically.")
        df[target_col], _ = pd.factorize(df[target_col])

    # 📊 Initial Data Plot
    st.write("### 📈 Actual Data Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[date_col], df[target_col], label="📊 Actual Data", marker="o", linestyle="-")
    ax.set_title("Actual Data Trend", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

    # 🔮 Forecasting Preparation
    forecast_index = pd.date_range(pd.to_datetime(df[date_col].iloc[-1]) + pd.Timedelta(days=1), periods=10, freq="D")
    
    # 🔍 Model Selection (ARIMA / RandomForest)
    try:
        model = ARIMA(df[target_col], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
        st.write("### 🔮 Forecasting with ARIMA")
    except Exception:
        st.write("⚠ ARIMA failed. Switching to Random Forest.")
        try:
            df["timestamp"] = (df[date_col] - df[date_col].min()).dt.days
            X = df[["timestamp"]]
            y = df[target_col]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            future_timestamps = (forecast_index - df[date_col].min()).days.values.reshape(-1, 1)
            forecast = model.predict(future_timestamps)
            st.write("### 🌲 Forecasting with Random Forest")
        except Exception as e:
            st.write("❌ Both ARIMA and Random Forest failed.")
            st.write("Error:", str(e))
            forecast = None

    # 📉 Forecast Visualization
    if forecast is not None:
        st.write("### 📊 Forecast vs Actual Data")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df[date_col], df[target_col], label="📊 Actual Data", marker="o", linestyle="-")
        ax.plot(forecast_index, forecast, label="🔮 Forecast", marker="x", linestyle="--", color="red")
        ax.set_title("Forecast vs Actual Data", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(target_col, fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
