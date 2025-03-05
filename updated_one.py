import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load trained model for car price prediction (if applicable)
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
except FileNotFoundError:
    model = None

# Streamlit app title
st.title("Universal CSV Forecasting and Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Handle missing values
    df.dropna(how='all', inplace=True)  # Drop empty rows
    df.dropna(axis=1, how='all', inplace=True)  # Drop empty columns
    
    # Detect date column dynamically
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
        st.sidebar.write("No valid date column found. Using index as time series.")
        df.reset_index(inplace=True)
        date_col = 'index'
    else:
        st.sidebar.write(f"Detected Date Column: {date_col}")

    df = df.sort_values(by=date_col)
    df = df.dropna(subset=[date_col])  # Drop rows with NaT values

    # If no valid data, exit
    if df.empty:
        st.sidebar.write("No valid date entries found. Forecasting cannot proceed.")
    else:
        # Select target column
        target_col = st.sidebar.selectbox("Select the Target Column (to forecast):", df.select_dtypes(include=[np.number]).columns)

        if target_col:
            fig, ax = plt.subplots()
            ax.plot(df[date_col], df[target_col], label="Actual Data")
            plt.xticks(rotation=45)
            plt.legend()
            st.sidebar.pyplot(fig)

            # Automatic Model Selection
            try:
                model = ARIMA(df[target_col], order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=10)
                selected_model = "ARIMA"
            except:
                df["timestamp"] = (df[date_col] - df[date_col].min()).dt.days
                X = df[["timestamp"]]
                y = df[target_col]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                last_valid_date = df[date_col].dropna().iloc[-1]
                if pd.isna(last_valid_date):
                    st.sidebar.write("Invalid last date for forecasting.")
                else:
                    future_dates = pd.date_range(last_valid_date, periods=10, freq="D")
                    future_timestamps = (future_dates - df[date_col].min()).days.values.reshape(-1, 1)
                    forecast = model.predict(future_timestamps)
                    selected_model = "Random Forest"

            if 'forecast' in locals():
                st.sidebar.write(f"### Forecasting with {selected_model}")
                forecast_dates = pd.date_range(df[date_col].iloc[-1], periods=10, freq="D")
                fig, ax = plt.subplots()
                ax.plot(df[date_col], df[target_col], label="Actual Data")
                ax.plot(forecast_dates, forecast, label="Forecast", color="red")
                plt.xticks(rotation=45)
                plt.legend()
                st.sidebar.pyplot(fig)
