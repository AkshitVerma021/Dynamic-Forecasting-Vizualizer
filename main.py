import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import json
import os
import chardet
import hashlib
import pdfplumber
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from auth import init_session_state, check_auth, sign_out
from chatbot import chatbot_section  # ✅ Import Chatbot Section
from prophet import Prophet
from plotly import graph_objs as go

# 📌 Set Streamlit Page Config (MUST BE FIRST!)
st.set_page_config(page_title="AI-Powered Data Analysis", layout="centered")

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
col1, col2, col3 = st.columns([1, 1, 1])  # Use 1:2:1 ratio for perfect centering
with col2:
    st.image("logo.png", width=350)

# 📚 Load CSS for Custom Styling
def load_css(styles):
    with open(styles, "r") as f:
        css_styles = f.read()
        st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

load_css("styles.css")

# 🔥 App Title
st.title("📊 AI-Powered Data Analysis ")

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

# Add custom text to show 5MB limit
st.sidebar.markdown(
    """
    <div style="color: #888888; font-size: 14px; margin-bottom: 10px;">
    📏 <b>Limit 5MB per file</b> • CSV, XLS, XLSX, JSON, PARQUET, PDF
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_files = st.sidebar.file_uploader(
    "Upload Files (CSV, Excel, JSON, Parquet, PDF)",
    type=["csv", "xls", "xlsx", "json", "parquet", "pdf"],  
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
        file_size = len(uploaded_file.getvalue())  # Get file size in bytes
        
        # 📏 Check File Size (Limit: 5MB)
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            st.sidebar.error(f"❌ {file_name} exceeds 5MB size limit. Skipping file.")
            continue
        st.sidebar.write(f"✅ {file_name} uploaded successfully.")
        progress_bar.progress((i + 1) / total_files)

        # Load CSV/Excel/JSON/Parquet/PDF
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
            elif file_name.endswith(".pdf"):
                # 📄 Extract Text from PDF Using pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()

                # 📄 Convert PDF Text to DataFrame (Line by Line)
                lines = text.split("\n")
                df = pd.DataFrame({"Text": lines})
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

# ✅ Dropdown to Select File and View Option
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
        if selected_file.endswith(".pdf"):
            st.text_area("📄 PDF Content", "\n".join(selected_df["Text"].tolist()), height=400)
        else:
            st.dataframe(selected_df.head(50))

    elif option == "📈 Chart":
        # Replace the existing date column detection code with this enhanced version
        date_column_keywords = ["year", "date", "age", "month", "time", "period", "quarter"]
        numeric_time_indicators = ["year", "age", "period","Year","Age"]
        
        # Initialize date_col as None
        date_col = None
        
        # First try to find datetime columns
        for col in selected_df.columns:
            try:
                col_lower = col.lower()
                
                # Check if column name contains date-related keywords
                if any(keyword in col_lower for keyword in date_column_keywords):
                    # For numeric columns that represent time (like year or age)
                    if any(indicator in col_lower for indicator in numeric_time_indicators):
                        if selected_df[col].dtype in ['int64', 'float64']:
                            date_col = col
                            # Convert numeric years to datetime if it's a year column
                            if "year" in col_lower:
                                selected_df[col] = pd.to_datetime(selected_df[col], format='%Y')
                            break
                    else:
                        # Try converting to datetime for date-like columns
                        selected_df[col] = pd.to_datetime(selected_df[col], errors="coerce")
                        if selected_df[col].notna().sum() > 0:
                            date_col = col
                            break
            except Exception:
                continue
        
        # If no date column found, look for numeric columns that could represent time series
        if date_col is None:
            numeric_cols = selected_df.select_dtypes(include=[np.number]).columns
            
            # Check if we have any numeric columns that could represent time periods
            for col in numeric_cols:
                col_lower = col.lower()
                # Check if the column is sorted and has regular intervals
                if selected_df[col].is_monotonic_increasing:
                    date_col = col
                    try:
                        # Create a datetime index with daily frequency instead of yearly
                        # This prevents the date from growing too large
                        selected_df[col] = pd.date_range(
                            start='2000-01-01',  # arbitrary start date
                            periods=len(selected_df),
                            freq='D'  # daily frequency instead of yearly
                        )
                        break
                    except pd.errors.OutOfBoundsDatetime:
                        # If even daily frequency is too large, use a smaller dataset
                        st.warning(f"Dataset too large to create date range. Using first 1000 rows.")
                        selected_df = selected_df.head(1000)
                        selected_df[col] = pd.date_range(
                            start='2000-01-01',
                            periods=len(selected_df),
                            freq='D'
                        )
                        break
        
        # Now proceed with forecasting if we found a suitable column
        if date_col is None:
            st.warning("⚠ No suitable time-based column found. Please ensure your data has a column representing time (date, year, age, or sequential periods).")
        else:
            # st.write(f"📅 Using column for time series analysis: `{date_col}`")
        
            numeric_columns = selected_df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                st.error(f"❌ No numeric columns available for forecasting in `{selected_file}`.")
            else:
                target_col = st.selectbox(f"Select Target Column for `{selected_file}`:", numeric_columns)

                # 📊 Forecasting Preparation
                # Use the last available date from the dataset to start the forecasting
                last_date = selected_df[date_col].max()  # Get the most recent date from the dataset
                
                # Check if the last_date is valid and ensure it's in datetime format
                if pd.to_datetime(last_date, errors="coerce") is pd.NaT:
                    st.error("❌ The last date in the dataset is invalid or missing.")
                else:
                    # Generate future forecast index starting from the last available date in the dataset
                    forecast_index = pd.date_range(
                        start=last_date,  # Use the last date from the uploaded data
                        periods=10,  # Forecast for the next 10 years (adjust as necessary)
                        freq="Y"  # Yearly frequency for forecasting
                    )

                    # 🔥 Try Prophet Forecasting, Else Use Random Forest
                    forecast = None
                    try:
                        # Prepare data for Prophet (requires 'ds' and 'y' columns)
                        prophet_df = pd.DataFrame({
                            'ds': selected_df[date_col],
                            'y': selected_df[target_col]
                        })
                        
                        # Initialize and fit Prophet model
                        model = Prophet(yearly_seasonality=True)
                        model.fit(prophet_df)
                        
                        # Create future dates dataframe
                        future = model.make_future_dataframe(periods=15, freq='Y')
                        
                        # Make predictions
                        forecast = model.predict(future)
                        
                        # Calculate metrics on training data
                        predictions = forecast['yhat'][:len(selected_df)]
                        actuals = prophet_df['y']
                    
                        # Visualization
                        fig = model.plot(forecast)
                        
                        # Customize the plot
                        ax = fig.gca()
                        ax.set_title(f"Forecast vs Actual Data for `{selected_file}`", size=14)
                        ax.set_xlabel("Year", size=12)
                        ax.set_ylabel(target_col, size=12)
                        ax.tick_params(axis="x", labelsize=10)
                        ax.tick_params(axis="y", labelsize=10)
                        
                        st.pyplot(fig)
                        
                        # You can also show the components plot
                        components_fig = model.plot_components(forecast)
                        st.write("### 📊 Forecast Components")
                        st.pyplot(components_fig)
                    
                    except Exception as e:
                        st.write("Prophet model failed, switching to Random Forest.")
                        try:
                            # Random Forest Forecasting
                            min_date = selected_df[date_col].min()
                            selected_df["timestamp"] = (selected_df[date_col] - min_date).dt.days
                            X = selected_df[["timestamp"]]
                            y = selected_df[target_col]

                            # Random Forest Model
                            model = RandomForestRegressor(n_estimators=150, random_state=42)
                            model.fit(X, y)

                            # Future Timestamps
                            future_timestamps = (forecast_index - min_date).days.values.reshape(-1, 1)
                            forecast = model.predict(future_timestamps)
                            st.write("Random Forest model forecast successful!")
                        except Exception as e:
                            st.write(f"❌ Forecasting failed for `{selected_file}`.")
                            # st.write("Error:", str(e))


st.write("")
chatbot_section(dataframes, file_names, bedrock_client)
