
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import json
import os
import chardet
import hashlib
import speech_recognition as sr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 📌 Set Streamlit Page Config
st.set_page_config(layout="wide")

# 🎨 Apply Custom CSS
def custom_css():
    st.markdown(
        """
        <style>
        /* 🌟 General Styles */
        body, .stApp {
            background-color: #DCDCDC;  /* Light Gray */
            font-family: 'Arial', sans-serif;
        }

        /* 🏷️ Header Styles */
        h1, h2, h3 {
            font-size: 32px;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        /* 🎨 Sidebar Background */
        .css-1d391kg { /* Sidebar container */
            background-color: #BDB76B !important; /* Olive Drab */
            padding-top: 20px;
            padding-bottom: 20px;
        }

        .css-1cpxqw2 { /* Sidebar Text Color */
            color: #2C3E50 !important;
            font-size: 22px;
        }

        /* 📂 Upload Widget Styles */
        .css-14xtw13 {
            border: 1px solid #4A90E2;
            border-radius: 12px;
            padding: 12px;
            background-color: #E5F0FF;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        /* 🔘 Button Styles */
        .stButton > button {
            background-color: #4A90E2;
            color: #FFFFFF;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            transition: all 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #357ABD;
            box-shadow: 0 0 12px rgba(74, 144, 226, 0.8);
        }

        /* ✍️ Text Input Styles */
        .stTextInput > div > div > input, textarea {
            border: 1px solid #BDC3C7;
            border-radius: 8px;
            padding: 8px 12px;
            background-color: #FFFFFF;
        }

        /* 📊 DataFrame Styling - Bigger Size */
        .stDataFrame {
            width: 100% !important;
            height: 400px !important;  /* Bigger height */
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            overflow: hidden;
        }

        /* 📊 Progress Bar Styling */
        .stProgress > div > div > div {
            background: linear-gradient(to right, #4A90E2, #357ABD);
            border-radius: 12px;
        }

        /* ✅ Hide Footer */
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# 💡 Apply Custom CSS
custom_css()

# 🔥 App Title
st.title("📊 AI-Powered Multi-File Data Analysis & Forecasting with Voice Chat")

# 🗄 Define Storage Directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
USER_DATA_FILE = "users.json"
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# 🔍 AWS Bedrock Client Initialization
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_client = get_bedrock_client()

# 🔹 Load or Initialize User Data
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)

# 🔐 Hash Password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 🔐 User Authentication State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "signup_mode" not in st.session_state:
    st.session_state.signup_mode = False

# 🔹 Authentication Page
def login_page():
    st.sidebar.subheader("🔑 User Authentication")
    users = load_users()

    if st.sidebar.button("🔄 Switch to Sign Up" if not st.session_state.signup_mode else "🔄 Switch to Login"):
        st.session_state.signup_mode = not st.session_state.signup_mode
        st.rerun()

    if st.session_state.signup_mode:
        st.sidebar.subheader("📝 Sign Up")
        new_username = st.sidebar.text_input("Choose a Username")
        new_password = st.sidebar.text_input("Choose a Password", type="password")

        if st.sidebar.button("✅ Sign Up"):
            if new_username in users:
                st.sidebar.error("🚨 Username already exists! Choose another.")
            else:
                users[new_username] = hash_password(new_password)
                save_users(users)
                st.sidebar.success("🎉 Account created successfully! Please log in.")
                st.session_state.signup_mode = False
                st.rerun()
    else:
        st.sidebar.subheader("🔐 Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("🔓 Login"):
            if username in users and users[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.sidebar.success("✅ Login Successful!")
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid username or password!")

# 🔐 Sign Out Function
def sign_out():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.sidebar.success("👋 You have been signed out.")
    st.rerun()

# 🔐 Check Authentication Before Proceeding
if not st.session_state.authenticated:
    login_page()
    st.stop()

# ✅ User is Authenticated
st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")
if st.sidebar.button("🚪 Sign Out"):
    sign_out()

# 🎙️ Convert Voice to Text
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write("🎙️ Listening... Please speak your query.")
        try:
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio)
            st.sidebar.write(f"🗨️ You said: **{user_input}**")
            return user_input
        except sr.UnknownValueError:
            st.sidebar.error("❌ Sorry, I couldn't understand the audio.")
            return ""
        except sr.RequestError:
            st.sidebar.error("❌ API error. Please check internet connection.")
            return ""

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
# 📚 Show Each Uploaded File Separately
if dataframes:
    for i, df in enumerate(dataframes):
        st.write(f"### 📊 Preview of `{file_names[i]}`")
        st.dataframe(df.head(11))

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
                # Attempt to convert the column to numeric
                pd.to_numeric(df[col], errors="raise")
                valid_columns.append(col)
            except:
                try:
                    # If the column is non-numeric, try label encoding
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    valid_columns.append(col)
                except:
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
            
# 🧠 Chatbot Section
st.sidebar.header("🤖 Chat with Your Dataset")

# 🗂️ Load Chat History for User
user_chat_history_file = os.path.join(CHAT_HISTORY_DIR, f"{st.session_state.username}_chat_history.json")
if os.path.exists(user_chat_history_file):
    with open(user_chat_history_file, "r") as f:
        chat_history = json.load(f)
else:
    chat_history = []

# 🔄 Clear Chat History
def clear_chat_history():
    if os.path.exists(user_chat_history_file):
        os.remove(user_chat_history_file)
    st.sidebar.success("✅ Chat history cleared!")
    st.rerun()

# 🎙️ Capture Voice or Text Input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# 🎙️ Button to Capture Voice Input
if st.sidebar.button("🎙️ Speak"):
    voice_input = voice_to_text()
    if voice_input:
        st.session_state.user_input = voice_input

# ✍️ Text Input Box (Always Available)
user_input = st.sidebar.text_input(
    "Ask a question about your dataset:", value=st.session_state.user_input
)

# 🔥 Query Bedrock to Analyze Dataset
def query_bedrock_stream(user_input, df):
    df_sample = df.head(20).to_json()

    prompt = f"""
    You are an AI analyzing a dataset:
    {df_sample}

    User's question: {user_input}

    Provide a detailed and structured response.
    """

    payload = {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 300,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"],
        "anthropic_version": "bedrock-2023-05-31"
    }

    try:
        response = bedrock_client.invoke_model_with_response_stream(
            body=json.dumps(payload),
            modelId="anthropic.claude-instant-v1",
            accept="application/json",
            contentType="application/json"
        )

        stream = response["body"]
        full_response = ""
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_data = json.loads(chunk.get("bytes").decode())
                chunk_text = chunk_data.get("completion", "")
                full_response += chunk_text

        return full_response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🔥 Process User Query
if user_input:
    if dataframes:
        selected_df_index = st.sidebar.selectbox("Select Dataset to Query", file_names)
        selected_df = dataframes[file_names.index(selected_df_index)]
        response_text = query_bedrock_stream(user_input, selected_df)
        st.sidebar.markdown(
            f"### 💬 Latest Chat\n"
            f"🗨️ **{user_input}**\n"
            f"🤖 {response_text}"
        )
        # Save to Chat History
        chat_history.insert(0, {"question": user_input, "answer": response_text})
        if len(chat_history) > 10:
            chat_history.pop()
        with open(user_chat_history_file, "w") as f:
            json.dump(chat_history, f)

# 📜 Show Chat History
st.sidebar.subheader("📜 Chat History")
if len(chat_history) > 0:
    for chat in chat_history[:5]:
        st.sidebar.write(f"**🗨️ {chat['question']}**")
        st.sidebar.write(f"🤖 {chat['answer']}")
        st.sidebar.write("---")
else:
    st.sidebar.write("🤖 No chat history available.")

# 🧹 Button to Clear Chat History
if st.sidebar.button("🧹 Clear Chat History"):
    clear_chat_history()
