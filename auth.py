import os
import streamlit as st
import json
import hashlib

# 📂 User Data File Location
USER_DATA_FILE = "users.json"

# 📝 Create users.json if it does not exist
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as f:
        json.dump({}, f)

# 🔐 Load Users from JSON File
def load_users():
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

# 💾 Save Users to JSON File
def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)

# 🔐 Hash Password for Security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 🔑 User Authentication State
def init_session_state():
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

    # Toggle between Login and Sign Up
    if st.sidebar.button("🔄 Switch to Sign Up" if not st.session_state.signup_mode else "🔄 Switch to Login"):
        st.session_state.signup_mode = not st.session_state.signup_mode
        st.rerun()

    # 🎉 Sign Up Section
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
        # 🔐 Login Section
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
def check_auth():
    if not st.session_state.authenticated:
        login_page()
        st.stop()
