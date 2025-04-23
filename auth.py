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
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0
    if "paid_user" not in st.session_state:
        st.session_state.paid_user = False

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
                users[new_username] = {
                    "password": hash_password(new_password),
                    "usage_count": 0,
                    "paid_user": False
                }
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
            if username in users:
                # Check if the user data is in the old format (just password hash)
                if isinstance(users[username], str):
                    # Migrate user to new format
                    if users[username] == hash_password(password):
                        users[username] = {
                            "password": users[username],
                            "usage_count": 0,
                            "paid_user": False
                        }
                        save_users(users)
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.usage_count = 0
                        st.session_state.paid_user = False
                        st.sidebar.success("✅ Login Successful!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Invalid username or password!")
                # New format with usage tracking
                elif users[username]["password"] == hash_password(password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.usage_count = users[username]["usage_count"]
                    st.session_state.paid_user = users[username]["paid_user"]
                    st.sidebar.success("✅ Login Successful!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Invalid username or password!")
            else:
                st.sidebar.error("❌ Invalid username or password!")

# 🔐 Sign Out Function
def sign_out():
    if st.session_state.authenticated and st.session_state.username:
        # Save usage count before signing out
        users = load_users()
        if st.session_state.username in users:
            if isinstance(users[st.session_state.username], dict):
                users[st.session_state.username]["usage_count"] = st.session_state.usage_count
                users[st.session_state.username]["paid_user"] = st.session_state.paid_user
                save_users(users)
    
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.usage_count = 0
    st.session_state.paid_user = False
    st.sidebar.success("👋 You have been signed out.")
    st.rerun()

# 🔢 Track Usage Function
def increment_usage():
    if st.session_state.authenticated and not st.session_state.paid_user:
        users = load_users()
        if st.session_state.username in users:
            st.session_state.usage_count += 1
            if isinstance(users[st.session_state.username], dict):
                users[st.session_state.username]["usage_count"] = st.session_state.usage_count
                save_users(users)

# 🛑 Check Usage Limits
def check_usage_limit():
    # Free usage limit
    FREE_USAGE_LIMIT = 10
    
    if st.session_state.authenticated:
        if st.session_state.paid_user:
            return True
        elif st.session_state.usage_count < FREE_USAGE_LIMIT:
            return True
        else:
            return False
    return False

# 🔐 Check Authentication Before Proceeding
def check_auth():
    if not st.session_state.authenticated:
        login_page()
        st.stop()
