import os
import json
import pandas as pd
import streamlit as st
import logging
from auth import increment_usage, DATA_DIR
from s3_storage import load_chat_history, save_chat_history, delete_chat_history

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📂 Directory for local backup of Chat History
CHAT_HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Print the chat history location for debugging
logger.info(f"Local chat history backup directory: {CHAT_HISTORY_DIR}")

def get_local_chat_file(username):
    """Get the local chat history file path for a user as backup"""
    return os.path.join(CHAT_HISTORY_DIR, f"{username}_chat_history.json")

def load_user_chat_history(username):
    """Load chat history from S3 with local fallback"""
    try:
        # Try to load from S3
        chat_history = load_chat_history(username)
        logger.info(f"Chat history for {username} loaded from S3 successfully")
        return chat_history
    except Exception as e:
        logger.error(f"Error loading chat history from S3: {str(e)}")
        logger.info(f"Falling back to local storage for {username}'s chat history")
        
        # Fallback to local storage
        local_file = get_local_chat_file(username)
        if os.path.exists(local_file):
            try:
                with open(local_file, "r") as f:
                    return json.load(f)
            except Exception as local_e:
                logger.error(f"Error loading chat history from local backup: {str(local_e)}")
        
        return []

def save_user_chat_history(username, messages):
    """Save chat history to S3 with local backup"""
    try:
        # Try to save to S3
        success = save_chat_history(username, messages)
        if success:
            logger.info(f"Chat history for {username} saved to S3 successfully")
        else:
            raise Exception("S3 save operation failed")
    except Exception as e:
        logger.error(f"Error saving chat history to S3: {str(e)}")
        logger.info(f"Falling back to local storage for {username}'s chat history")
        
        # Fallback to local storage
        local_file = get_local_chat_file(username)
        try:
            with open(local_file, "w") as f:
                json.dump(messages, f)
            logger.info(f"Chat history for {username} saved to local backup successfully")
        except Exception as local_e:
            logger.error(f"Error saving chat history to local backup: {str(local_e)}")

def clear_user_chat_history(username):
    """Clear chat history from S3 and local backup"""
    try:
        # Try to delete from S3
        success = delete_chat_history(username)
        if success:
            logger.info(f"Chat history for {username} deleted from S3 successfully")
    except Exception as e:
        logger.error(f"Error deleting chat history from S3: {str(e)}")
    
    # Always try to clear local backup
    local_file = get_local_chat_file(username)
    if os.path.exists(local_file):
        try:
            os.remove(local_file)
            logger.info(f"Chat history for {username} deleted from local backup successfully")
        except Exception as e:
            logger.error(f"Error deleting chat history from local backup: {str(e)}")
    
    st.session_state.messages = []
    st.success("✅ Chat history cleared!")
    st.rerun()

# 📤 Process User Input and Get Response
def query_bedrock_stream(user_input, df, bedrock_client):
    # Get sample data as text
    df_sample = df.head(5).to_string()
    df_summary = df.describe().to_string()
    
    prompt = f"""
    You are a helpful AI data analyst. The user has asked about a dataset.

    Dataset Information:
    - Columns: {', '.join(df.columns)}
    
    Sample Data (first 5 rows):
    {df_sample}
    
    Statistical Summary:
    {df_summary}
    
    User's Question: {user_input}
    
    Please provide a clear, well-formatted response in plain text with:
    - A direct answer to the question
    - Relevant insights from the data
    - Simple formatting using bullet points or numbered lists when helpful
    """

    payload = {
        "modelId": "amazon.titan-text-lite-v1",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "inputText": prompt[:4000],
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(payload["body"]),
            modelId=payload["modelId"],
            accept=payload["accept"],
            contentType=payload["contentType"]
        )

        result = json.loads(response["body"].read())
        full_response = result["results"][0]["outputText"]
        return full_response.strip()

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🧠 Chatbot Section
def chatbot_section(dataframes, file_names, bedrock_client):
    st.subheader("🤖 Chat with Your Dataset")

    # Check authentication
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("⚠️ Please log in to use the chatbot.")
        return

    username = st.session_state.username

    # Initialize chat messages
    if "messages" not in st.session_state:
        # Load from history or initialize new
        st.session_state.messages = []
        history = load_user_chat_history(username)
        
        # Convert history to message format
        for item in history:
            st.session_state.messages.append({"role": "user", "content": item["question"]})
            st.session_state.messages.append({"role": "assistant", "content": item["answer"]})
        
        # Add welcome message if empty
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Hello! I'm your data assistant. Ask me anything about your datasets."
            })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input form
    with st.form("chat_form"):
        user_input = st.text_input("Ask a question about your dataset:", key="chat_input")
        submitted = st.form_submit_button("Submit")

    # Initialize usage tracking for chat
    if "chat_submission_counted" not in st.session_state:
        st.session_state.chat_submission_counted = False

    # Process when submitted
    if submitted and user_input.strip():
        if not dataframes:
            st.error("Please upload at least one dataset first.")
            return
        
        # Count usage for this chat interaction if not already counted
        if not st.session_state.chat_submission_counted:
            increment_usage()
            st.session_state.chat_submission_counted = True
            
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get selected dataset
        selected_df_index = st.selectbox("Select Dataset to Query", file_names, key="dataset_select")
        selected_df = dataframes[file_names.index(selected_df_index)]
        
        # Get and display response
        with st.spinner("Analyzing your data..."):
            response_text = query_bedrock_stream(user_input, selected_df, bedrock_client)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Save to history (convert messages to history format)
        history = []
        for i in range(0, len(st.session_state.messages), 2):
            if i+1 < len(st.session_state.messages):
                history.append({
                    "question": st.session_state.messages[i]["content"],
                    "answer": st.session_state.messages[i+1]["content"],
                    "timestamp": str(pd.Timestamp.now())
                })
        
        save_user_chat_history(username, history[-10:])  # Keep last 10 conversations
        st.rerun()

    # Clear chat button
    if st.button("🧹 Clear Conversation", key="clear_chat_button"):
        # Reset the usage tracking when chat is cleared
        st.session_state.chat_submission_counted = False
        clear_user_chat_history(username)
