import os
import json
import streamlit as st

# 📂 Directory for Chat History
CHAT_HISTORY_DIR = "chat_history"

# 🔄 Clear Chat History
def clear_chat_history(username):
    user_chat_history_file = os.path.join(CHAT_HISTORY_DIR, f"{username}_chat_history.json")
    if os.path.exists(user_chat_history_file):
        os.remove(user_chat_history_file)
    st.success("✅ Chat history cleared!")
    st.rerun()

# 📤 Process User Input and Get Response
def query_bedrock_stream(user_input, df, bedrock_client):
    # 🛑 Reduce dataset size to avoid token overflow
    df_sample = df.head(5).to_json()  # Reduce to 5 rows
    df_summary = df.describe().to_json()  # Summary stats

    prompt = f"""
    You are an AI analyzing a dataset.

    Dataset Overview:
    Columns: {list(df.columns)}
    Sample Data: {df_sample}
    Summary: {df_summary}

    User's question: {user_input}

    Provide a structured response.
    """

    # 📸 Updated Payload for Amazon Titan Text G1 - Lite
    payload = {
        "modelId": "amazon.titan-text-lite-v1",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "inputText": prompt[:4000],  # Ensure it stays within the token limit
            "textGenerationConfig": {
                "maxTokenCount": 1024,  # Reduce max token count
                "stopSequences": [],
                "temperature": 0.7,  # Adjust temperature for better responses
                "topP": 0.9
            }
        }
    }

    try:
        # ✅ Use Amazon Titan Text G1 - Lite API
        response = bedrock_client.invoke_model(
            body=json.dumps(payload["body"]),
            modelId=payload["modelId"],
            accept=payload["accept"],
            contentType=payload["contentType"]
        )

        result = json.loads(response["body"].read())
        full_response = result["results"][0]["outputText"]

        return full_response

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🧠 Chatbot Section
def chatbot_section(dataframes, file_names, bedrock_client):
    st.subheader("🤖 Chat with Your Dataset")

    # 🗂️ Load Chat History for User
    if "username" in st.session_state and st.session_state.username:
        username = st.session_state.username
        if not os.path.exists(CHAT_HISTORY_DIR):
            os.makedirs(CHAT_HISTORY_DIR)
        user_chat_history_file = os.path.join(CHAT_HISTORY_DIR, f"{username}_chat_history.json")
    else:
        st.warning("⚠️ You must be logged in to use the chatbot.")
        return

    if os.path.exists(user_chat_history_file):
        with open(user_chat_history_file, "r") as f:
            chat_history = json.load(f)
    else:
        chat_history = []

    # 🎙️ Capture Voice or Text Input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # ✍️ Text Input Box for User Query
    user_input = st.text_input(
        "Ask a question about your dataset:", value=st.session_state.user_input
    )

    # 📤 Process User Input and Get Response
    if user_input:
        if dataframes:
            selected_df_index = st.selectbox("Select Dataset to Query", file_names)
            selected_df = dataframes[file_names.index(selected_df_index)]
            response_text = query_bedrock_stream(user_input, selected_df, bedrock_client)

            st.markdown(
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

    # 📜 Show/Hide Full Chat History
    if len(chat_history) > 1:  # Only show if more than 1 chat exists
        with st.expander("📜 Chat History", expanded=False):
            for chat in chat_history[1:]:  # Skip the latest chat
                st.write(f"**🗨️ {chat['question']}**")
                st.write(f"🤖 {chat['answer']}")
                st.write("---")

    # 🧹 Clear Chat History Button
    if st.button("🧹 Clear Chat History"):
        clear_chat_history(username)
