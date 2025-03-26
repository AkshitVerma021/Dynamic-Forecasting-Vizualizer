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
    df_sample = df.head(50).to_json()

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
