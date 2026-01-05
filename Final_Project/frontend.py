import streamlit as st
import requests
import uuid

# Configuration
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Springboard AI Chat", page_icon="🤖", layout="wide")

# Custom CSS for Premium Feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
    }
    div[data-testid="stChatMessage"] {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stChatMessage"][data-testid="user"] {
        background-color: #0e1117; 
        border: 1px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def load_chat_history(chat_id):
    try:
        response = requests.get(f"{API_URL}/chats/{chat_id}")
        if response.status_code == 200:
            data = response.json()
            # Handle different response formats if needed
            if isinstance(data, list):
                st.session_state.messages = data
            elif isinstance(data, dict):
                 st.session_state.messages = data.get('messages', [])
            st.session_state.chat_id = chat_id
    except Exception as e:
        st.error(f"Failed to load chat: {e}")

def create_new_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

# Sidebar
with st.sidebar:
    st.title("🗂️ Chat History")
    if st.button("+ New Chat", use_container_width=True):
        create_new_chat()
    
    st.divider()
    
    try:
        response = requests.get(f"{API_URL}/chats")
        if response.status_code == 200:
            chats = response.json()
            for chat in chats:
                label = f"Chat {chat['id'][:8]}..."
                if st.button(label, key=chat['id'], use_container_width=True):
                    load_chat_history(chat['id'])
    except Exception:
        st.warning("Could not connect to backend.")

# Main Chat Area
st.title("🤖 Springboard AI Chatbot")

if not st.session_state.chat_id:
    st.info("Start a new chat or select one from the sidebar to begin.")
else:
    # Display Messages
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        with st.chat_message(role):
            st.markdown(content)

    # Chat Input
    if prompt := st.chat_input("Type a message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Call Backend
        try:
            payload = {
                "chat_id": st.session_state.chat_id,
                "message": prompt,
                "history": st.session_state.messages[:-1]  # Send history excluding new msg (backend adds it? or assumes it?)
                                                           # Backend app.main:93 appends current user msg.
                                                           # So we should send history WITHOUT the new message?
                                                           # Backend: 
                                                           # current_history = request.history
                                                           # user_message = request.message
                                                           # current_history.append(... user_message ...)
                                                           # So yes, we send history BEFORE this message.
            }
            
            with st.spinner("Thinking..."):
                response = requests.post(f"{API_URL}/chat", json=payload)
                
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get("response")
                updated_history = data.get("updated_history") # Full history including AI
                
                st.session_state.messages = updated_history # Sync state
                
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            else:
                st.error("Error getting response from backend.")
                
        except Exception as e:
            st.error(f"Connection error: {e}")
