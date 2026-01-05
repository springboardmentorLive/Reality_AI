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
                # Use title if available, else fallback
                title = chat.get('title', f"Chat {chat['id'][:8]}...")
                if st.button(title, key=chat['id'], use_container_width=True):
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
                "history": st.session_state.messages[:-1], # Send history excluding new msg
                "title": st.session_state.get("chat_title") # Pass current title if specific state exists? 
                                                            # Actually backend handles title generation/persistence. 
                                                            # We can just rely on backend unless we want to persist known title.
            }
            # Wait, if we reload a chat, we don't store its title in session_state currently.
            # Ideally we should store st.session_state.chat_title when loading.
            
            with st.spinner("Thinking..."):
                response = requests.post(f"{API_URL}/chat", json=payload)
                
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get("response")
                updated_history = data.get("updated_history")
                new_title = data.get("title")
                
                st.session_state.messages = updated_history
                if new_title:
                   # Force reload sidebar if title changed (new chat)? 
                   # Streamlit rerun might be needed or just let next interaction update it.
                   pass
                
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            else:
                st.error("Error getting response from backend.")
                
        except Exception as e:
            st.error(f"Connection error: {e}")
