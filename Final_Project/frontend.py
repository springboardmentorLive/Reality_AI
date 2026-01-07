import streamlit as st
import requests
import uuid
import json

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Springboard AI Chat", page_icon="🤖", layout="wide")

# Custom CSS
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
            # Handle different response formats
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
                title = chat.get('title')
                # If title is missing or empty, try to derive from messages (legacy support)
                if not title:
                    title = f"Chat {chat['id'][:8]}..."
                
                if st.button(title, key=chat['id'], use_container_width=True):
                    load_chat_history(chat['id'])
    except Exception:
        st.warning("Could not connect to backend.")

# --- Document Upload portion ---
import pypdf

def extract_text_from_file(uploaded_file):
    try:
        text = ""
        if uploaded_file.type == "application/pdf":
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else: # Text file
            text = uploaded_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None





st.title("🤖 Springboard AI Chatbot")

if not st.session_state.chat_id:
    st.info("Start a new chat or select one from the sidebar to begin.")
else:
    for message in st.session_state.messages:
        role = message.get("role")
        if role == "system":
            continue # Don't show context documents in chat
            
        content = message.get("content")
        with st.chat_message(role):
            st.markdown(content)
    

    with st.popover("➕ Add Context", use_container_width=False):
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], key="main_uploader")
        
        if uploaded_file:
            # Check if we already processed this file
            if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != uploaded_file.name:
                with st.spinner("Processing document..."):
                    text = extract_text_from_file(uploaded_file)
                    if text:
                        context_msg = {
                            "role": "system", 
                            "content": f"Use the following document content as context to answer user questions:\n\n{text[:50000]}"
                        }
                        st.session_state.messages.append(context_msg)
                        st.session_state.last_processed_file = uploaded_file.name
                        st.success(f"Attached: {uploaded_file.name}")

    # Chat Input
    if prompt := st.chat_input("Type a message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        #### Call Backend
        try:
            payload = {
                "chat_id": st.session_state.chat_id,
                "message": prompt,
                "history": st.session_state.messages[:-1],
                "title": st.session_state.get("chat_title") 
            }
            
            with st.chat_message("assistant"):
                response = requests.post(f"{API_URL}/chat", json=payload, stream=True)
                
                if response.status_code == 200:
                    def stream_parser():
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    if data['type'] == 'chunk':
                                        yield data['content']
                                    elif data['type'] == 'complete':
                                        st.session_state.stream_metadata = data
                                    elif data['type'] == 'error':
                                        st.error(data['content'])
                                except json.JSONDecodeError:
                                    pass
                    
                    full_response = st.write_stream(stream_parser())
                    
                    # Update history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Handle metadata (e.g. title update)
                    if "stream_metadata" in st.session_state:
                        meta = st.session_state.pop("stream_metadata")
                        new_title = meta.get("title")
                        chat_id = meta.get("chat_id")
                        
                        # If a new title was generated, we might want to update the UI (optional: rerun)
                        if meta.get("new_title_generated"):
                             st.rerun()
                else:
                    st.error("Error getting response from backend.")
                
        except Exception as e:
            st.error(f"Connection error: {e}")
