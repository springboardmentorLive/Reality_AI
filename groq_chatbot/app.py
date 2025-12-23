import streamlit as st
import os
from groq_bot import ChatBot
from poem_generator import ask_bot as generate_poem

# -----------------------------------------------------------------------------
# Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Groq AI Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }

    /* Message bubbles */
    .stChatMessage {
        background-color: transparent; 
        border: 1px solid #30363d;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    /* Poem Card */
    .poem-card {
        background-color: #1f242c;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        font-family: 'Georgia', serif;
        line-height: 1.6;
        white-space: pre-wrap; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌌 Groq AI Suite")
    st.markdown("---")
    
    selected_app = st.radio(
        "Choose Application:",
        ["💬 General Chatbot", "📝 Poetry Generator"],
        index=0
    )
    
    st.markdown("---")
    st.caption("Powered by Groq & Llama 3")
    
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# App: General Chatbot
# -----------------------------------------------------------------------------
if selected_app == "💬 General Chatbot":
    st.header("💬 Intelligent Assistant")
    st.caption("Ask me anything. I'm fast and versatile.")

    # Initialize ChatBot in session state
    if "chatbot" not in st.session_state:
        api_key = os.environ.get("PUBLIC_GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ API Key missing! Please set PUBLIC_GROQ_API_KEY.")
            st.stop()
        
        st.session_state.chatbot = ChatBot(api_key)
        # Store message history separate from the bot object for UI rendering if needed, 
        # but the bot object manages its own history. We'll use the bot's history logic.
        # Ideally, we sync UI with bot.history or just store UI messages.
        # Let's keep a simple UI history list to render.
        st.session_state.messages = [{"role": "system", "content": st.session_state.chatbot.system_prompt}]

    # Display chat messages (excluding system prompt for cleaner UI if desired, but let's show user/ai)
    # We'll skip the first system message in display
    for msg in st.session_state.chatbot.history:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Type your message..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.ask(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

# -----------------------------------------------------------------------------
# App: Poetry Generator
# -----------------------------------------------------------------------------
elif selected_app == "📝 Poetry Generator":
    st.header("📝 AI Poet")
    st.caption("Generate creative, beautiful poetry in seconds.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        theme = st.text_input("What should the poem be about?", placeholder="e.g., A rainy night in Tokyo")
    
    with col2:
        style = st.selectbox(
            "Choose a style", 
            ["Romantic", "Melancholic", "Motivational", "Humorous", "Dark", "Shakespearean", "Haiku"]
        )

    if st.button("✨ Generate Poem"):
        if not theme:
            st.warning("Please enter a theme first.")
        else:
            with st.spinner("Composing verses..."):
                try:
                    poem = generate_poem(theme, style)
                    st.success("Poem Generated successfully!")
                    
                    st.markdown(f"""
                    <div class="poem-card">
                        <h3>{theme.title()} ({style})</h3>
                        <br>
                        {poem}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating poem: {e}")

