import streamlit as st
from audiorecorder import audiorecorder
import os
from dotenv import load_dotenv
from utils.transcriber import transcribe_audio
from utils.summarizer import summarize_meeting

from utils.sample_data import DEFAULT_MEETING_TRANSCRIPT

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Live Meeting Summarizer", layout="wide")

# Custom CSS Loading
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

def main():
    st.title("🎙️ Live Meeting Summarizer")
    st.markdown("### Transform your meetings into actionable insights instantly.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API Key")
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                st.success("API Key loaded from environment")
        
        st.markdown("---")
        if st.button("Load Sample Meeting"):
            st.session_state.transcript = DEFAULT_MEETING_TRANSCRIPT
            st.session_state.summary = None # Clear previous summary
            st.success("Sample meeting loaded!")

        st.info("Ensure you have a microphone connected.")

    # Main Content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Record Audio")
        # Audio Recorder
        audio = audiorecorder("Click to Record", "Click to Stop")
        
        if len(audio) > 0:
            st.audio(audio.export().read())
            
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing..."):
                    audio_bytes = audio.export().read()
                    transcript = transcribe_audio(audio_bytes)
                    
                    if "Error" in transcript:
                        st.error(transcript)
                    else:
                        st.session_state.transcript = transcript
                        st.session_state.summary = None # Clear previous summary
                        st.success("Transcription Complete!")
        
        # Display current transcript source
        if 'transcript' in st.session_state:
            st.markdown("---")
            st.subheader("Current Transcript")
            st.text_area("Content", st.session_state.transcript, height=300)
            
            # Download Transcript Button
            st.download_button(
                label="Download Transcript",
                data=st.session_state.transcript,
                file_name="meeting_transcript.txt",
                mime="text/plain"
            )

            if st.button("Generate Summary"):
                if not api_key:
                    st.error("Please provide a Groq API Key.")
                else:
                    with st.spinner("Summarizing..."):
                        summary = summarize_meeting(st.session_state.transcript, api_key)
                        st.session_state.summary = summary


    with col2:
        st.subheader("2. Meeting Notes")
        
        if 'summary' in st.session_state and st.session_state.summary:
            st.markdown("### AI Summary")
            st.markdown(st.session_state.summary)
            
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name="meeting_summary.md",
                mime="text/markdown"
            )
        elif 'transcript' not in st.session_state:
            st.info("Record audio or load the sample meeting to see results here.")

if __name__ == "__main__":
    main()
