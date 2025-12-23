import streamlit as st
import os
from audio import record
from lang_detection import transcribe_and_detect_language, detect_language_llm

st.set_page_config(page_title="Groq Language Detection", layout="centered")

st.title("🎙️ Groq Language Detection")
st.write("Record audio or upload an audio file to detect the language using Whisper and Llama 3 models on Groq.")

# Create tabs for Record and Upload
tab1, tab2 = st.tabs(["🔴 Record Audio", "📂 Upload Audio"])

with tab1:
    st.header("Record Audio")
    if st.button("Start Recording (5s)"):
        with st.spinner("Recording..."):
            file_path = record("streamlit_recording.wav")
        
        st.success("Recording complete!")
        
        if os.path.exists(file_path):
            st.audio(file_path, format="audio/wav")
            
            with st.spinner("Transcribing and Detecting Language..."):
                transcription = transcribe_and_detect_language(file_path)
                
                if transcription:
                    st.subheader("Transcribed Text")
                    st.info(transcription)
                    
                    st.subheader("Detected Language (LLM Verification)")
                    with st.spinner("Analyzing text with LLM..."):
                        language_llm = detect_language_llm(transcription)
                        st.success(language_llm)
                else:
                    st.error("Could not transcribe audio.")
        else:
            st.error("Recording file not found.")

with tab2:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Transcribe Uploaded Audio"):
            # Save the uploaded file temporarily
            temp_filename = "temp_upload_" + uploaded_file.name
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Transcribing and Detecting Language..."):
                transcription = transcribe_and_detect_language(temp_filename)
                
                if transcription:
                    st.subheader("Transcribed Text")
                    st.info(transcription)
                    
                    st.subheader("Detected Language (LLM Verification)")
                    with st.spinner("Analyzing text with LLM..."):
                        language_llm = detect_language_llm(transcription)
                        st.success(language_llm)
                else:
                    st.error("Could not transcribe audio.")
            
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

st.markdown("---")
st.caption("Powered by Groq, Whisper, and Llama 3.")