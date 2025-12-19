import streamlit as st
import os
from groq import Groq
# from audio import record # Not using this directly as we need custom duration

# Setup Groq Client
client = Groq(api_key="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")

st.set_page_config(page_title="Live Meeting Summarizer", layout="wide")
st.title("📝 Live Meeting Summarizer")

# Session State Initialization
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = "Waiting for meeting content..."
if "action_items" not in st.session_state:
    st.session_state.action_items = "Waiting for action items..."
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "participants" not in st.session_state:
    st.session_state.participants = ""


# Functions
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=file
            )
        return transcription.text
    except Exception as e:
        st.error(f"Error transcribing: {e}")
        return ""

def update_summary(text, participants):
    try:
        context = ""
        if participants:
            context = f"The expected participants are: {participants}."
            
        prompt = (
            f"You are a professional meeting minute-taker. {context}\n"
            "Analyze the meeting transcript and generate a comprehensive report.\n"
            "Strictly use the following default format:\n\n"
            "### 🗣️ Conversation & Key Points\n"
            "(Outline the flow of the conversation)\n\n"
            "### 📝 Summary\n"
            "(A concise overview)\n\n"
            "### 🎯 Conclusion\n"
            "(Final decisions)\n\n"
            "### ✅ Action Items\n"
            "(Bulleted list of tasks)\n\n"
            "--------------------------------------------------\n"
            f"TRANSCRIPT:\n{text}"
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        
        parts = content.split("### ✅ Action Items")
        summary_part = parts[0]
        action_part = "### ✅ Action Items\n" + parts[1] if len(parts) > 1 else "No action items detected yet."
        
        return summary_part, action_part
        
    except Exception as e:
        st.error(f"Error summarizing: {e}")
        return st.session_state.summary, st.session_state.action_items

# Sidebar Controls

with st.sidebar:
    st.header("Meeting Setup")
    
    # Audio Source Selection
    audio_source = st.radio("Audio Source", ["Live Microphone", "Upload Audio File"])
    
    # Input for Participants
    participants_input = st.text_input("Participants (Comma separated)", 
                                      placeholder="Alice, Bob, Charlie",
                                      value=st.session_state.participants)
    st.session_state.participants = participants_input

    st.markdown("---")
    st.header("Controls")
    
    if audio_source == "Live Microphone":
        if st.button("Start Meeting", type="primary"):
            st.session_state.is_recording = True
            st.rerun()
        
        if st.button("Stop Meeting"):
            st.session_state.is_recording = False
            st.rerun()
    else:
        # Upload Controls
        uploaded_file = st.file_uploader("Upload Recording", type=["wav", "mp3", "m4a"])
        if uploaded_file and st.button("Process Recording", type="primary"):
            with st.spinner("Processing file..."):
                # Save temp
                temp_filename = "temp_upload_" + uploaded_file.name
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Transcribe
                text = transcribe_audio(temp_filename)
                st.session_state.transcript = text
                
                # Summarize
                summ, acts = update_summary(text, st.session_state.participants)
                st.session_state.summary = summ
                st.session_state.action_items = acts
                
                # Cleanup
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            st.success("Processing Complete!")
        
    st.markdown("---")
    if st.button("Load Demo Transcript"):
        demo_transcript = (
            "Alice: Hi everyone, thanks for joining. We need to discuss the new marketing campaign.\n"
            "Bob: I've prepared the draft for the social media posts. The focus is on our new product features.\n"
            "Charlie: That sounds good. Have we decided on the launch date yet?\n"
            "Alice: Not yet. I was thinking of next Monday. Does that work for everyone?\n"
            "Bob: Monday works for me. I can get the final visuals ready by Friday.\n"
            "Charlie: I need to check with the dev team if the landing page will be ready. I'll get back to you by tomorrow.\n"
            "Alice: Okay, let's tentatively aim for Monday. Charlie, please update us tomorrow.\n"
            "Bob: Also, are we doing a paid ad campaign?\n"
            "Alice: Yes, we have a budget of $5000 for the first month.\n"
            "Charlie: Great. I'll make sure the tracking pixels are set up correctly.\n"
            "Alice: Perfect. So, action items: Bob on visuals, Charlie on landing page and tracking, and I'll finalize the budget allocation.\n"
            "Bob: Got it.\n"
            "Charlie: Understood."
        )
        st.session_state.transcript = demo_transcript
        st.session_state.participants = "Alice, Bob, Charlie"
        
        with st.spinner("Generating summary for demo..."):
            summ, acts = update_summary(demo_transcript, st.session_state.participants)
            st.session_state.summary = summ
            st.session_state.action_items = acts
        st.success("Demo Transcript Loaded!")
        
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.transcript = ""
        st.session_state.summary = "Waiting for meeting content..."
        st.session_state.action_items = "Waiting for action items..."
        st.rerun()




# Main UI Layout
tab1, tab2, tab3, tab4 = st.tabs(["📄 Live Transcript", "📊 Summary", "✅ Action Items", "💾 Export"])

with tab1:
    st.subheader("Live Transcript")
    st.text_area("Transcript", value=st.session_state.transcript, height=600, label_visibility="collapsed")

with tab2:
    st.subheader("Meeting Summary")
    st.markdown(st.session_state.summary)

with tab3:
    st.subheader("Action Items")
    st.markdown(st.session_state.action_items)

with tab4:
    st.subheader("Export Meeting Data")
    if st.button("Download Transcript & Summary"):
        full_report = f"TRANSCRIPT:\n\n{st.session_state.transcript}\n\n" \
                      f"--------------------------------------------------\n\n" \
                      f"SUMMARY REPORT:\n\n{st.session_state.summary}\n\n" \
                      f"{st.session_state.action_items}"
        st.download_button(
            label="Download Report as Text",
            data=full_report,
            file_name="meeting_report.txt",
            mime="text/plain"
        )

# Recording Loop Logic
if audio_source == "Live Microphone" and st.session_state.is_recording:
    with st.spinner("Recording... (Speak now)"):
        # Temporary local record function
        import sounddevice as sd
        import scipy.io.wavfile as wav
        
        def record_chunk(filename, duration=10):
            fs = 16000
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            wav.write(filename, fs, recording)
            
        chunk_file = "temp_live_chunk.wav"
        record_chunk(chunk_file, duration=8) # 8 seconds chunks
        
        # Transcribe
        new_text = transcribe_audio(chunk_file)
        
        if new_text:
            st.session_state.transcript += new_text + " "
            # Update summary
            summ, acts = update_summary(
                st.session_state.transcript, 
                st.session_state.participants
            )
            st.session_state.summary = summ
            st.session_state.action_items = acts
        
        # Clean up
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            
        # Rerun to continue the loop and update UI
        st.rerun()
