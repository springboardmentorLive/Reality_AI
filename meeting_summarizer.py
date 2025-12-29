import streamlit as st
import time
import datetime
import random
import json
import threading
import queue
import sounddevice as sd # Library for audio input/output
import numpy as np # For processing audio chunks

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Audio settings
FS = 16000  # Sample rate (standard for STT)
BLOCKSIZE = 1024 # Buffer size
CHANNELS = 1
MIN_VOLUME_THRESHOLD = 0.01 # Minimum volume to trigger a simulated 'utterance'

# Page config
st.set_page_config(
    page_title="Live Meeting Summarizer (Microphone)",
    page_icon="🎤",
    layout="wide"
)
st.title("🎤 Live Meeting Summarizer (Microphone Input)")
st.markdown("Capturing real-time audio input using a background thread.")

# Sample data for simulated STT output
SAMPLE_SPEAKERS = ["John", "Sarah", "Mike", "Alex"]
SAMPLE_TEXTS = [
    "The client demo deadline needs to be finalized today.",
    "We should schedule a code review for tomorrow morning.",
    "I will handle the budget sign-off by the end of the day.",
    "The current focus is the backend refactoring, due next Tuesday.",
    "I can take the lead on the UI changes."
]

# Initialize session state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'key_points' not in st.session_state:
    st.session_state.key_points = []
if 'action_items' not in st.session_state:
    st.session_state.action_items = []

# Thread status, data queue, and audio stream object
if 'recorder_thread' not in st.session_state:
    st.session_state.recorder_thread = None
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = queue.Queue()


# --- 2. BACKGROUND THREAD CLASS (LIVE AUDIO CAPTURE) ---

class MicrophoneRecorderThread(threading.Thread):
    """
    A background thread that continuously captures audio from the microphone
    and processes it.
    """
    def __init__(self, data_queue):
        super().__init__()
        self.data_queue = data_queue
        self._stop_event = threading.Event()
        self.stream = None

    def stop(self):
        """Sets internal flag and closes the audio stream safely."""
        self._stop_event.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def stopped(self):
        return self._stop_event.is_set()
    
    def callback(self, indata, frames, time, status):
        """This function is called by the sounddevice stream for every audio block."""
        if status:
            print(status, flush=True)

        # Calculate volume (Root Mean Square - RMS)
        # We multiply by a factor (e.g., 10) for better visibility in the UI progress bar
        volume_norm = np.linalg.norm(indata) * 10 
        
        if volume_norm > MIN_VOLUME_THRESHOLD:
            # --- STT SIMULATION: REPLACE THIS WITH YOUR REAL API CALL ---
            
            # 1. Simulate STT/Diarization using the loudness trigger
            new_text = random.choice(SAMPLE_TEXTS)
            new_speaker = random.choice(SAMPLE_SPEAKERS)

            new_entry = {
                'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
                'speaker': new_speaker,
                'text': new_text,
                'volume': f"{volume_norm:.2f}"
            }
            
            # 2. Put the simulated transcription data into the queue
            self.data_queue.put(new_entry)
            
            # Add a slight delay to prevent flooding the queue
            time.sleep(0.1)


    def run(self):
        """The main loop opens the audio stream."""
        try:
            # Open the audio stream
            self.stream = sd.InputStream(
                samplerate=FS, 
                blocksize=BLOCKSIZE, 
                channels=CHANNELS, 
                callback=self.callback
            )
            # Start the stream and block until stop() is called
            with self.stream:
                while not self.stopped():
                    sd.sleep(100) # Sleep for 100 milliseconds
                    
        except Exception as e:
            # Handle potential exceptions like microphone not found
            print(f"Error in audio thread: {e}")
            self.data_queue.put({"error": str(e), "timestamp": datetime.datetime.now().strftime("%H:%M:%S")})
        finally:
            print("Microphone thread shutting down.")


# --- 3. CORE AI/ML FUNCTIONS (PLACEHOLDERS) ---

def generate_summary(transcript_text):
    """Placeholder for LLM call."""
    time.sleep(3) 
    st.session_state.summary = "The team discussed the upcoming sprint priorities, focusing on backend refactoring and the client demo deadline. A code review was scheduled, and action items were assigned to various team members."
    st.session_state.key_points = [
        "Backend refactoring is the top priority for the current sprint.",
        "A code review is scheduled for tomorrow at 10 AM.",
        "The budget proposal is ready to be sent out for manager sign-off."
    ]
    st.success("✅ Summary and Key Points Generated!")
    st.rerun()

def extract_action_items(transcript_text):
    """Placeholder for LLM call."""
    time.sleep(3) 
    st.session_state.action_items = [
        {"task": "Backend refactoring", "assignee": "John", "deadline": "Next Tuesday"},
        {"task": "Schedule code review", "assignee": "Mike", "deadline": "EOD Today"},
        {"task": "Update JIRA ticket for refactoring", "assignee": "John", "deadline": "Tomorrow"},
        {"task": "UI changes for demo", "assignee": "Sarah", "deadline": "End of Week"},
    ]
    st.success("✅ Action Items Extracted!")
    st.rerun()


# --- 4. UI LOGIC (Thread Management) ---

def start_recording():
    """Starts the microphone thread."""
    if not st.session_state.is_recording:
        if st.session_state.recorder_thread is not None:
            st.session_state.recorder_thread.stop()
        
        thread = MicrophoneRecorderThread(st.session_state.data_queue)
        thread.start()
        st.session_state.recorder_thread = thread
        st.session_state.is_recording = True
        st.rerun() 

def stop_recording():
    """Stops the microphone thread safely."""
    if st.session_state.is_recording and st.session_state.recorder_thread:
        st.session_state.recorder_thread.stop()
        st.session_state.recorder_thread.join()
        st.session_state.is_recording = False
        st.session_state.recorder_thread = None
        st.rerun()

def clear_all():
    """Resets all state."""
    stop_recording()
    st.session_state.transcript = []
    st.session_state.summary = ""
    st.session_state.key_points = []
    st.session_state.action_items = []
    st.rerun()


# --- Check the queue for new data and update the transcript ---
if st.session_state.is_recording:
    try:
        # Get all available data from the queue without blocking
        while not st.session_state.data_queue.empty():
            new_entry = st.session_state.data_queue.get_nowait()
            
            # Check for error message from the thread
            if "error" in new_entry:
                st.error(f"FATAL ERROR: Microphone thread failed. {new_entry['error']}")
                stop_recording()
                break
                
            st.session_state.transcript.append(new_entry)
        
        # Rerun to check queue and update UI periodically (non-blocking update)
        time.sleep(0.1) 
        st.rerun() 
    except queue.Empty:
        pass


# --- 5. UI LAYOUT ---

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Meeting Controls")

    col1, col2 = st.columns(2)
    with col1:
        st.button("▶️ Start Listening", on_click=start_recording, disabled=st.session_state.is_recording, use_container_width=True)
    with col2:
        st.button("🛑 Stop Listening", on_click=stop_recording, disabled=not st.session_state.is_recording, use_container_width=True)

    if st.button("🗑️ Clear All", on_click=clear_all, use_container_width=True):
        pass
    
    st.divider()

    st.subheader("Meeting Info")
    meeting_title = st.text_input("Meeting Title", "Team Standup", key='meeting_title')
    participants = st.text_area("Participants (Comma Separated)", "John, Sarah, Mike", key='participants')
    
    st.divider()

    st.subheader("Settings")
    st.markdown(f"**Current Mic Sensitivity:** `{MIN_VOLUME_THRESHOLD}`")
    st.slider("Auto-summarize every (seconds)", 30, 300, 60, key='summarize_interval')
    st.selectbox("Language", ["English", "Spanish", "French", "German"], key='language')

# Status indicator
st.divider()
if st.session_state.is_recording:
    st.success("🟢 LIVE: Microphone is active. Speak to see the transcript update.")
else:
    st.info("⚪ Not listening. Click 'Start Listening' to begin.")
st.divider()

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Live Transcript", "📊 Summary", "✅ Action Items", "📥 Export Data"])

# --- TAB 1: Live Transcript (Real-Time Audio Input) ---
with tab1:
    st.subheader("Live Transcription")
    st.caption("This transcript updates only when the microphone detects a voice above the sensitivity threshold.")
    transcript_container = st.container(height=500)
    
    with transcript_container:
        if st.session_state.transcript:
            for entry in reversed(st.session_state.transcript):
                timestamp = entry.get('timestamp', '')
                speaker = entry.get('speaker', 'Unknown')
                text = entry.get('text', '')
                volume = entry.get('volume', '0.00')
                
                col_ts, col_vol, col_text = st.columns([1, 1, 4])
                with col_ts:
                    st.caption(f"**{speaker}**")
                    st.caption(timestamp)
                with col_vol:
                    # Display the volume that triggered the utterance using a progress bar
                    st.progress(min(float(volume) * 0.1, 1.0)) 
                    st.caption(f"Vol: {volume}")
                with col_text:
                    st.write(text)
        else:
            st.info("Start the recording and speak into your microphone to see live updates.")


# --- TAB 2: Meeting Summary ---
with tab2:
    st.subheader("Meeting Summary")
    if st.button("✨ Generate Summary & Key Points", use_container_width=False, key="gen_summary"):
        # Stop listening before summarizing to get a stable final transcript
        stop_recording() 
        if not st.session_state.transcript:
            st.warning("Cannot summarize, the transcript is empty.")
        else:
            transcript_text = "\n".join([f"[{e['speaker']} - {e['timestamp']}]: {e['text']}" for e in st.session_state.transcript])
            with st.spinner("Analyzing transcript and generating summary..."):
                generate_summary(transcript_text) 

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        st.subheader("Key Points")
        for i, point in enumerate(st.session_state.key_points, 1):
            st.markdown(f"**{i}.** {point}")
    else:
        st.info("Generate a summary to see the meeting overview and key points.")

# --- TAB 3: Action Items ---
with tab3:
    st.subheader("Action Items")
    if st.button("✅ Extract Action Items", use_container_width=False, key="extract_actions"):
        # Stop listening before extracting actions
        stop_recording() 
        if not st.session_state.transcript:
            st.warning("Cannot extract action items, the transcript is empty.")
        else:
            transcript_text = "\n".join([f"[{e['speaker']} - {e['timestamp']}]: {e['text']}" for e in st.session_state.transcript])
            with st.spinner("Extracting action items..."):
                extract_action_items(transcript_text)

    if st.session_state.action_items:
        for i, item in enumerate(st.session_state.action_items, 1):
            with st.expander(f"Action Item {i}: {item['task']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Assignee:** {item['assignee']}")
                with col2:
                    st.write(f"**Deadline:** {item['deadline']}")
                with col3:
                    st.checkbox("Completed", key=f"action_{i}")
    else:
        st.info("Extract action items to see tasks and assignments from the meeting.")

# --- TAB 4: Export Meeting Data ---
with tab4:
    st.subheader("Export Meeting Data")
    export_data = {
        "meeting_title": meeting_title,
        "participants": participants.split(', '),
        "summary": st.session_state.summary,
        "key_points": st.session_state.key_points,
        "action_items": st.session_state.action_items,
        "full_transcript": st.session_state.transcript,
        "timestamp": datetime.datetime.now().isoformat()
    }
    st.code(json.dumps(export_data, indent=2), language='json')
    st.download_button(
        label="Download Meeting Data (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"{meeting_title.replace(' ', '_')}_summary.json",
        mime="application/json",
        use_container_width=True
    )