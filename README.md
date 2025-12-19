# Groq AI Project

This repository contains several Python applications and scripts powered by Groq's high-speed inference API (Llama 3 and Whisper).

## Project Structure

- **`app.py`**: [Documentation](docs/app.md)
  - A Streamlit app for detecting the language of audio recordings or uploads.

- **`meeting.py`**: [Documentation](docs/meeting.md)
  - A "Live Meeting Summarizer" dashboard that records, transcribes, and summarizes meetings into action items.

- **`chatbot.py`**: [Documentation](docs/chatbot.md)
  - A simple command-line chatbot interface.

- **`groq_chat.py`**: [Documentation](docs/groq_chat.md)
  - A creative poetry generator CLI.

- **`audio.py`**: [Documentation](docs/audio.md)
  - A utility module for recording audio.

- **`lang_detection.py`**: [Documentation](docs/lang_detection.md)
  - A backend module for processing audio files and identifying languages.

## Getting Started

1. **Install Dependencies**:
   Ensure you have the required packages:
   ```bash
   pip install groq streamlit sounddevice scipy numpy
   ```

2. **API Key**:
   make sure to replace `GROQ_API_KEY` in the files with your actual Groq API key.

3. **Run an App**:
   Example:
   ```bash
   streamlit run meeting.py
   ```
