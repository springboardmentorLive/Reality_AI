# Live Meeting Summarizer (`meeting.py`)

## Overview
`meeting.py` is a comprehensive Streamlit dashboard for real-time meeting transcription and summarization.

## Features
- **Live Recording**: Records audio in chunks (simulated live stream) and transcribes in near real-time.
- **File Upload**: Upload pre-recorded meetings to process.
- **Summarization**: Generates specific sections: Conversation, Summary, Conclusion, and Action Items.
- **Participants**: Allows specifying meeting participants for better context.
- **Export**: Download the full report as a text file.

## Dependencies
- `streamlit`
- `groq`
- `sounddevice` (for live recording)
- `scipy`

## Usage
Run the app:
```bash
streamlit run meeting.py
```
