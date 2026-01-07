# Live Meeting Summarizer

A real-time meeting summarizer built with Streamlit and Python. This application captures audio, transcribes it, and generates concise summaries using Groq's high-speed AI inference (Llama 3).

## Features

- **Live Audio Recording**: Capture meeting audio directly from your browser.
- **Transcription**: Converts speech to text (Note: Currently uses basic SpeechRecognition).
- **AI Summarization**: Generates summaries, action items, and key takeaways using Groq API.
- **Sample Meeting**: Load a default transcript to test the summarization without recording.

## Prerequisites

- Python 3.8+
- A [Groq API Key](https://console.groq.com/keys).

## Installation

1. Clone the repository (or download the files).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   - Create a `.env` file or enter your API Key directly in the app sidebar.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter your Groq API Key in the sidebar.
3. Use the recorder to capture audio or load the sample meeting.
4. Click "Summarize" to get the results.

## Technologies

- Streamlit
- SpeechRecognition
- Groq (Llama 3)
- Pydub
