# Groq Language Detection App (`app.py`)

## Overview
`app.py` is a Streamlit web application that uses Groq's API (Whisper and Llama 3) to detect the language of audio input. It supports both live recording and file uploads.

## Features
- **Record Audio**: Allows users to record 5 seconds of audio directly from the browser.
- **Upload Audio**: Supports uploading `.wav`, `.mp3`, and `.m4a` files.
- **Transcription**: Uses Whisper (via Groq) to transcribe the audio.
- **Language Detection**: Uses Llama 3 (via Groq) to analyze the transcription and identify the language.

## Dependencies
- `streamlit`
- `groq`
- `audio` (local module)
- `lang_detection` (local module)

## Usage
Run the app using Streamlit:
```bash
streamlit run app.py
```
