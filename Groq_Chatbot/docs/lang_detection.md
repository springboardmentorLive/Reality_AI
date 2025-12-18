# Language Detection Logic (`lang_detection.py`)

## Overview
`lang_detection.py` contains the core logic for transcribing audio and detecting language using Groq.

## Features
- **`transcribe_and_detect_language(file_path)`**: Transcribes an audio file using Whisper.
- **`detect_language_llm(text)`**: Uses Llama 3 to analyze text and identify the language (ISO 639-1 code + name).
- **Standalone Mode**: If run directly, it asks for a file path to process.

## Dependencies
- `groq`

## Usage
As a module:
```python
from lang_detection import transcribe_and_detect_language, detect_language_llm

text = transcribe_and_detect_language("audio.wav")
lang = detect_language_llm(text)
```

As a script:
```bash
python lang_detection.py
```
