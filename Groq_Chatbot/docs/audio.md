# Audio Recorder (`audio.py`)

## Overview
`audio.py` is a helper script and module for recording audio and determining its content/language using Groq's Whisper model.

## Features
- **Recording Function**: `record(filename)` records audio from the microphone for a fixed duration (default 5 seconds).
- **Standalone Mode**: If run directly (`python audio.py`), it enters a loop where it records audio and prints the transcribed text and detected language.

## Dependencies
- `groq`
- `sounddevice`
- `numpy`
- `scipy`

## Usage
As a module:
```python
from audio import record
filename = record("output.wav")
```

As a script:
```bash
python audio.py
```
