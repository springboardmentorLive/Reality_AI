import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from langdetect import detect, LangDetectException
import numpy as np
import time

# ----------------------------------------------------------
# Step 1: Record audio from microphone
# ----------------------------------------------------------

duration = 5  # seconds of recording
sample_rate = 16000  
output_path = "live_audio.wav"

print("🎤 Speak now... Recording starts in 2 seconds")
time.sleep(2)
print("Recording...")

audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()  

# Convert to 16-bit PCM WAV
audio_data_int16 = np.int16(audio_data * 32767)
write(output_path, sample_rate, audio_data_int16)

print(f"Audio saved to {output_path}")

# ----------------------------------------------------------
# Step 2: Load Whisper model
# ----------------------------------------------------------
print("Loading Whisper model...(takes 20–30 sec)")
model = whisper.load_model("small")

# ----------------------------------------------------------
# Step 3: Transcribe the audio
# ----------------------------------------------------------
print("\nTranscribing...")
result = model.transcribe(output_path)
text = result["text"].strip()

print("\nTranscribed Text:\n")
print(text)

# ----------------------------------------------------------
# Step 4: Detect language
# ----------------------------------------------------------
try:
    lang = detect(text)
    print("\nDetected Language Code:", lang)
except LangDetectException:
    print("\nError: Could not detect language (text too short)")
