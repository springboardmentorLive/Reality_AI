from groq import Groq
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

client = Groq(api_key="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")

def record(filename="audio.wav"):
    print("Recording...")
    duration = 5
    samplerate = 16000
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, samplerate, audio)
    return filename

while True:
    file_path = record()
    response = client.audio.transcriptions.create(
        file=open(file_path, "rb"),
        model="whisper-large-v3",
        response_format="verbose_json"

    )
    print("Language:", response.language)
    print("Text:", response.text)
    print("-" * 50)