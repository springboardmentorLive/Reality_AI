from groq import Groq
import sounddevice as sd
import numpy as np 
import scipy.io.wavfile as wav 
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("PUBLIC_GROQ_API_KEY"))

def record(filename="record.wav"):
    print("Recording...")
    duration = 5
    samplerate = 16000

    audio = sd.rec(int(duration * samplerate), samplerate = samplerate,
                channels = 1, dtype = 'int16')
    sd.wait()

    wav.write(filename, samplerate, audio)
    return filename

if __name__ == "__main__":
    while True:
        file_path = record()

        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=open(file_path, "rb"),
            response_format="verbose_json",
        )

        print("Language:",response.language)
        print("Text:",response.text)
        print("-" * 50)