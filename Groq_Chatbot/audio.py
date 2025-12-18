from groq import Groq
import sounddevice as sd
import numpy as np 
import scipy.io.wavfile as wav 

client = Groq(api_key="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")

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