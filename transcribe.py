from groq import Groq
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os # Added import to manage the audio file

# It is best practice not to hardcode API keys directly in the script,
# but use environment variables instead.
# For simplicity, keeping it as you had it:
client = Groq(api_key="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")

def record(filename="audio.wav"):
    print(f"Recording '{filename}' for 5 seconds... Speak now.")
    duration = 5
    # 16000 Hz is recommended for Whisper model
    samplerate = 16000
    # Capture audio data
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='int16')
    # Wait until recording is finished
    sd.wait()
    # Write the numpy array data to a WAV file
    wav.write(filename, samplerate, audio)
    print(f"Finished recording '{filename}'.")
    return filename

while True:
    try:
        file_path = record()
        
        # Open the recorded file and send to Groq API
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        
        print("Language:", response.language)
        print("Text:", response.text)
        print("-" * 50)

        # Optional: remove the temporary audio file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Break the loop or continue based on the nature of the error
        break