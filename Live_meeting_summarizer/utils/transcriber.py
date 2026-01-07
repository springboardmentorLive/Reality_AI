import speech_recognition as sr
import io
from pydub import AudioSegment

def transcribe_audio(audio_bytes):
    """
    Transcribes audio bytes to text using SpeechRecognition.
    """
    try:
        # Convert audio bytes to a file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        # Use pydub to convert to wav if necessary (SpeechRecognition likes wav)
        # Assuming input might be webm or other browser formats, 
        # but let's try direct processing or simple conversion first.
        # For robustness, let's try to decode with pydub and export to wav
        try:
            audio = AudioSegment.from_file(audio_file)
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            source_file = wav_io
        except Exception as e:
            print(f"Error converting audio: {e}")
            return f"Error processing audio format: {e}"

        recognizer = sr.Recognizer()
        with sr.AudioFile(source_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"An error occurred: {e}"
