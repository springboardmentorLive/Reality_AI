from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("PUBLIC_GROQ_API_KEY"))


def transcribe_and_detect_language(file_path):
    try:
        # Open audio file in binary mode
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
            )

        transcription = response.text
        # Remove reference to `response.language` since it doesn't exist
        return transcription

    except Exception as e:
        print("Error:", e)
        return None


def detect_language_llm(text):
    """
    Optional: fallback using LLM to detect language in text accurately.
    """
    prompt = (
        "Identify the language of this text and return ISO 639-1 code + language name:\n"
        f"Text: {text}"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# -------------------- Main Loop --------------------
# -------------------- Main Loop --------------------
if __name__ == "__main__":
    while True:
        file_path = input("\nEnter path to MP3/M4A/WAV audio file (or type 'quit'): ").strip()

        if file_path.lower() == "quit":
            print("Goodbye!")
            break

        transcription = transcribe_and_detect_language(file_path)

        if transcription:
            print("\n---------- Transcribed Text ----------\n")
            print(transcription)

            # Use LLM to detect language
            print("\n---------- Detected Language (LLM Verification) ----------\n")
            language_llm = detect_language_llm(transcription)
            print(language_llm)

            print("\n----------------------------------------\n")