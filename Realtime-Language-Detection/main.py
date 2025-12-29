# main.py - FastAPI Backend, Analytics & Reporting

import os
import io
import time
import numpy as np
import scipy.io.wavfile as wav
import pyaudio
import sqlite3
import nltk
import pandas as pd
import json

from fastapi import FastAPI, Request, UploadFile, File, Form # UploadFile इम्पोर्ट किया गया
from fastapi.responses import HTMLResponse, JSONResponse, Response 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ReportLab इम्पोर्ट्स
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from groq import Groq
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLTK डेटा डाउनलोड (भावना विश्लेषण के लिए आवश्यक)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

load_dotenv()

# --- कॉन्फ़िगरेशन और इनिशियलाइज़ेशन ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")

DURATION = int(os.getenv("AUDIO_DURATION", 7))
SAMPLERATE = int(os.getenv("AUDIO_SAMPLERATE", 16000))
AUDIO_FILENAME = "temp_audio.wav"

client = Groq(api_key=GROQ_API_KEY)
vader_analyzer = SentimentIntensityAnalyzer()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- यूटिलिटी फंक्शन्स ---

def init_db():
    """SQLite डेटाबेस को शुरू करता है और 'history' टेबल बनाता है।"""
    # ... (Code remains the same) ...
    conn = sqlite3.connect('transcriptions.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            text TEXT,
            language TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            pitch_data REAL
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_db()

def analyze_pitch(audio_data, samplerate):
    """ऑडियो डेटा से पिच प्रॉक्सी (ZCR) का अनुमान लगाता है।"""
    # ... (Code remains the same) ...
    if len(audio_data) == 0: return 0.0
    audio_float = audio_data.astype(np.float32)
    zcr = np.mean(np.abs(np.diff(np.sign(audio_float)))) * samplerate / 2
    return float(zcr)

def get_sentiment(text):
    """VADER का उपयोग करके भावना (Sentiment) और स्कोर की गणना करता है।"""
    # ... (Code remains the same) ...
    vs = vader_analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05: sentiment = "Positive (सकारात्मक) 😊"
    elif vs['compound'] <= -0.05: sentiment = "Negative (नकारात्मक) 😠"
    else: sentiment = "Neutral (तटस्थ) 😐"
    return sentiment, vs['compound']

def save_to_db(text, language, sentiment, score, pitch):
    """विश्लेषण परिणामों को SQLite डेटाबेस में सेव करता है।"""
    # ... (Code remains the same) ...
    conn = sqlite3.connect('transcriptions.db')
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO history (timestamp, text, language, sentiment, sentiment_score, pitch_data) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, text, language, sentiment, score, pitch)
    )
    conn.commit()
    conn.close()

def record_audio(duration, samplerate, channels=1, dtype=np.int16):
    """PyAudio का उपयोग करके लाइव रिकॉर्डिंग करता है।"""
    # ... (Code remains the same - PyAudio logic) ...
    p = pyaudio.PyAudio()
    chunk = 1024
    
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=samplerate,
                        input=True,
                        frames_per_buffer=chunk)
    except OSError as e:
        p.terminate()
        raise ValueError(f"PyAudio Recording Error (Check Microphone/Drivers): {e}")

    frames = []
    for _ in range(0, int(samplerate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=dtype)
    return audio_data

# --- API एंडपॉइंट्स ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """होमपेज (रिकॉर्डर और अपलोड इंटरफ़ेस) रेंडर करें।"""
    return templates.TemplateResponse("index.html", {"request": request, "duration": DURATION})

# --- नया एंडपॉइंट: फाइल अपलोड द्वारा ट्रांसक्रिप्शन ---
@app.post("/transcribe-file-upload")
async def transcribe_file_upload(audio_file: UploadFile = File(...)):
    """अपलोड की गई ऑडियो फ़ाइल (WAV) को Groq को भेजता है और विश्लेषण करता है।"""
    
    # 1. फ़ाइल को बफ़र में पढ़ें
    file_content = await audio_file.read()
    
    # सुनिश्चित करें कि यह WAV फ़ाइल है (या PyAudio/Whisper-compatible)
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
         return JSONResponse({"success": False, "error": "Invalid file format. Please upload WAV, MP3, or FLAC."}, status_code=400)
    
    # 2. पिच एनालिसिस (NumPy array के रूप में)
    try:
        # फ़ाइल को बफ़र से WAV डेटा के रूप में पढ़ें (scipy का उपयोग करके)
        samplerate, audio_data = wav.read(io.BytesIO(file_content))
        
        # यदि यह स्टीरियो है, तो एक चैनल लें
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        pitch_proxy = analyze_pitch(audio_data, samplerate)
        
    except Exception as e:
        print(f"Error reading audio file for pitch analysis: {e}")
        pitch_proxy = 0.0 # यदि पिच एनालिसिस विफल हो जाए तो 0.0 सेट करें

    # 3. Groq ट्रांसक्रिप्शन
    try:
        # BytesIO का उपयोग करके फ़ाइल को Groq को सीधे स्ट्रीम करें (डिस्क पर सेव किए बिना)
        file_like_object = io.BytesIO(file_content)
        file_like_object.name = audio_file.filename # Groq को फ़ाइल नाम की आवश्यकता है
        
        response = client.audio.transcriptions.create(
            file=file_like_object,
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        
        # 4. भावना विश्लेषण
        sentiment, score = get_sentiment(response.text)
        
        # 5. डेटाबेस में सेव करें
        save_to_db(response.text, response.language, sentiment, score, pitch_proxy)

        # 6. JSON रिस्पॉन्स
        return JSONResponse({
            "success": True,
            "language": response.language,
            "text": response.text,
            "sentiment": sentiment,
            "sentiment_score": score,
            "pitch_proxy": f"{pitch_proxy:.2f}",
            "gender_hint": "Low Pitch (Male Hint)" if pitch_proxy < 1000 else "High Pitch (Female Hint)"
        })

    except Exception as e:
        print(f"Error during Groq transcription or analysis: {e}")
        return JSONResponse({"success": False, "error": f"Transcription/API Error: {str(e)}"}, status_code=500)


# --- मौजूदा लाइव ट्रांसक्रिप्शन एंडपॉइंट ---
@app.post("/transcribe-live")
async def transcribe_live():
    """माइक्रोफ़ोन से ऑडियो रिकॉर्ड करता है, Groq को भेजता है, और विश्लेषण करता है।"""
    
    if os.path.exists(AUDIO_FILENAME): os.remove(AUDIO_FILENAME)
    
    try:
        # 1. रिकॉर्डिंग और पिच एनालिसिस
        audio = record_audio(DURATION, SAMPLERATE)
        pitch_proxy = analyze_pitch(audio, SAMPLERATE)
        wav.write(AUDIO_FILENAME, SAMPLERATE, audio)

        # 2. Groq ट्रांसक्रिप्शन
        with open(AUDIO_FILENAME, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        
        os.remove(AUDIO_FILENAME)

        # 3. भावना विश्लेषण और डेटाबेस सेव
        sentiment, score = get_sentiment(response.text)
        save_to_db(response.text, response.language, sentiment, score, pitch_proxy)

        # 4. JSON रिस्पॉन्स
        return JSONResponse({
            "success": True,
            "language": response.language,
            "text": response.text,
            "sentiment": sentiment,
            "sentiment_score": score,
            "pitch_proxy": f"{pitch_proxy:.2f}",
            "gender_hint": "Low Pitch (Male Hint)" if pitch_proxy < 1000 else "High Pitch (Female Hint)"
        })

    except ValueError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    
    except Exception as e:
        print(f"Error during live transcription: {e}")
        if os.path.exists(AUDIO_FILENAME): os.remove(AUDIO_FILENAME)
        return JSONResponse({"success": False, "error": f"API/General Error: {str(e)}"}, status_code=500)

@app.get("/history")
async def get_history():
    """अंतिम 10 ट्रांसक्रिप्शन हिस्ट्री लौटाता है।"""
    # ... (Code remains the same) ...
    conn = sqlite3.connect('transcriptions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, text, language, sentiment, sentiment_score FROM history ORDER BY id DESC LIMIT 10")
    history = cursor.fetchall()
    conn.close()
    
    results = []
    for row in history:
        results.append({
            "timestamp": row[0],
            "text": row[1],
            "language": row[2],
            "sentiment": row[3],
            "sentiment_score": row[4]
        })
    return JSONResponse(results)

# --- डैशबोर्ड और रिपोर्टिंग एंडपॉइंट्स ---

@app.get("/dashboard_data")
async def get_dashboard_data():
    # ... (Code remains the same) ...
    conn = sqlite3.connect('transcriptions.db')
    df = pd.read_sql_query("SELECT * FROM history", conn)
    conn.close()

    if df.empty:
        return JSONResponse({"success": False, "error": "No data available."})

    sentiment_counts = df['sentiment'].str.split(' ').str[0].value_counts().to_dict()
    avg_score = df['sentiment_score'].mean()
    
    pitch_counts = (df['pitch_data'] > 1000).value_counts()
    gender_counts = {
        "High_Pitch": pitch_counts.get(True, 0),
        "Low_Pitch": pitch_counts.get(False, 0)
    }

    return JSONResponse({
        "success": True,
        "total_records": len(df),
        "sentiment_breakdown": sentiment_counts,
        "average_sentiment_score": f"{avg_score:.3f}",
        "gender_breakdown": gender_counts
    })

# main.py में:

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """डैशबोर्ड पेज रेंडर करें।"""
    # सुनिश्चित करें कि 'templates/' फोल्डर और 'dashboard.html' फाइल मौजूद है
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/report")
async def generate_report():
    # ... (Code remains the same) ...
    conn = sqlite3.connect('transcriptions.db')
    df = pd.read_sql_query("SELECT timestamp, text, language, sentiment, sentiment_score FROM history ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        return HTMLResponse("<h1>No data to generate report.</h1>", status_code=404)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Advanced Audio Analytics Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Transcriptions', str(len(df))],
        ['Average Sentiment Score', f"{df['sentiment_score'].mean():.3f}"],
        ['Most Common Sentiment', df['sentiment'].str.split(' ').str[0].mode().iloc[0] if not df['sentiment'].empty else 'N/A']
    ]
    table = Table(summary_data, colWidths=[150, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(Paragraph("<b>1. Summary Statistics</b>", styles['Heading2']))
    elements.append(table)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("<b>2. Detailed Transcription History (Last 10)</b>", styles['Heading2']))
    history_data = [["Timestamp", "Text Snippet", "Sentiment"]]
    for index, row in df.head(10).iterrows(): 
        history_data.append([
            row['timestamp'], 
            row['text'][:50] + "...", 
            f"{row['sentiment'].split(' ')[0]} ({row['sentiment_score']:.2f})"
        ])
    
    history_table = Table(history_data, colWidths=[100, 250, 100])
    history_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(history_table)
    elements.append(Spacer(1, 12))


    doc.build(elements)
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="application/pdf", 
                    headers={"Content-Disposition": "attachment; filename=Analysis_Report.pdf"})