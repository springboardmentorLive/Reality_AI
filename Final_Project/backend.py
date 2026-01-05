import os
import json
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uuid
import boto3
from typing import List, Optional
from botocore.exceptions import ClientError


load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



## S3
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def list_chats():
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME)
        chats = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.json'):
                    chat_id = key.replace('.json', '')
                    chats.append({
                        "id": chat_id,
                        "last_modified": obj['LastModified'].isoformat()
                    })
        chats.sort(key=lambda x: x['last_modified'], reverse=True)
        return chats
    except ClientError as e:
        print(f"Error listing chats: {e}")
        return []

def load_chat(chat_id: str):
    s3 = get_s3_client()
    try:
        key = f"{chat_id}.json"
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        print(f"Error loading chat {chat_id}: {e}")
        return None

def save_chat(chat_id: str, messages: list):
    s3 = get_s3_client()
    try:
        key = f"{chat_id}.json"
        s3.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=key,
            Body=json.dumps(messages),
            ContentType='application/json'
        )
        return True
    except ClientError as e:
        print(f"Error saving chat {chat_id}: {e}")
        return False



# Groq
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"

def get_groq_response(messages: list):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, I encountered an error processing your request."



# =============
# API Endpoints
# =============
@app.get("/api/chats")
async def get_chats_endpoint():
    return list_chats()

@app.get("/api/chats/{chat_id}")
async def get_chat_endpoint(chat_id: str):
    chat = load_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@app.post("/api/chat")
async def chat_interaction(payload: dict):
    # payload expected: {"chat_id": "...", "message": "...", "history": [...]}
    chat_id = payload.get("chat_id")
    if not chat_id:
        chat_id = str(uuid.uuid4())
    
    current_history = payload.get("history", [])
    user_message = payload.get("message")
    
    current_history.append({"role": "user", "content": user_message})

    ai_response_content = get_groq_response(current_history)
    # 3. Add AI message
    current_history.append({"role": "assistant", "content": ai_response_content})
    # 4. Save to S3
    save_chat(chat_id, current_history)
    
    return {
        "response": ai_response_content, 
        "updated_history": current_history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
