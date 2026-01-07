import os
import json
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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



## S3 Service
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
                    try:
                        obj_resp = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
                        content = json.loads(obj_resp['Body'].read().decode('utf-8'))
                        
                        title = content.get("title")
                        
                        if not title:
                            messages = content.get("messages", [])
                            if messages and len(messages) > 0:
                                first_content = messages[0].get("content", "")
                                # title = " ".join(first_content.split()[:5]) + "..."
                                title = generate_title(first_content[:50])
                        
                        if not title:
                            title = f"Chat {chat_id[:8]}"
                            
                    except Exception as e:
                        print(f"Error reading chat {chat_id}: {e}")
                        title = f"Chat {chat_id[:8]}"

                    chats.append({
                        "id": chat_id,
                        "title": title,
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

def save_chat(chat_id: str, chat_data: dict):
    s3 = get_s3_client()
    try:
        key = f"{chat_id}.json"
        s3.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=key,
            Body=json.dumps(chat_data),
            ContentType='application/json'
        )
        return True
    except ClientError as e:
        print(f"Error saving chat {chat_id}: {e}")
        return False






groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"

def get_groq_response_stream(messages: list):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.7,
            stream=True
        )
        return chat_completion
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return None

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


def generate_title(user_message: str):
    """
    Generate a short 2-4 word title using Groq based on the first user message or based upon the file(PDF/txt) uploaded by user.
    """
    try:
        messages = [
            {"role": "system", "content": "Generate a very short, concise topic title (2-4 words maximum) for a chat starting with this message. Return ONLY the title, no quotes, no extra text."},
            {"role": "user", "content": user_message}
        ]
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=20
        )
        title = chat_completion.choices[0].message.content.strip().replace('"', '')
        return title
    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback to simple heuristic
        return " ".join(user_message.split()[:4]) + "..."



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
    # payload expected: {"chat_id": "...", "message": "...", "history": [...], "title": "..."}
    chat_id = payload.get("chat_id")
    if not chat_id:
        chat_id = str(uuid.uuid4())
    
    current_history = payload.get("history", [])
    user_message = payload.get("message")
    
    current_history.append({"role": "user", "content": user_message})

    async def response_generator():
        stream = get_groq_response_stream(current_history)
        if stream is None:
            yield json.dumps({"type": "error", "content": "Failed to connect to AI service."}) + "\n"
            return

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_response += content_chunk
                # Yield chunk as JSON line
                yield json.dumps({"type": "chunk", "content": content_chunk}) + "\n"
        
        # After stream finishes, handle side effects
        
        # 3. Add AI message
        current_history.append({"role": "assistant", "content": full_response})
        
        # 4. Determine Title (if new or not provided)
        title = payload.get("title")
        # Generate title only if it's the very first exchange (len <= 2) and no title exists
        new_title_generated = False
        if not title and len(current_history) <= 4: 
            document_name = payload.get("document_name")
            if document_name:
                title = os.path.splitext(document_name)[0]
            else:
                title = generate_title(user_message)
            new_title_generated = True
        
        # 5. Save to S3
        chat_data = {
            "title": title,
            "messages": current_history
        }
        save_chat(chat_id, chat_data)
        
        # Yield metadata
        yield json.dumps({
            "type": "complete", 
            "chat_id": chat_id, 
            "title": title,
            "new_title_generated": new_title_generated
            # We don't necessarily need to send the whole history back if the frontend manages it,
            # but it is good practice to sync.
        }) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
