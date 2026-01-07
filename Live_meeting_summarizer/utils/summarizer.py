from groq import Groq

def summarize_meeting(transcript, api_key):
    """
    Summarizes the meeting transcript using Groq API (Llama 3).
    """
    if not api_key:
        return "Please provide a valid Groq API Key."

    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are a professional meeting secretary. 
        Please summarize the following meeting transcript.
        
        Transcript:
        {transcript}
        
        Structure your response into the following sections:
        1. **Executive Summary**: A brief overview of the meeting.
        2. **Key Discussion Points**: Bullet points of main topics discussed.
        3. **Action Items**: A list of tasks assigned, with owners if mentioned.
        4. **Decisions Made**: Any clear decisions or agreements reached.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred during summarization: {e}"
