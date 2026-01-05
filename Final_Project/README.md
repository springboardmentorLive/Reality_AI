## Springboard AI Chatbot

### Features
- **AI Model**: Inference from *`Groq`* (Moonshot AI Kimi K2).
- **Storage**: Chat history saved to `AWS S3`.
- **Frontend**: *`Streamlit`* for a clean, responsive UI.
- **Backend**: *`FastAPI`* for robust API handling.

---

### Setup

##### 1. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

##### 2. Environment Variables
Create a `.env` file in this directory and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_BUCKET_NAME=your_bucket_name
```

##### 3. Run Application
You need to run the backend and frontend in separate terminals.

> **Terminal 1 (Backend):**

Start the FastAPI server.
```bash
uvicorn backend:app --reload
```
*The backend will run on http://localhost:8000*

> **Terminal 2 (Frontend):**

Start the Streamlit interface.
```bash
streamlit run frontend.py
```
*The frontend will open in your browser at http://localhost:8501*

## Project Structure
- `frontend.py`: Contains the user interface code.
- `backend.py`: Contains all server logic, API endpoints, and **`S3`** connections.
- `requirements.txt`: List of Python libraries required.
