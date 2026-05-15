# AI Support Chatbot

A web-based AI customer support chatbot built with FastAPI, LangGraph, OpenAI and Gemini. Supports RAG over PDF documents and websites, consultant escalation, service request handling in the fourm of tickets. speech support in English, Sinhala and Tamil. Web based realtime avatar id added to give an human appearance.

---

## Conditions

1. Uploading file must be text PDFs, **No OCR processing is implimented to support image PDFs**
2. Use chrome web-browser
4. If run in a VM ,avtar feature may not work  due to missing grafical libraiers and drivers, but chatbot will functional as usual
---

## System Requirements

- Docker and Docker Compose installed
- Port `5001` (fast API) and `27017`(MongoDB) avaialble

---
## Create a project folder with a suitable name

---
## Clone following repos in to the project folder
1. ai chatbot code : git clone https://github.com/ruchiniabeysinghe18/slt-visual-agent.git
2. visual avatar : git clone https://github.com/wass08/wawa-lipsync/

---

## Credentials

The `.env` file and Google Cloud service credentials (.json) are provided
---

## Docker Deployment (Recomended)

### Build and start

```bash
docker compose up --build -d
```

This starts two containers:
- **app** — FastAPI server on port `5001`
- **mongodb** — MongoDB 7 with a persistent volume

The app waits for MongoDB to be healthy before starting.

### Stop

```bash
docker compose down
```

### Stop and remove all data (including MongoDB volume)

```bash
docker compose down -v
```

### View logs

```bash
docker compose logs -f app
```

### Rebuild after code changes

```bash
docker compose up --build -d
```

---

## Persistent Volumes

The following directories are mounted from the host into the container so data survives restarts:

| Host path | Container path | Purpose |
|---|---|---|
| `./chat_uploads` | `/app/chat_uploads` | User file uploads |
| `./audio_tts` | `/app/audio_tts` | TTS audio output |
| `./avatar_videos` | `/app/avatar_videos` | Avatar video files |
| `./chroma_store` | `/app/chroma_store` | ChromaDB vector store |
| `./uploaded_pdfs` | `/app/uploaded_pdfs` | Uploaded PDF documents |
| `./logs` | `/app/logs` | Application logs |
| `mongo_data` (Docker volume) | `/data/db` | MongoDB data |

---

## Run with Python (without Docker)

### Prerequisites

- Python 3.11
- MongoDB running locally on port `27017`

### 1. Install dependencies

```bash
cd slt-visual-agent
pip install -r requirements.txt
```

### 2. Start the app

```bash
python main.py
```
---

## Web service

The APP will be available at `http://localhost:5001`.

- Chat UI: `http://localhost:5001/app`
- Admin UI: `http://localhost:5001/admin`

Admin panel will be used to 
- Upload files and websited to the knowledge base
- visualize tickets, later this can be connected with real ticketing system. 

---

## APIs

The app exposes a REST API at `http://localhost:5001`.

| Endpoint | Method | Description |
|---|---|---|
| `/common_chat/` | POST | Main chat endpoint (web portal) |
| `/webhook` | GET/POST | WhatsApp webhook |
| `/upload_pdf/` | POST | Upload a PDF to the knowledge base |
| `/upload_website/` | POST | Add a website URL to the knowledge base |
| `/list_pdf` | POST | List uploaded PDFs |
| `/delete_pdf` | POST | Delete a PDF |
| `/list_user_ids` | POST | List all users |
| `/get_conversation` | POST | Get chat history for a user |
| `/consultant_mode_switch` | POST | Toggle consultant mode |
| `/get_bookings` | POST | List service bookings |
| `/admin` | GET | Admin UI |
| `/app` | GET | Chat UI |
