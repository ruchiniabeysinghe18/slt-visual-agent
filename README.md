# AI Support Chatbot

A WhatsApp and web-based AI customer support chatbot built with FastAPI, LangGraph, and OpenAI. Supports RAG over PDF documents and websites, consultant escalation, booking/service request handling, and speech-to-text.

---

## System Requirements

- Docker and Docker Compose installed
- Port `5001` open (FastAPI app)
- Port `27017` open if you need external MongoDB access

---

## Credentials and Configuration

### 1. `.env` file

Create a `.env` file in the project root with the following variables:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# MongoDB
DATABASE=SLT_AI_ASSIST
DB_HOST=localhost        # use "mongodb" when running via Docker Compose
DB_PORT=27017

# WhatsApp Business API
ACCESS_TOKEN=<whatsapp_permanent_access_token>
VERSION=v25.0
PHONE_NUMBER_ID=<whatsapp_phone_number_id>
VERIFY_TOKEN=<your_webhook_verify_token>

# Knowledge base selection — include "doc", "web", or both
KB=["doc", "web"]

# Agent phone numbers to notify on escalation
AGENTS=["94710844007"]

# Upload limits
DOCUMENT_LIMIT=5
WEBSITE_LIMIT=5

# Timezone
TIME_ZONE=Asia/Colombo

# Google Cloud credentials (path on host; Docker overrides this automatically)
GOOGLE_APPLICATION_CREDENTIALS=./sltaichallange-599b3b8646ad.json

# Text-to-speech
USE_TTS=true
TTS_PROVIDER=gemini        # "gemini" or "claude"
GEMINI_API_KEY=<gemini_api_key>
```

### 2. `sltaichallange-599b3b8646ad.json` — Google Cloud Service Account

This file is required for **Google Cloud Speech-to-Text** and **Text-to-Speech**.

To obtain it:
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → IAM & Admin → Service Accounts
2. Select or create a service account with the following roles:
   - `Cloud Speech-to-Text User`
   - `Cloud Text-to-Speech User`
3. Click **Keys** → **Add Key** → **Create new key** → **JSON**
4. Rename the downloaded file to `sltaichallange-599b3b8646ad.json`
5. Place it in the project root (same directory as `docker-compose.yml`)

> Docker Compose mounts this file into the container at `/app/sltaichallange-599b3b8646ad.json` and sets `GOOGLE_APPLICATION_CREDENTIALS` automatically — no extra configuration needed when using Docker.

---

## Docker Deployment

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

## API

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
| `/consultant_send_message` | POST | Send message as consultant |
| `/consultant_mode_switch` | POST | Toggle consultant mode |
| `/get_bookings` | POST | List service bookings |
| `/admin` | GET | Admin UI |
| `/app` | GET | Chat UI |
