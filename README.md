# OpenRouter Interview Analyzer

A FastAPI + React app that analyzes interview responses from either uploaded media or pasted transcript text.

## What It Does

1. Accepts audio, video, or transcript text input
2. Transcribes media locally with Whisper
3. Sends the transcript to OpenRouter for structured evaluation
4. Returns scores, summary, strengths, concerns, and improvement suggestions

## Stack

- FastAPI backend
- React + Vite frontend
- Whisper for local transcription
- FFmpeg for media conversion
- OpenRouter for interview evaluation

## Project Layout

```text
openrouter_interview_analyzer/
|-- api.py
|-- services/
|   |-- audio.py
|   |-- openrouter_client.py
|   `-- pipeline.py
|-- tests/
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- dist/
|-- requirements.txt
|-- requirements-dev.txt
`-- .env.example
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg available on `PATH`
- An `OPENROUTER_API_KEY`

## Setup

Install backend dependencies:

```powershell
pip install -r requirements-dev.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Example configuration:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
WHISPER_MODEL=tiny
API_HOST=0.0.0.0
API_PORT=8010
PORT=8010
CORS_ALLOW_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

`openrouter/free` is the safest default for this project because OpenRouter automatically routes to an available free model instead of depending on one fixed free model staying available.

## Local Development

Build the frontend for backend serving:

```powershell
cd frontend
npm run build
cd ..
```

Start the API server:

```powershell
python api.py
```

Open `http://127.0.0.1:8010`.

For frontend hot reload during development:

```powershell
cd frontend
npm run dev
```

The Vite dev server proxies `/api` and `/health` to `http://127.0.0.1:8010`.

## Verification

Run backend tests:

```powershell
pytest -q
```

Build the frontend:

```powershell
cd frontend
npm run build
```

## API Endpoints

- `GET /health`
- `GET /api/config`
- `POST /api/analyze`
- `POST /api/analyze-text`
- `GET /api/result/{result_id}`

## Deployment Notes

- The FastAPI app serves the built React app automatically from `frontend/dist`.
- Set `CORS_ALLOW_ORIGINS` to your deployed frontend origin if the frontend is hosted separately.
- Keep `OPENROUTER_API_KEY` out of git; `.env` is already ignored.
- The service reports `healthy` only when both the OpenRouter API key and FFmpeg are available.
