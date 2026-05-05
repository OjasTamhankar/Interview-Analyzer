from __future__ import annotations

from collections import OrderedDict
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from services.audio import count_filler_words, count_words, transcribe_media
from services.pipeline import analyze_transcript_text

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST_DIR / "index.html"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny").strip() or "tiny"
DEFAULT_ROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free"
MAX_STORED_RESULTS = 100
DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8010",
    "http://localhost:8010",
    "https://localhost:3000",
    "https://*.vercel.app",
    "https://*.netlify.app",
]


class TextAnalysisRequest(BaseModel):
    transcription: str = Field(..., description="Interview transcript text.")
    wpm: float | None = Field(default=None, ge=0, description="Optional words per minute.")
    filler_count: int | None = Field(default=None, ge=0, description="Optional filler word count.")
    router_model: str | None = Field(default=None, description="OpenRouter model or router name.")


def _resolve_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not configured:
        return DEFAULT_CORS_ALLOW_ORIGINS.copy()
    if configured == "*":
        return ["*"]

    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ALLOW_ORIGINS.copy()


CORS_ALLOW_ORIGINS = _resolve_cors_origins()
CORS_ALLOW_CREDENTIALS = CORS_ALLOW_ORIGINS != ["*"]


app = FastAPI(
    title="OpenRouter Interview Analyzer API",
    description="Upload media, transcribe with Whisper, and evaluate the transcript with OpenRouter free models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_analysis_results: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _store_result(result: dict[str, Any]) -> str:
    result_id = uuid.uuid4().hex
    if len(_analysis_results) >= MAX_STORED_RESULTS:
        _analysis_results.popitem(last=False)
    _analysis_results[result_id] = result
    return result_id


def _build_runtime_status() -> dict[str, Any]:
    api_key_present = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    return {
        "openrouter_api_key_configured": api_key_present,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "frontend_available": FRONTEND_INDEX.exists(),
        "frontend_assets_available": FRONTEND_ASSETS_DIR.exists(),
        "default_whisper_model": DEFAULT_WHISPER_MODEL,
        "default_router_model": DEFAULT_ROUTER_MODEL,
        "cors_allow_origins": CORS_ALLOW_ORIGINS,
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    runtime = _build_runtime_status()
    healthy = runtime["openrouter_api_key_configured"] and runtime["ffmpeg_available"]
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "OpenRouter Interview Analyzer API",
        "version": app.version,
        "checks": runtime,
    }


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "status": "success",
        "runtime": _build_runtime_status(),
        "upload_types": ["mp3", "wav", "mp4", "avi", "mov", "m4a", "webm", "ogg", "aac"],
    }


@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    whisper_model: str | None = Form(default=None, description="Whisper model name."),
    router_model: str | None = Form(default=None, description="OpenRouter model or router name."),
) -> dict[str, Any]:
    resolved_whisper_model = str(whisper_model or DEFAULT_WHISPER_MODEL).strip()
    resolved_router_model = str(router_model or DEFAULT_ROUTER_MODEL).strip()

    logging.info(f"Starting analysis for file {file.filename} with whisper_model={resolved_whisper_model}, router_model={resolved_router_model}")

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    suffix = Path(file.filename).suffix or ".bin"
    temp_path: Path | None = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        logging.info(f"File saved to {temp_path}, starting transcription")

        transcription_result = transcribe_media(temp_path, whisper_model=resolved_whisper_model)

        logging.info(f"Transcription completed: {len(transcription_result['transcription'])} chars, duration {transcription_result['duration_seconds']}s, WPM {transcription_result['wpm']}")

        result = analyze_transcript_text(
            transcription=str(transcription_result["transcription"]),
            wpm=float(transcription_result["wpm"]),
            filler_count=int(transcription_result["filler_count"]),
            router_model=resolved_router_model,
        )
        result["duration_seconds"] = float(transcription_result["duration_seconds"])
        result["word_count"] = int(transcription_result["word_count"])
        result["media_filename"] = file.filename
        result["whisper_model"] = resolved_whisper_model

        logging.info("Analysis completed successfully")

        result_id = _store_result(result)
        return {"status": "success", "result_id": result_id, "data": result}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/analyze-text")
async def analyze_text(payload: TextAnalysisRequest) -> dict[str, Any]:
    transcription = payload.transcription.strip()
    if count_words(transcription) < 3:
        raise HTTPException(status_code=400, detail="Transcription must contain at least 3 words.")

    logging.info(f"Starting text analysis for {len(transcription)} chars")

    try:
        result = analyze_transcript_text(
            transcription=transcription,
            wpm=payload.wpm,
            filler_count=payload.filler_count if payload.filler_count is not None else count_filler_words(transcription),
            router_model=payload.router_model,
        )
        logging.info("Text analysis completed successfully")
        result_id = _store_result(result)
        return {"status": "success", "result_id": result_id, "data": result}
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/api/result/{result_id}")
async def get_result(result_id: str) -> dict[str, Any]:
    result = _analysis_results.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    return {"status": "success", "data": result}


if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name="assets")


@app.get("/", include_in_schema=False)
async def root() -> Any:
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    return {
        "message": "OpenRouter Interview Analyzer API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> Any:
    if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi") or full_path.startswith("redoc"):
        raise HTTPException(status_code=404, detail="Not found.")
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    raise HTTPException(status_code=404, detail="Frontend not built.")


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("API_PORT", "8010"))),
        reload=False,
    )
