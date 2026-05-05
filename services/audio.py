from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import whisper


FILLER_TERMS = [
    "um",
    "uh",
    "like",
    "you know",
    "basically",
    "actually",
    "literally",
    "kind of",
    "sort of",
    "i mean",
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", clean_text(text)))


def count_filler_words(text: str) -> int:
    lowered = clean_text(text).lower()
    total = 0
    for filler in FILLER_TERMS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        total += len(re.findall(pattern, lowered))
    return total


def compute_wpm(word_count: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return round(word_count / (duration_seconds / 60.0), 2)


def convert_to_wav(input_path: str | Path, output_path: str | Path | None = None, sample_rate: int = 16000) -> Path:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg is required but was not found in PATH.")

    if output_path is None:
        output_path = source.with_suffix(".wav")

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg conversion failed: {exc.stderr.strip()}") from exc

    return target


def ensure_wav_audio(input_path: str | Path) -> Path:
    source = Path(input_path).expanduser().resolve()
    if source.suffix.lower() == ".wav":
        return source

    temp_dir = Path(tempfile.gettempdir()) / "openrouter_interview_analyzer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return convert_to_wav(source, temp_dir / f"{source.stem}_converted.wav")


@lru_cache(maxsize=2)
def load_whisper_model(model_name: str = "tiny") -> Any:
    return whisper.load_model(model_name)


def transcribe_media(input_path: str | Path, whisper_model: str = "tiny") -> dict[str, float | int | str]:
    logging.info(f"Starting transcription of {input_path} with model {whisper_model}")

    wav_path = ensure_wav_audio(input_path)
    logging.info(f"Converted to WAV: {wav_path}")

    audio, sample_rate = librosa.load(str(wav_path), sr=16000)
    if audio is None or len(audio) == 0:
        raise ValueError(f"Unable to decode audio from: {wav_path}")

    duration = round(float(librosa.get_duration(y=audio, sr=sample_rate)), 2)
    if duration <= 0:
        raise ValueError("Audio duration is zero. Please upload a valid media file.")

    logging.info(f"Audio loaded: duration {duration}s, sample_rate {sample_rate}")

    model = load_whisper_model(whisper_model)
    logging.info("Whisper model loaded, starting transcription")

    transcription_result = model.transcribe(str(wav_path), fp16=False)
    transcription = clean_text(transcription_result.get("text", ""))
    word_count = count_words(transcription)
    filler_count = count_filler_words(transcription)

    logging.info(f"Transcription complete: {word_count} words, {filler_count} fillers")

    return {
        "transcription": transcription,
        "duration_seconds": duration,
        "word_count": word_count,
        "wpm": compute_wpm(word_count, duration) if word_count else 0.0,
        "filler_count": filler_count,
    }
