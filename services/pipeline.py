from __future__ import annotations

import logging
from typing import Any

from services.audio import count_filler_words, count_words
from services.openrouter_client import analyze_transcript_with_openrouter


DEFAULT_WPM = 130.0


def analyze_transcript_text(
    transcription: str,
    wpm: float | None = None,
    filler_count: int | None = None,
    router_model: str | None = None,
) -> dict[str, Any]:
    normalized = " ".join(str(transcription or "").split()).strip()
    if count_words(normalized) < 3:
        raise ValueError("Transcription must contain at least 3 words.")

    resolved_wpm = float(DEFAULT_WPM if wpm is None else wpm)
    resolved_filler_count = int(count_filler_words(normalized) if filler_count is None else filler_count)

    logging.info(f"Sending to OpenRouter: {len(normalized)} chars, WPM {resolved_wpm}, fillers {resolved_filler_count}")

    analysis = analyze_transcript_with_openrouter(
        transcription=normalized,
        wpm=resolved_wpm,
        filler_count=resolved_filler_count,
        model=router_model,
    )

    logging.info("Received analysis from OpenRouter")

    return {
        "transcription": normalized,
        "wpm": resolved_wpm,
        "filler_count": resolved_filler_count,
        "analysis": analysis,
    }
