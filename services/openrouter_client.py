from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests


DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
FALLBACK_OPENROUTER_MODELS = ["openrouter/free"]

SYSTEM_PROMPT = """
You are an expert interview evaluator.
Analyze the candidate transcript fairly and conservatively.
Return valid JSON only, with no markdown fences and no extra commentary.

Schema:
{
  "overall_rating": "Strong" | "Average" | "Weak",
  "hire_recommendation": "Recommend" | "Borderline" | "Do Not Recommend",
  "communication_score": integer from 1 to 10,
  "content_score": integer from 1 to 10,
  "confidence_score": integer from 1 to 10,
  "summary": string,
  "strengths": [string, string, string],
  "concerns": [string, string, string],
  "suggestions": [string, string, string]
}

Rules:
- Base the evaluation only on the transcript and metadata provided.
- Penalize vague language, missing ownership, no measurable outcomes, and filler-heavy speech.
- Reward clarity, ownership, structured thinking, and concrete impact.
- Do not claim that metrics are missing if the transcript explicitly mentions numbers, percentages, latency, accuracy, time saved, or other measurable outcomes.
- Return all of strengths, concerns, and suggestions as JSON arrays, even if there is only one item.
- Keep each list item concise and useful.
""".strip()


def build_user_prompt(transcription: str, wpm: float, filler_count: int) -> str:
    return (
        "Evaluate this interview answer.\n\n"
        f"Words per minute: {wpm}\n"
        f"Detected filler words: {filler_count}\n"
        "Transcript:\n"
        f"{transcription}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
        normalized = re.sub(r"\s*```$", "", normalized)

    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", normalized, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON object.")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("Model response contained invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def normalize_analysis_payload(payload: dict[str, Any], raw_text: str) -> dict[str, Any]:
    overall_rating = str(payload.get("overall_rating", "Average")).strip() or "Average"
    hire_recommendation = str(payload.get("hire_recommendation", "Borderline")).strip() or "Borderline"
    summary = str(payload.get("summary", raw_text.strip())).strip() or raw_text.strip()

    def normalize_score(name: str, default: int = 5) -> int:
        value = payload.get(name, default)
        try:
            return max(1, min(10, int(value)))
        except (TypeError, ValueError):
            return default

    def normalize_list(name: str) -> list[str]:
        value = payload.get(name, [])
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return items[:3]
        if isinstance(value, str) and value.strip():
            text = value.strip()
            split_items = [item.strip(" -\t") for item in re.split(r"(?:\r?\n|;)", text) if item.strip(" -\t")]
            if split_items:
                return split_items[:3]
        return []

    return {
        "overall_rating": overall_rating,
        "hire_recommendation": hire_recommendation,
        "communication_score": normalize_score("communication_score"),
        "content_score": normalize_score("content_score"),
        "confidence_score": normalize_score("confidence_score"),
        "summary": summary,
        "strengths": normalize_list("strengths"),
        "concerns": normalize_list("concerns"),
        "suggestions": normalize_list("suggestions"),
        "raw_model_text": raw_text.strip(),
    }


def _fallback_analysis(raw_text: str) -> dict[str, Any]:
    return {
        "overall_rating": "Average",
        "hire_recommendation": "Borderline",
        "communication_score": 5,
        "content_score": 5,
        "confidence_score": 5,
        "summary": raw_text,
        "strengths": [],
        "concerns": [],
        "suggestions": [],
    }


def _resolve_requested_model(model: str | None = None) -> str:
    requested = (model or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL).strip()
    return requested or DEFAULT_OPENROUTER_MODEL


def _candidate_models(model: str | None = None) -> list[str]:
    requested = _resolve_requested_model(model)
    candidates = [requested, *FALLBACK_OPENROUTER_MODELS]
    deduplicated: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in deduplicated:
            deduplicated.append(normalized)
    return deduplicated


def analyze_transcript_with_openrouter(
    transcription: str,
    wpm: float,
    filler_count: int,
    model: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not configured.")

    api_base = os.getenv("OPENROUTER_API_BASE", DEFAULT_OPENROUTER_API_BASE).strip()
    requested_model = _resolve_requested_model(model)
    errors: list[str] = []

    for candidate_model in _candidate_models(model):
        logging.info(f"Calling OpenRouter with model {candidate_model}")

        response = requests.post(
            api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "OpenRouter Interview Analyzer",
            },
            json={
                "model": candidate_model,
                "temperature": 0.1,
                "max_tokens": 700,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(transcription, wpm, filler_count)},
                ],
            },
            timeout=timeout,
        )

        if response.status_code >= 400:
            logging.warning(f"OpenRouter error for {candidate_model}: HTTP {response.status_code}")
            errors.append(f"{candidate_model}: HTTP {response.status_code}")
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                continue
            response.raise_for_status()

        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            errors.append(f"{candidate_model}: no choices returned")
            continue

        raw_text = str(choices[0].get("message", {}).get("content", "")).strip()
        if not raw_text:
            errors.append(f"{candidate_model}: empty response")
            continue

        try:
            parsed = extract_json_object(raw_text)
        except ValueError:
            parsed = _fallback_analysis(raw_text)

        logging.info(f"Successfully parsed response from {candidate_model}")

        normalized = normalize_analysis_payload(parsed, raw_text)
        normalized["model_requested"] = requested_model
        normalized["model_used"] = str(payload.get("model", candidate_model))
        normalized["usage"] = payload.get("usage", {})
        if errors:
            normalized["fallback_attempts"] = errors
        return normalized

    error_text = "; ".join(errors) if errors else "Unknown OpenRouter error."
    raise RuntimeError(f"All OpenRouter model attempts failed. {error_text}")
