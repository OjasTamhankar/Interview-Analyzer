from __future__ import annotations

from services.openrouter_client import _candidate_models, _resolve_requested_model, extract_json_object, normalize_analysis_payload


def test_extract_json_object_handles_code_fences() -> None:
    payload = extract_json_object(
        """```json
        {"overall_rating":"Weak","hire_recommendation":"Do Not Recommend"}
        ```"""
    )
    assert payload["overall_rating"] == "Weak"


def test_normalize_analysis_payload_applies_defaults() -> None:
    normalized = normalize_analysis_payload({"summary": "Short summary"}, "Short summary")
    assert normalized["overall_rating"] == "Average"
    assert normalized["hire_recommendation"] == "Borderline"
    assert normalized["summary"] == "Short summary"
    assert normalized["communication_score"] == 5


def test_resolve_requested_model_prefers_env_when_model_missing(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    assert _resolve_requested_model(None) == "meta-llama/llama-3.3-70b-instruct:free"


def test_candidate_models_deduplicates_requested_and_fallback() -> None:
    assert _candidate_models("openrouter/free") == ["openrouter/free"]
