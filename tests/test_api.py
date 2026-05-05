from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import api
from api import _analysis_results, app


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_results() -> None:
    _analysis_results.clear()


class TestHealth:
    def test_health_endpoint_responds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["service"] == "OpenRouter Interview Analyzer API"
        assert payload["checks"]["openrouter_api_key_configured"] is True


class TestConfig:
    def test_config_endpoint_returns_runtime_metadata(self) -> None:
        response = client.get("/api/config")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert "runtime" in payload
        assert "upload_types" in payload
        assert "cors_allow_origins" in payload["runtime"]
        assert "cors_allow_origin_regex" in payload["runtime"]

    def test_resolve_cors_origins_supports_csv_and_wildcard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://app.example.com, https://admin.example.com")
        assert api._resolve_cors_origins() == ["https://app.example.com", "https://admin.example.com"]

        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
        assert api._resolve_cors_origins() == ["*"]

    def test_resolve_cors_origin_regex_prefers_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ALLOW_ORIGIN_REGEX", r"^https://preview\.example\.com$")
        assert api._resolve_cors_origin_regex() == r"^https://preview\.example\.com$"

    def test_preflight_allows_local_vite_origin(self) -> None:
        response = client.options(
            "/api/config",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


class TestFrontendServing:
    def test_root_serves_frontend_or_fallback_message(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "<div id=\"root\"></div>" in response.text or "OpenRouter Interview Analyzer API" in response.text


class TestTextAnalysis:
    def test_analyze_text_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_analyze_transcript_text(transcription: str, wpm: float | None, filler_count: int | None, router_model: str | None) -> dict:
            return {
                "transcription": transcription,
                "wpm": 120.0,
                "filler_count": 1,
                "analysis": {
                    "overall_rating": "Average",
                    "hire_recommendation": "Borderline",
                    "communication_score": 6,
                    "content_score": 5,
                    "confidence_score": 5,
                    "summary": "The answer is understandable but not compelling.",
                    "strengths": ["Clear enough to follow."],
                    "concerns": ["Not specific enough."],
                    "suggestions": ["Add measurable impact."],
                    "model_requested": router_model or "openrouter/free",
                    "model_used": "openrouter/free",
                    "usage": {"total_tokens": 10},
                },
            }

        monkeypatch.setattr(api, "analyze_transcript_text", fake_analyze_transcript_text)

        response = client.post(
            "/api/analyze-text",
            json={"transcription": "I worked on a project and delivered my assigned tasks on time.", "wpm": 120, "filler_count": 1},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["data"]["analysis"]["overall_rating"] == "Average"

    def test_analyze_text_rejects_short_transcript(self) -> None:
        response = client.post("/api/analyze-text", json={"transcription": "Too short"})
        assert response.status_code == 400
        assert "at least 3 words" in response.json()["detail"]

    def test_result_endpoint_returns_stored_analysis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_analyze_transcript_text(transcription: str, wpm: float | None, filler_count: int | None, router_model: str | None) -> dict:
            return {
                "transcription": transcription,
                "wpm": float(wpm or 130.0),
                "filler_count": int(filler_count or 0),
                "analysis": {
                    "overall_rating": "Strong",
                    "hire_recommendation": "Recommend",
                    "communication_score": 8,
                    "content_score": 8,
                    "confidence_score": 7,
                    "summary": "Clear answer with measurable impact.",
                    "strengths": ["Strong ownership."],
                    "concerns": ["Could be more detailed."],
                    "suggestions": ["Add scale context."],
                    "model_requested": router_model or "openrouter/free",
                    "model_used": "openrouter/free",
                    "usage": {"total_tokens": 12},
                },
            }

        monkeypatch.setattr(api, "analyze_transcript_text", fake_analyze_transcript_text)

        create_response = client.post(
            "/api/analyze-text",
            json={"transcription": "I led the migration and improved reliability across the service.", "wpm": 125, "filler_count": 0},
        )

        result_id = create_response.json()["result_id"]
        fetch_response = client.get(f"/api/result/{result_id}")

        assert fetch_response.status_code == 200
        assert fetch_response.json()["data"]["analysis"]["overall_rating"] == "Strong"


class TestUploadAnalysis:
    def test_upload_analysis_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_transcribe_media(input_path: str, whisper_model: str = "tiny") -> dict:
            return {
                "transcription": "I led a migration and reduced latency by thirty percent.",
                "duration_seconds": 14.0,
                "word_count": 10,
                "wpm": 110.0,
                "filler_count": 0,
            }

        def fake_analyze_transcript_text(transcription: str, wpm: float | None, filler_count: int | None, router_model: str | None) -> dict:
            return {
                "transcription": transcription,
                "wpm": float(wpm or 0.0),
                "filler_count": int(filler_count or 0),
                "analysis": {
                    "overall_rating": "Strong",
                    "hire_recommendation": "Recommend",
                    "communication_score": 8,
                    "content_score": 8,
                    "confidence_score": 7,
                    "summary": "Strong ownership and impact.",
                    "strengths": ["Clear ownership."],
                    "concerns": ["Could add more detail."],
                    "suggestions": ["Mention stakeholder impact."],
                    "model_requested": router_model or "openrouter/free",
                    "model_used": "openrouter/free",
                    "usage": {"total_tokens": 12},
                },
            }

        monkeypatch.setattr(api, "transcribe_media", fake_transcribe_media)
        monkeypatch.setattr(api, "analyze_transcript_text", fake_analyze_transcript_text)

        response = client.post(
            "/api/analyze",
            files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
            data={"whisper_model": "tiny", "router_model": "openrouter/free"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["data"]["analysis"]["overall_rating"] == "Strong"
        assert payload["data"]["whisper_model"] == "tiny"

    def test_upload_analysis_rejects_empty_file(self) -> None:
        response = client.post("/api/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
