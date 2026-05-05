from __future__ import annotations

import os

from dotenv import load_dotenv
import requests
import streamlit as st


load_dotenv()

API_URL = os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8010").rstrip("/")

st.set_page_config(page_title="OpenRouter Interview Analyzer", layout="wide")

st.title("OpenRouter Interview Analyzer")
st.markdown("Upload an interview recording, transcribe it with Whisper, and evaluate the transcript with OpenRouter free models.")

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

upload_tab, text_tab, info_tab = st.tabs(["Upload Media", "Analyze Text", "API Info"])


def render_analysis(result: dict) -> None:
    analysis = result.get("analysis", {})

    top_metrics = st.columns(4)
    top_metrics[0].metric("Overall Rating", analysis.get("overall_rating", "N/A"))
    top_metrics[1].metric("Hire Recommendation", analysis.get("hire_recommendation", "N/A"))
    top_metrics[2].metric("WPM", f"{result.get('wpm', 0.0):.2f}")
    top_metrics[3].metric("Fillers", int(result.get("filler_count", 0)))

    if "duration_seconds" in result:
        meta_columns = st.columns(4)
        meta_columns[0].metric("Duration", f"{result.get('duration_seconds', 0.0):.2f}s")
        meta_columns[1].metric("Communication", analysis.get("communication_score", 0))
        meta_columns[2].metric("Content", analysis.get("content_score", 0))
        meta_columns[3].metric("Confidence", analysis.get("confidence_score", 0))
    else:
        meta_columns = st.columns(3)
        meta_columns[0].metric("Communication", analysis.get("communication_score", 0))
        meta_columns[1].metric("Content", analysis.get("content_score", 0))
        meta_columns[2].metric("Confidence", analysis.get("confidence_score", 0))

    st.subheader("Summary")
    st.write(analysis.get("summary", "No summary returned."))

    section_columns = st.columns(3)
    with section_columns[0]:
        st.subheader("Strengths")
        for item in analysis.get("strengths", []):
            st.success(item)
    with section_columns[1]:
        st.subheader("Concerns")
        for item in analysis.get("concerns", []):
            st.warning(item)
    with section_columns[2]:
        st.subheader("Suggestions")
        for item in analysis.get("suggestions", []):
            st.info(item)

    with st.expander("Transcript"):
        st.text_area("Transcript", value=result.get("transcription", ""), height=220, disabled=True)

    with st.expander("Model Details"):
        st.json(
            {
                "model_requested": analysis.get("model_requested"),
                "model_used": analysis.get("model_used"),
                "usage": analysis.get("usage", {}),
                "whisper_model": result.get("whisper_model"),
                "word_count": result.get("word_count"),
            }
        )


with upload_tab:
    st.header("Upload Interview Media")
    file_col, options_col = st.columns([3, 2])

    with file_col:
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=["mp3", "wav", "mp4", "avi", "mov", "m4a", "webm", "ogg", "aac"],
        )

    with options_col:
        whisper_model = st.selectbox("Whisper Model", ["tiny", "base"], index=0)
        router_model = st.text_input(
            "OpenRouter Model",
            value="openrouter/free",
            key="upload_router_model",
            help="Use openrouter/free to route to a free available model.",
        )

    if uploaded_file is not None:
        st.success(f"Selected file: {uploaded_file.name}")
        if st.button("Transcribe and Analyze", key="analyze_upload"):
            with st.spinner("Transcribing media and calling OpenRouter..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "application/octet-stream")}
                    response = requests.post(
                        f"{API_URL}/api/analyze",
                        files=files,
                        data={"whisper_model": whisper_model, "router_model": router_model},
                        timeout=600,
                    )
                    response.raise_for_status()
                    st.session_state.upload_result = response.json()["data"]
                    st.success("Analysis complete.")
                except requests.RequestException as exc:
                    error_message = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Upload analysis failed: {error_message}")

    if st.session_state.upload_result:
        st.divider()
        render_analysis(st.session_state.upload_result)


with text_tab:
    st.header("Analyze a Transcript Directly")
    form_columns = st.columns(3)
    with form_columns[0]:
        wpm = st.number_input("Words Per Minute", min_value=0, max_value=300, value=130)
    with form_columns[1]:
        filler_count = st.number_input("Filler Count", min_value=0, max_value=100, value=0)
    with form_columns[2]:
        router_model = st.text_input("OpenRouter Model", value="openrouter/free", key="text_router_model")

    transcription = st.text_area("Transcript", height=200, placeholder="Paste the transcript here...")
    if st.button("Analyze Transcript", key="analyze_text"):
        if len(transcription.split()) < 3:
            st.error("Please enter at least 3 words.")
        else:
            with st.spinner("Sending transcript to OpenRouter..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/analyze-text",
                        json={
                            "transcription": transcription,
                            "wpm": wpm,
                            "filler_count": filler_count,
                            "router_model": router_model,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    render_analysis(response.json()["data"])
                except requests.RequestException as exc:
                    error_message = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Text analysis failed: {error_message}")


with info_tab:
    st.header("API Information")
    try:
        response = requests.get(f"{API_URL}/api/config", timeout=20)
        response.raise_for_status()
        payload = response.json()
        runtime = payload.get("runtime", {})

        columns = st.columns(4)
        columns[0].metric("API Key", "Configured" if runtime.get("openrouter_api_key_configured") else "Missing")
        columns[1].metric("FFmpeg", "Available" if runtime.get("ffmpeg_available") else "Missing")
        columns[2].metric("Whisper Default", runtime.get("default_whisper_model", "tiny"))
        columns[3].metric("Router Default", runtime.get("default_router_model", "openrouter/free"))

        st.subheader("Supported Upload Types")
        st.write(", ".join(payload.get("upload_types", [])))
    except requests.RequestException as exc:
        error_message = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Could not connect to the API: {error_message}")

st.divider()
st.caption(f"Backend API URL: {API_URL}")
