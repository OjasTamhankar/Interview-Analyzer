import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").trim() || window.location.origin;
const DEFAULT_ANALYSIS_MODEL = "openrouter/free";
const SUPPORTED_EXTENSIONS = ["mp3", "wav", "mp4", "avi", "mov", "m4a", "webm", "ogg", "aac"];

const TABS = [
  { id: "upload", label: "Upload Media" },
  { id: "text", label: "Analyze Text" },
  { id: "info", label: "API Info" },
];

function countWords(text) {
  const matches = String(text || "").match(/\b[\w']+\b/g);
  return matches ? matches.length : 0;
}

function formatScore(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "N/A";
}

function requestErrorMessage(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong.";
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(`Unexpected response from server: ${text}`);
    }
  }

  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : `Request failed with status ${response.status}`);
  }

  return payload;
}

function MetricCard({ label, value, tone = "neutral" }) {
  return (
    <div className={`metric-card metric-card--${tone}`}>
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
    </div>
  );
}

function StatusPill({ label, tone }) {
  return <span className={`status-pill status-pill--${tone}`}>{label}</span>;
}

function SectionCard({ title, eyebrow, children }) {
  return (
    <section className="section-card">
      {eyebrow ? <p className="section-card__eyebrow">{eyebrow}</p> : null}
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function AnalysisResults({ result }) {
  const analysis = result?.analysis || {};
  const ratingTone = String(analysis.overall_rating || "").toLowerCase() === "strong"
    ? "good"
    : String(analysis.overall_rating || "").toLowerCase() === "weak"
      ? "danger"
      : "warn";

  return (
    <div className="results-stack">
      <div className="hero-result">
        <div>
          <p className="eyebrow">Interview Outcome</p>
          <h2>{analysis.overall_rating || "Average"}</h2>
          <p className="hero-result__summary">{analysis.summary || "No summary returned."}</p>
        </div>
        <div className="hero-result__badges">
          <StatusPill label={analysis.hire_recommendation || "Borderline"} tone={ratingTone} />
          <StatusPill label={analysis.model_used || "openrouter/free"} tone="neutral" />
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard label="Communication" value={analysis.communication_score ?? "N/A"} tone="good" />
        <MetricCard label="Content" value={analysis.content_score ?? "N/A"} tone="neutral" />
        <MetricCard label="Confidence" value={analysis.confidence_score ?? "N/A"} tone="warn" />
        <MetricCard label="Words Per Minute" value={formatScore(result.wpm)} tone="neutral" />
        <MetricCard label="Filler Words" value={String(result.filler_count ?? 0)} tone="danger" />
        <MetricCard label="Word Count" value={String(result.word_count ?? countWords(result.transcription))} tone="neutral" />
      </div>

      <div className="content-grid">
        <SectionCard title="Strengths" eyebrow="What worked">
          <ul className="bullet-list bullet-list--success">
            {(analysis.strengths || []).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </SectionCard>

        <SectionCard title="Concerns" eyebrow="What to watch">
          <ul className="bullet-list bullet-list--warning">
            {(analysis.concerns || []).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </SectionCard>

        <SectionCard title="Suggestions" eyebrow="How to improve">
          <ul className="bullet-list bullet-list--info">
            {(analysis.suggestions || []).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </SectionCard>
      </div>

      <div className="content-grid content-grid--two">
        <SectionCard title="Transcript" eyebrow="Source text">
          <div className="transcript-box">{result.transcription || "No transcript available."}</div>
        </SectionCard>

        <SectionCard title="Run Details" eyebrow="Model and usage">
          <div className="detail-list">
            <div className="detail-row">
              <span>Whisper model</span>
              <strong>{result.whisper_model || "text-only input"}</strong>
            </div>
            <div className="detail-row">
              <span>Requested model</span>
              <strong>{analysis.model_requested || "openrouter/free"}</strong>
            </div>
            <div className="detail-row">
              <span>Actual model</span>
              <strong>{analysis.model_used || "N/A"}</strong>
            </div>
            <div className="detail-row">
              <span>Duration</span>
              <strong>{result.duration_seconds ? `${formatScore(result.duration_seconds)}s` : "N/A"}</strong>
            </div>
            <div className="detail-row">
              <span>Total tokens</span>
              <strong>{analysis.usage?.total_tokens ?? "N/A"}</strong>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [selectedFile, setSelectedFile] = useState(null);
  const [whisperModel, setWhisperModel] = useState("tiny");
  const [uploadRouterModel, setUploadRouterModel] = useState(DEFAULT_ANALYSIS_MODEL);
  const [textRouterModel, setTextRouterModel] = useState(DEFAULT_ANALYSIS_MODEL);
  const [uploadResult, setUploadResult] = useState(null);
  const [textResult, setTextResult] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [textLoading, setTextLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [textError, setTextError] = useState("");
  const [config, setConfig] = useState(null);
  const [configError, setConfigError] = useState("");
  const [transcription, setTranscription] = useState("");
  const [wpm, setWpm] = useState("130");
  const [fillerCount, setFillerCount] = useState("0");

  useEffect(() => {
    let cancelled = false;

    async function loadConfig() {
      try {
        const payload = await requestJson(`${API_BASE_URL}/api/config`);
        if (!cancelled) {
          setConfig(payload);
          const configuredModel = payload?.runtime?.default_router_model;
          if (configuredModel) {
            setUploadRouterModel(configuredModel);
            setTextRouterModel(configuredModel);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setConfigError(requestErrorMessage(error));
        }
      }
    }

    loadConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  const runtime = config?.runtime || {};
  const uploadButtonLabel = uploadLoading ? "Transcribing and analyzing..." : "Run Full Analysis";
  const textButtonLabel = textLoading ? "Analyzing transcript..." : "Analyze Transcript";

  const heroCopy = useMemo(
    () => ({
      title: "Whisper + OpenRouter Interview Review",
      subtitle:
        "Upload interview media or paste a transcript. The app transcribes with Whisper, evaluates the answer with a free OpenRouter model, and returns structured interview feedback you can showcase.",
    }),
    [],
  );

  async function handleUploadSubmit(event) {
    event.preventDefault();
    if (!selectedFile) {
      setUploadError("Choose an audio or video file before analyzing.");
      return;
    }

    setUploadLoading(true);
    setUploadError("");
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("whisper_model", whisperModel);
      formData.append("router_model", uploadRouterModel);

      const payload = await requestJson(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      setUploadResult(payload.data);
    } catch (error) {
      setUploadError(requestErrorMessage(error));
    } finally {
      setUploadLoading(false);
    }
  }

  async function handleTextSubmit(event) {
    event.preventDefault();
    if (countWords(transcription) < 3) {
      setTextError("Please enter at least 3 words of transcript text.");
      return;
    }

    const parsedWpm = Number(wpm);
    const parsedFillers = Number(fillerCount);
    if (Number.isNaN(parsedWpm) || parsedWpm < 0 || Number.isNaN(parsedFillers) || parsedFillers < 0) {
      setTextError("Please provide valid non-negative values for WPM and filler count.");
      return;
    }

    setTextLoading(true);
    setTextError("");
    setTextResult(null);

    try {
      const payload = await requestJson(`${API_BASE_URL}/api/analyze-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcription,
          wpm: parsedWpm,
          filler_count: parsedFillers,
          router_model: textRouterModel,
        }),
      });
      setTextResult(payload.data);
    } catch (error) {
      setTextError(requestErrorMessage(error));
    } finally {
      setTextLoading(false);
    }
  }

  function onFileChange(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setUploadError("");
  }

  function renderUploadTab() {
    return (
      <div className="tab-panel">
        <div className="split-layout">
          <SectionCard title="Upload Interview Media" eyebrow="Audio or video">
            <form className="form-stack" onSubmit={handleUploadSubmit}>
              <label className="upload-zone" htmlFor="mediaUpload">
                <input
                  id="mediaUpload"
                  type="file"
                  accept="audio/*,video/*"
                  onChange={onFileChange}
                  hidden
                />
                <div className="upload-zone__icon">+</div>
                <h4>Drop a recording here or click to browse</h4>
                <p>Supports {SUPPORTED_EXTENSIONS.join(", ")} and other FFmpeg-friendly formats.</p>
                <span className="upload-zone__file">
                  {selectedFile ? `Selected: ${selectedFile.name}` : "No media selected yet"}
                </span>
              </label>

              <div className="field-grid">
                <label className="field">
                  <span>Whisper model</span>
                  <select value={whisperModel} onChange={(event) => setWhisperModel(event.target.value)}>
                    <option value="tiny">Tiny</option>
                    <option value="base">Base</option>
                  </select>
                </label>

                <label className="field">
                  <span>OpenRouter model</span>
                  <input
                    value={uploadRouterModel}
                    onChange={(event) => setUploadRouterModel(event.target.value)}
                    placeholder={DEFAULT_ANALYSIS_MODEL}
                  />
                </label>
              </div>

              {uploadError ? <div className="message message--error">{uploadError}</div> : null}
              <button className="primary-button" type="submit" disabled={uploadLoading}>
                {uploadButtonLabel}
              </button>
            </form>
          </SectionCard>

          <SectionCard title="How It Works" eyebrow="Pipeline">
            <div className="timeline">
              <div className="timeline__item">
                <strong>1. Upload media</strong>
                <p>Drag in an interview recording or choose a file from disk.</p>
              </div>
              <div className="timeline__item">
                <strong>2. Transcribe with Whisper</strong>
                <p>The backend converts the media, runs Whisper locally, and calculates WPM plus filler count.</p>
              </div>
              <div className="timeline__item">
                <strong>3. Evaluate with OpenRouter</strong>
                <p>The transcript is sent to a free available model for structured interview feedback.</p>
              </div>
            </div>
          </SectionCard>
        </div>

        {uploadResult ? <AnalysisResults result={uploadResult} /> : null}
      </div>
    );
  }

  function renderTextTab() {
    return (
      <div className="tab-panel">
        <SectionCard title="Analyze Transcript Text" eyebrow="Text-based input">
          <form className="form-stack" onSubmit={handleTextSubmit}>
            <label className="field">
              <span>Transcript</span>
              <textarea
                rows={9}
                value={transcription}
                onChange={(event) => setTranscription(event.target.value)}
                placeholder="Paste an interview answer here..."
              />
            </label>

            <div className="field-grid field-grid--triple">
              <label className="field">
                <span>Words per minute</span>
                <input type="number" min="0" value={wpm} onChange={(event) => setWpm(event.target.value)} />
              </label>
              <label className="field">
                <span>Filler count</span>
                <input type="number" min="0" value={fillerCount} onChange={(event) => setFillerCount(event.target.value)} />
              </label>
              <label className="field">
                <span>OpenRouter model</span>
                <input
                  value={textRouterModel}
                  onChange={(event) => setTextRouterModel(event.target.value)}
                  placeholder={DEFAULT_ANALYSIS_MODEL}
                />
              </label>
            </div>

            <div className="inline-note">
              <span>Word count</span>
              <strong>{countWords(transcription)}</strong>
            </div>

            {textError ? <div className="message message--error">{textError}</div> : null}
            <button className="primary-button" type="submit" disabled={textLoading}>
              {textButtonLabel}
            </button>
          </form>
        </SectionCard>

        {textResult ? <AnalysisResults result={textResult} /> : null}
      </div>
    );
  }

  function renderInfoTab() {
    return (
      <div className="tab-panel">
        <div className="metric-grid">
          <MetricCard
            label="API Key"
            value={runtime.openrouter_api_key_configured ? "Configured" : "Missing"}
            tone={runtime.openrouter_api_key_configured ? "good" : "danger"}
          />
          <MetricCard
            label="FFmpeg"
            value={runtime.ffmpeg_available ? "Available" : "Missing"}
            tone={runtime.ffmpeg_available ? "good" : "danger"}
          />
          <MetricCard label="Whisper Default" value={runtime.default_whisper_model || "tiny"} tone="neutral" />
          <MetricCard label="Router Default" value={runtime.default_router_model || DEFAULT_ANALYSIS_MODEL} tone="neutral" />
        </div>

        <div className="content-grid content-grid--two">
          <SectionCard title="API Endpoints" eyebrow="Backend surface">
            <div className="endpoint-list">
              <div className="endpoint-card">
                <code>POST /api/analyze</code>
                <p>Upload media, transcribe it, and get a full OpenRouter evaluation.</p>
              </div>
              <div className="endpoint-card">
                <code>POST /api/analyze-text</code>
                <p>Send transcript text directly for evaluation.</p>
              </div>
              <div className="endpoint-card">
                <code>GET /api/config</code>
                <p>Inspect runtime defaults and supported upload types.</p>
              </div>
              <div className="endpoint-card">
                <code>GET /health</code>
                <p>Quick health check for API key and FFmpeg availability.</p>
              </div>
            </div>
          </SectionCard>

          <SectionCard title="Runtime Status" eyebrow="Current environment">
            {configError ? (
              <div className="message message--error">{configError}</div>
            ) : (
              <div className="detail-list">
                <div className="detail-row">
                  <span>API Base URL</span>
                  <strong>{API_BASE_URL}</strong>
                </div>
                <div className="detail-row">
                  <span>Supported uploads</span>
                  <strong>{(config?.upload_types || []).join(", ") || "Loading..."}</strong>
                </div>
                <div className="detail-row">
                  <span>Frontend mode</span>
                  <strong>React + Vite</strong>
                </div>
              </div>
            )}
          </SectionCard>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="orb orb--left" />
      <div className="orb orb--right" />

      <header className="hero">
        <div className="hero__content">
          <p className="eyebrow">Interview Intelligence</p>
          <h1>{heroCopy.title}</h1>
          <p className="hero__subtitle">{heroCopy.subtitle}</p>
        </div>
        <div className="hero__stats">
          <MetricCard label="Input Modes" value="Media + Text" tone="good" />
          <MetricCard label="Speech Engine" value={runtime.default_whisper_model || "Whisper"} tone="neutral" />
          <MetricCard label="Analysis Model" value={runtime.default_router_model || DEFAULT_ANALYSIS_MODEL} tone="warn" />
        </div>
      </header>

      <main className="dashboard">
        {config && runtime.openrouter_api_key_configured === false ? (
          <div className="message message--error message--banner">
            OpenRouter is not configured yet...
          </div>
        ) : null}

        <div className="tab-strip" role="tablist" aria-label="Analyzer sections">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? "tab-button--active" : ""}`}
              type="button"
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === "upload" ? renderUploadTab() : null}
        {activeTab === "text" ? renderTextTab() : null}
        {activeTab === "info" ? renderInfoTab() : null}
      </main>
    </div>
  );
}

export default App;
