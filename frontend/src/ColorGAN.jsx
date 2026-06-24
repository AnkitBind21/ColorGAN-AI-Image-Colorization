import { useState, useRef, useCallback } from "react";

const API_BASE = "https://colorgan-ai-image-colorization.onrender.com";

// ── Styles ────────────────────────────────────────────────────
const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:      #0d0d0d;
    --surface: #161616;
    --border:  #2a2a2a;
    --accent:  #f4a300;
    --accent2: #e05c00;
    --text:    #e8e8e8;
    --muted:   #666;
    --success: #2ecc71;
    --error:   #e74c3c;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    min-height: 100vh;
  }

  @keyframes slide {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
  }
  @keyframes spin { to { transform: rotate(360deg); } }
`;

// ── Sub-components ────────────────────────────────────────────

function Spinner() {
  return (
    <span style={{
      width: 16, height: 16,
      border: "2px solid rgba(0,0,0,0.2)",
      borderTopColor: "#000",
      borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
      display: "inline-block",
    }} />
  );
}

function StatusBar({ message, type }) {
  if (!message) return null;
  const colors = {
    info:    { bg: "#1a1a2e", color: "#6699ff",  border: "#6699ff" },
    success: { bg: "#0d2016", color: "#2ecc71",  border: "#2ecc71" },
    error:   { bg: "#2a0d0d", color: "#e74c3c",  border: "#e74c3c" },
    warning: { bg: "#1f1800", color: "#f4a300",  border: "#f4a300" },
  };
  const c = colors[type] || colors.info;
  return (
    <div style={{
      marginTop: 16, padding: "10px 14px", borderRadius: 2,
      fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.78rem",
      display: "flex", alignItems: "center", gap: 10,
      background: c.bg, color: c.color, borderLeft: `3px solid ${c.border}`,
    }}>
      {message}
    </div>
  );
}

function ProgressBar({ visible }) {
  if (!visible) return null;
  return (
    <div style={{
      marginTop: 12, height: 3, background: "var(--border)",
      borderRadius: 2, overflow: "hidden",
    }}>
      <div style={{
        height: "100%", width: "40%",
        background: "linear-gradient(90deg, var(--accent), var(--accent2))",
        borderRadius: 2,
        animation: "slide 1.2s ease-in-out infinite",
      }} />
    </div>
  );
}

function ImagePanel({ title, badge, badgeActive, children }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 4, overflow: "hidden",
    }}>
      <div style={{
        padding: "10px 16px", borderBottom: "1px solid var(--border)",
        fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.75rem",
        color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        {title}
        <span style={{
          background: badgeActive ? "var(--accent)" : "var(--border)",
          color: badgeActive ? "#000" : "var(--muted)",
          padding: "2px 8px", borderRadius: 2, fontSize: "0.7rem",
        }}>
          {badge}
        </span>
      </div>
      {children}
    </div>
  );
}

function MetricCard({ label, children }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 4, padding: "16px 24px", minWidth: 140,
    }}>
      <div style={{
        fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.7rem",
        color: "var(--muted)", textTransform: "uppercase",
        letterSpacing: 1, marginBottom: 6,
      }}>
        {label}
      </div>
      {children}
    </div>
  );
}

const PIPELINE_STEPS = [
  { label: "RGB Input",               highlight: false },
  { label: "→ LAB Space",            highlight: false },
  { label: "L Channel (1×256×256)",  highlight: true  },
  { label: "U-Net Generator",        highlight: true  },
  { label: "AB Channels (2×256×256)",highlight: true  },
  { label: "L+AB → RGB",             highlight: false },
  { label: "Colorized Output",       highlight: false },
];

// ── Main Component ────────────────────────────────────────────

export default function ColorGAN() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [inputSrc,     setInputSrc]     = useState(null);
  const [inputBadge,   setInputBadge]   = useState("grayscale");
  const [resultUrl,    setResultUrl]    = useState(null);
  const [status,       setStatus]       = useState({ message: "", type: "info" });
  const [loading,      setLoading]      = useState(false);
  const [metrics,      setMetrics]      = useState(null);  // { time }
  const [dragOver,     setDragOver]     = useState(false);

  const fileInputRef = useRef(null);
  const startTimeRef = useRef(null);

  // ── File handling ──────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return;

    const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
    if (!allowed.includes(file.type)) {
      setStatus({ message: "Unsupported format. Use JPG, PNG, or BMP.", type: "error" });
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setStatus({ message: "File too large (max 10MB).", type: "error" });
      return;
    }

    setSelectedFile(file);
    setResultUrl(null);
    setMetrics(null);
    setInputBadge(file.type.split("/")[1].toUpperCase());
    setStatus({
      message: `Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB) — click Run Colorization`,
      type: "info",
    });

    const reader = new FileReader();
    reader.onload = (e) => setInputSrc(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const onFileInputChange = (e) => handleFile(e.target.files[0]);

  const onDragOver  = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = ()  => setDragOver(false);
  const onDrop      = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // ── Colorize ───────────────────────────────────────────────
  const runColorize = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setStatus({ message: "Sending to colorization server…", type: "info" });
    startTimeRef.current = Date.now();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res  = await fetch(`${API_BASE}/colorize`, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || !data.success) throw new Error(data.error || `Server error ${res.status}`);

      const elapsed = ((Date.now() - startTimeRef.current) / 1000).toFixed(2);
      const url = `${API_BASE}${data.url}?t=${Date.now()}`;

      setResultUrl(url);
      setMetrics({ time: elapsed });
      setStatus({ message: `✓ Colorization complete in ${elapsed}s`, type: "success" });
    } catch (err) {
      const msg = err.message.includes("Failed to fetch")
        ? "Cannot connect to backend. Please try again in a few moments."
        : err.message;
      setStatus({ message: `Error: ${msg}`, type: "error" });
    } finally {
      setLoading(false);
    }
  };

  // ── Download ───────────────────────────────────────────────
const downloadResult = async () => {
  if (!resultUrl) return;

  try {
    const response = await fetch(resultUrl);
    const blob = await response.blob();

    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "colorized_result.jpg";

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Download failed:", err);
  }
};

  // ── Reset ──────────────────────────────────────────────────
  const resetAll = () => {
    setSelectedFile(null);
    setInputSrc(null);
    setResultUrl(null);
    setStatus({ message: "", type: "info" });
    setMetrics(null);
    setInputBadge("grayscale");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const hasFile = !!selectedFile;

  // ── Render ─────────────────────────────────────────────────
  return (
    <>
      <style>{globalStyles}</style>

      {/* Header */}
      <header style={{
        borderBottom: "1px solid var(--border)", padding: "20px 40px",
        display: "flex", alignItems: "baseline", gap: 16,
      }}>
        <div style={{
          fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600,
          fontSize: "1.4rem", color: "var(--accent)", letterSpacing: "-0.5px",
        }}>
          color<span style={{ color: "var(--muted)" }}>GAN</span>
        </div>
        <div style={{
          fontSize: "0.8rem", color: "var(--muted)",
          fontFamily: "'IBM Plex Mono', monospace",
        }}>
          conditional GAN · LAB colorspace · pix2pix
        </div>
      </header>

      {/* Main */}
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 40px" }}>

        {/* Upload Zone */}
        {!hasFile && (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            style={{
              border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`,
              borderRadius: 4, padding: "60px 40px", textAlign: "center",
              cursor: "pointer", transition: "border-color 0.2s, background 0.2s",
              background: dragOver ? "#1c1a14" : "var(--surface)",
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.bmp"
              style={{ display: "none" }}
              onChange={onFileInputChange}
            />
            <div style={{ fontSize: "3rem", marginBottom: 16, opacity: 0.4 }}>◫</div>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 400, marginBottom: 8, color: "var(--text)" }}>
              Drop a grayscale or color image here
            </h2>
            <p style={{
              fontSize: "0.85rem", color: "var(--muted)",
              fontFamily: "'IBM Plex Mono', monospace",
            }}>
              JPG / PNG / BMP · max 10MB · will be resized to 256×256
            </p>
          </div>
        )}

        {/* Status + Progress */}
        <StatusBar message={status.message} type={status.type} />
        <ProgressBar visible={loading} />

        {/* Images Grid */}
        {hasFile && (
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr",
            gap: 24, marginTop: 40,
          }}>
            {/* Input panel */}
            <ImagePanel title="Input" badge={inputBadge} badgeActive={false}>
              <img
                src={inputSrc}
                alt="Input"
                style={{ width: "100%", display: "block", aspectRatio: 1, objectFit: "contain", background: "#000" }}
              />
            </ImagePanel>

            {/* Output panel */}
            <ImagePanel title="Colorized Output" badge={resultUrl ? "COLORIZED" : "—"} badgeActive={!!resultUrl}>
              {resultUrl ? (
                <img
                  src={resultUrl}
                  alt="Colorized output"
                  style={{ width: "100%", display: "block", aspectRatio: 1, objectFit: "contain", background: "#000" }}
                />
              ) : (
                <div style={{
                  aspectRatio: 1, display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", gap: 12,
                  color: "var(--muted)", fontFamily: "'IBM Plex Mono', monospace",
                  fontSize: "0.8rem", background: "#000",
                }}>
                  <span style={{ fontSize: "2rem", opacity: 0.2 }}>◩</span>
                  run colorization to see result
                </div>
              )}
            </ImagePanel>
          </div>
        )}

        {/* Controls */}
        {hasFile && (
          <div style={{ marginTop: 24, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <button
              onClick={runColorize}
              disabled={loading}
              style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.85rem",
                padding: "12px 24px", border: "none", borderRadius: 2,
                cursor: loading ? "not-allowed" : "pointer",
                fontWeight: 600, letterSpacing: "0.5px",
                background: "var(--accent)", color: "#000",
                opacity: loading ? 0.35 : 1,
                display: "flex", alignItems: "center", gap: 8,
              }}
            >
              {loading ? <><Spinner /> Processing…</> : "Run Colorization"}
            </button>

            <button
              onClick={resetAll}
              style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.85rem",
                padding: "12px 24px", border: "1px solid var(--border)", borderRadius: 2,
                cursor: "pointer", fontWeight: 600, letterSpacing: "0.5px",
                background: "transparent", color: "var(--text)",
              }}
            >
              New Image
            </button>

            {resultUrl && (
              <button
                onClick={downloadResult}
                style={{
                  fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.85rem",
                  padding: "12px 24px", border: "none", borderRadius: 2,
                  cursor: "pointer", fontWeight: 600, letterSpacing: "0.5px",
                  background: "var(--success)", color: "#000",
                }}
              >
                Download Result
              </button>
            )}
          </div>
        )}

        {/* Metrics */}
        {metrics && (
          <div style={{ marginTop: 24, display: "flex", gap: 16, flexWrap: "wrap" }}>
            <MetricCard label="Processing Time">
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: "1.6rem",
                color: "var(--accent)", fontWeight: 600,
              }}>
                {metrics.time}s
              </div>
            </MetricCard>
            <MetricCard label="Output Size">
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: "1.6rem",
                color: "var(--accent)", fontWeight: 600,
              }}>
                256<span style={{ fontSize: "0.75rem", color: "var(--muted)", marginLeft: 4 }}>px</span>
              </div>
            </MetricCard>
            <MetricCard label="Model">
              <div style={{
                fontFamily: "'IBM Plex Mono', monospace", fontSize: "1rem",
                color: "var(--accent)", fontWeight: 600, paddingTop: 6,
              }}>
                U-Net<span style={{ fontSize: "0.75rem", color: "var(--muted)", marginLeft: 4 }}>+ PatchGAN</span>
              </div>
            </MetricCard>
          </div>
        )}

        {/* Pipeline */}
        <div style={{ marginTop: 64, paddingTop: 32, borderTop: "1px solid var(--border)" }}>
          <h3 style={{
            fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.75rem",
            color: "var(--muted)", textTransform: "uppercase",
            letterSpacing: 2, marginBottom: 20,
          }}>
            Pipeline
          </h3>
          <div style={{ display: "flex", overflowX: "auto", paddingBottom: 8 }}>
            {PIPELINE_STEPS.map((step, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
                <div style={{
                  background: "var(--surface)",
                  border: `1px solid ${step.highlight ? "var(--accent)" : "var(--border)"}`,
                  padding: "10px 16px",
                  fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.72rem",
                  color: step.highlight ? "var(--accent)" : "var(--text)",
                  whiteSpace: "nowrap",
                }}>
                  {step.label}
                </div>
                {i < PIPELINE_STEPS.length - 1 && (
                  <span style={{
                    color: "var(--muted)", fontFamily: "'IBM Plex Mono', monospace",
                    padding: "0 6px", fontSize: "0.9rem",
                  }}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: "1px solid var(--border)", padding: "20px 40px",
        fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.72rem",
        color: "var(--muted)", marginTop: 80,
      }}>
        colorGAN · cGAN-based image colorization · U-Net generator + 70×70 PatchGAN discriminator
      </footer>
    </>
  );
}
