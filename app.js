import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000"; // Change if deployed

function App() {
  const [textQuery, setTextQuery] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [evidence, setEvidence] = useState([]);
  const [mode, setMode] = useState("text"); // "text" or "audio"

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAnswer("");
    setEvidence([]);
    try {
      const res = await axios.post(
        `${API_URL}/query`,
        new URLSearchParams({ query: textQuery }),
        { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
      );
      setAnswer(res.data.answer || "");
      setEvidence(res.data.evidence || []);
    } catch (err) {
      setAnswer("Error: " + (err.response?.data?.error || err.message));
    }
    setLoading(false);
  };

  const handleAudioSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAnswer("");
    setEvidence([]);
    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("top_k", 5);
      const res = await axios.post(`${API_URL}/query_audio`, formData);
      setAnswer(res.data.answer || "");
      setEvidence(res.data.evidence || []);
    } catch (err) {
      setAnswer("Error: " + (err.response?.data?.error || err.message));
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 700, margin: "auto", padding: 32 }}>
      <h2>Voxarch Multimodal RAG Demo</h2>
      <div style={{ marginBottom: 24 }}>
        <button onClick={() => setMode("text")} disabled={mode === "text"}>Text Query</button>
        <button onClick={() => setMode("audio")} disabled={mode === "audio"}>Audio Query</button>
      </div>

      {mode === "text" ? (
        <form onSubmit={handleTextSubmit} style={{ marginBottom: 32 }}>
          <textarea
            rows={3}
            style={{ width: "100%", fontSize: 16 }}
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            placeholder="Type your question here..."
            required
          />
          <button type="submit" disabled={loading || !textQuery}>
            {loading ? "Querying..." : "Ask"}
          </button>
        </form>
      ) : (
        <form onSubmit={handleAudioSubmit} style={{ marginBottom: 32 }}>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setAudioFile(e.target.files[0])}
            required
          />
          <button type="submit" disabled={loading || !audioFile}>
            {loading ? "Querying..." : "Ask (Audio)"}
          </button>
        </form>
      )}

      {answer && (
        <div style={{ marginTop: 32, padding: 20, border: "1px solid #ccc", borderRadius: 6 }}>
          <h3>Answer</h3>
          <p>{answer}</p>
        </div>
      )}

      {evidence.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h4>Evidence</h4>
          {evidence.map((e, i) => (
            <div key={i} style={{ marginBottom: 18, padding: 10, borderBottom: "1px solid #eee" }}>
              <strong>Source:</strong> {e.meta?.filename || "N/A"} <br />
              <strong>Section:</strong> {e.meta?.section || e.meta?.chunk_index || "N/A"} <br />
              <strong>Text:</strong> <span style={{ color: "#333" }}>{e.text?.slice(0, 400) || "[audio chunk]"}</span>
              {e.meta?.start_time !== undefined && (
                <div>
                  <strong>Timestamp:</strong> {e.meta.start_time}s - {e.meta.end_time || "end"}s
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
