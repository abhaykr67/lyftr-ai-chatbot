import React, { useEffect, useRef, useState } from "react";
import { fetchHealth, getApiBaseUrl, sendChatMessage } from "./api";

function generateSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

export default function App() {
  // Always start a brand new session on every page load
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulateFailure, setSimulateFailure] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const bottomRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    fetchHealth()
      .then((data) => {
        if (mounted) {
          setHealth(data);
          setHealthError(null);
        }
      })
      .catch((err) => {
        if (mounted) {
          setHealthError(err.message);
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleNewSession = () => {
    setSessionId(generateSessionId());
    setMessages([]);
    setInput("");
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMsgId = `u-${Date.now()}-${Math.random()}`;
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: trimmed },
    ]);
    // Clear input immediately after sending (per frontend contract)
    setInput("");
    setIsLoading(true);

    try {
      const data = await sendChatMessage({
        sessionId,
        message: trimmed,
        systemPrompt,
        simulateFailure,
      });

      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}-${Math.random()}`,
          role: "assistant",
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch (err) {
      const statusPrefix = err.status ? `Error ${err.status}: ` : "Error: ";
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}-${Math.random()}`,
          role: "error",
          content: `${statusPrefix}${err.message || "Unable to reach backend"}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <h1>Lyftr AI Chatbot</h1>
            <p className="subtitle">
              LiteLLM Key Failover &bull; Response Caching &bull; SQLite State
            </p>
          </div>
        </div>

        <div className="header-controls">
          <div className="status-pill" title={`Backend: ${getApiBaseUrl()}`}>
            <span className={`indicator ${health ? "online" : "offline"}`} />
            {health ? (
              <span>
                Online &middot; {health.keys_configured} keys &middot; Cache{" "}
                {health.cache_enabled ? "ON" : "OFF"}
              </span>
            ) : (
              <span>{healthError ? "Backend offline" : "Connecting..."}</span>
            )}
          </div>

          <div className="session-badge">
            <span>Session:</span>
            <code>{sessionId}</code>
          </div>

          <button
            type="button"
            className="secondary-btn"
            onClick={handleNewSession}
            disabled={isLoading}
          >
            New Session
          </button>
        </div>
      </header>

      <section className="config-strip">
        <div className="config-group system-prompt-group">
          <label htmlFor="system-prompt">System Prompt (Optional)</label>
          <input
            id="system-prompt"
            data-testid="system-prompt-input"
            type="text"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="e.g. You are a concise senior engineer..."
            maxLength={4000}
          />
        </div>

        <div className="config-group simulator-group">
          <label htmlFor="failure-sim">Failover Hook (X-Simulate-Failure)</label>
          <select
            id="failure-sim"
            value={simulateFailure}
            onChange={(e) => setSimulateFailure(e.target.value)}
          >
            <option value="">Normal (Cache + Key Pool)</option>
            <option value="key_1=429">Simulate key_1=429 (Failover → key_2)</option>
            <option value="key_1=500">Simulate key_1=500 (Failover → key_2)</option>
            <option value="*=429">Simulate *=429 (All keys fail → 503)</option>
          </select>
        </div>
      </section>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>
              Ask any question to inspect live inference latency, per-request token usage,
              automatic key failover, and cross-session cache hits.
            </p>
            <div className="quick-prompts">
              {[
                "What is the capital of France?",
                "Explain HTTP 429 rate limiting in two sentences.",
                "Remember my favorite language is Python.",
              ].map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="chip"
                  onClick={() => setInput(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => {
          if (msg.role === "user") {
            return (
              <div key={msg.id} className="message-row user-row">
                <div className="bubble user-bubble">
                  <div className="role-label">You</div>
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            );
          }

          if (msg.role === "error") {
            return (
              <div key={msg.id} className="message-row error-row" role="alert">
                <div className="bubble error-bubble">
                  <div className="role-label">Request Error</div>
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            );
          }

          const meta = msg.metadata || {};
          const isCache = meta.resolution_source === "LLM_CACHE";

          return (
            <div key={msg.id} className="message-row assistant-row">
              <div
                className="bubble assistant-bubble"
                data-testid="assistant-message"
              >
                <div className="role-label">Assistant</div>
                <div className="message-text">{msg.content}</div>

                <div className="metadata-bar">
                  <span
                    className={`meta-badge source-badge ${
                      isCache ? "source-cache" : "source-live"
                    }`}
                    data-testid="meta-source"
                  >
                    {meta.resolution_source}
                  </span>

                  <span className="meta-item">
                    <span className="meta-key">Model:</span>{" "}
                    <span data-testid="meta-model">{meta.model}</span>
                  </span>

                  <span className="meta-item">
                    <span className="meta-key">Latency:</span>{" "}
                    <span data-testid="meta-latency">{meta.latency_ms} ms</span>
                  </span>

                  <span className="meta-item">
                    <span className="meta-key">Tokens:</span>{" "}
                    <span data-testid="meta-input-tokens">{meta.input_tokens}</span> in /{" "}
                    <span data-testid="meta-output-tokens">{meta.output_tokens}</span> out
                  </span>

                  {meta.key_id && (
                    <span className="meta-item">
                      <span className="meta-key">Key:</span> {meta.key_id}
                    </span>
                  )}

                  {typeof meta.fallback_attempts === "number" &&
                    meta.fallback_attempts > 0 && (
                      <span className="meta-badge failover-badge">
                        Failovers: {meta.fallback_attempts}
                      </span>
                    )}
                </div>
              </div>
            </div>
          );
        })}

        {isLoading && (
          <div className="message-row assistant-row">
            <div className="bubble assistant-bubble loading-bubble">
              <span className="typing-dots">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
          rows={2}
          disabled={isLoading}
        />
        <button
          type="button"
          data-testid="send-button"
          className="send-btn"
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? "Sending..." : "Send"}
        </button>
      </footer>
    </div>
  );
}
