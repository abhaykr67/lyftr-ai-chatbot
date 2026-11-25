const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export function getApiBaseUrl() {
  return API_BASE_URL;
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }
  return response.json();
}

export async function sendChatMessage({
  sessionId,
  message,
  systemPrompt,
  simulateFailure,
}) {
  const headers = {
    "Content-Type": "application/json",
  };
  if (simulateFailure && simulateFailure.trim()) {
    headers["X-Simulate-Failure"] = simulateFailure.trim();
  }

  const body = {
    session_id: sessionId,
    message,
  };
  if (systemPrompt && systemPrompt.trim()) {
    body.system_prompt = systemPrompt.trim();
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    let detailMessage = `Request failed with status ${response.status}`;
    if (data && data.detail) {
      if (typeof data.detail === "string") {
        detailMessage = data.detail;
      } else if (Array.isArray(data.detail)) {
        detailMessage = data.detail
          .map((item) => item.msg || JSON.stringify(item))
          .join("; ");
      }
    }
    const err = new Error(detailMessage);
    err.status = response.status;
    throw err;
  }

  return data;
}
