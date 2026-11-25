# Lyftr AI — LLM Chatbot with Key Failover and Response Caching

Full-stack multi-turn LLM chatbot built with **React**, **FastAPI**, **LiteLLM**, and **SQLite**, featuring automatic API key failover, response caching, per-request token tracking, and Dockerized deployment.

## 1. Live URLs

- **Frontend (Live App):** `https://<your-vercel-app>.vercel.app`
- **Backend (API Base URL):** `https://<your-backend-service>.onrender.com`
- **Health Check:** `https://<your-backend-service>.onrender.com/health`
- **Demo Video:** `<your-loom-or-youtube-unlisted-link>`

*(Replace the placeholders above after deploying to Render/Railway and Vercel.)*

## 2. Run locally: with Docker and without

### With Docker (Recommended)

1. Copy the environment template and add at least 2 API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and populate `LLM_API_KEYS=keyA,keyB` (for example, two Google AI Studio Gemini keys with `LLM_MODEL=gemini/gemini-2.0-flash`, or two Groq keys with `LLM_MODEL=groq/llama-3.1-8b-instant`).

2. Start both services:
   ```bash
   make up
   ```
   - Backend API: `http://localhost:8000`
   - Frontend UI: `http://localhost:3000`

3. Stop the stack:
   ```bash
   make down
   ```

### Without Docker

#### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

export LLM_API_KEYS="your_key_1,your_key_2"
export LLM_MODEL="gemini/gemini-2.0-flash"
export DATABASE_URL="sqlite:///./data/app.db"
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173"
export ENABLE_FAILURE_SIMULATION="true"

uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

#### Run Backend Tests
```bash
cd backend
pytest -v
```

## 3. Design decisions: key fallback, caching, SQLite session storage

### Key Fallback (`backend/app/llm_gateway.py`)
- Keys from `LLM_API_KEYS` are indexed deterministically as `key_1`, `key_2`, ... in order. Raw API keys are never returned in API responses or written to logs.
- Each request walks the key pool sequentially with `num_retries=0` inside LiteLLM so failover across keys happens immediately without redundant retries against a failing key.
- On HTTP `429` (rate limit / quota exhaustion), `5xx` server errors, timeouts, connection errors, or simulated failures (`X-Simulate-Failure`), `fallback_attempts` increments and the request automatically fails over to the next key.
- If every key fails, the backend returns HTTP `503` with `{"detail": "all providers unavailable"}` and does **not** persist the failed turn to SQLite.
- When `ENABLE_FAILURE_SIMULATION=true` and `X-Simulate-Failure` is present, cache reads are bypassed for that request so failover behavior can be demonstrated deterministically even on previously cached prompts.

### Response Caching (LiteLLM Built-In Cache)
- Uses LiteLLM's native `Cache(type="local")` (configurable to `disk` via `CACHE_TYPE`).
- LiteLLM hashes the model and full message list (`system prompt + conversation history + new user message`) while excluding `api_key`. As a result, identical first messages across different sessions hit the cache (`resolution_source="LLM_CACHE"`, `key_id=null`, `fallback_attempts=0`), while different messages or divergent multi-turn histories always miss cache and run `LIVE_INFERENCE`.
- Cache hits return the exact stored completion text and original per-request token counts in single-digit milliseconds.

### SQLite Session Storage (`backend/app/database.py`)
- Pure SQLite (`sqlite3`) with WAL mode (`PRAGMA journal_mode=WAL`) enabled for safe concurrent reads/writes and automatic schema creation on startup.
- Stores conversation history in the `messages` table (indexed on `(session_id, id)`) and optional per-session system prompts in a `sessions` table.
- Both user and assistant turns are written inside a single atomic transaction only after a valid LLM or cache response is produced, guaranteeing failed (`422`/`503`) requests never pollute session history.
- In Docker, `DATABASE_URL=sqlite:////data/app.db` writes to a named volume (`sqlite_data:/data`), preserving conversation history across container restarts.

## 4. Verification: how to see failover, cache hits and per-message token usage

Set your base URL (`http://localhost:8000` locally or your deployed Render URL):
```bash
BASE_URL=http://localhost:8000
```

### 1. Health check
```bash
curl -s $BASE_URL/health | jq
```

### 2. Live inference vs. cache hit across sessions
```bash
# First call -> LIVE_INFERENCE, key_id="key_1"
curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"verify-1","message":"What is the capital of France?"}' | jq

# Same first message in a NEW session -> LLM_CACHE, key_id=null, much lower latency_ms
curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"verify-2","message":"What is the capital of France?"}' | jq
```

### 3. Multi-turn memory & SQLite persistence
```bash
curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"verify-multi","message":"My favorite city is Kyoto."}' | jq

curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"verify-multi","message":"What is my favorite city?"}' | jq

# Inspect stored history (oldest first, no system prompt)
curl -s $BASE_URL/sessions/verify-multi/messages | jq
```

### 4. Key failover & all-keys-down 503 via `X-Simulate-Failure`
```bash
# Force key_1 to 429 -> responds 200 with key_id="key_2", fallback_attempts=1
curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"verify-failover","message":"Say hello in Spanish."}' | jq

# Force key_1 to 500 -> responds 200 with key_id="key_2", fallback_attempts=1
curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=500" \
  -d '{"session_id":"verify-failover-500","message":"Say hello in German."}' | jq

# Force all keys to fail -> responds 503 {"detail":"all providers unavailable"}
curl -i -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"verify-503","message":"Should fail cleanly"}'

# Confirm nothing was added to SQLite history (404)
curl -i $BASE_URL/sessions/verify-503/messages
```

### 5. Input validation (422)
```bash
curl -i -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"verify-422","message":"   "}'
```

## 5. Setup Used: your editor and AI tools

- **Editor:** VS Code
- **AI Tools:** Gemini (architecture design, backend/frontend scaffolding, test generation, and Docker configuration)

## 6. Bonus features (if any): what you added beyond the requirements and why

- **System Prompt Override (`system-prompt-input`):** Added the optional per-session system prompt input in the UI and persisted overrides in a `sessions` table so multi-turn conversations maintain their custom persona across turns.
- **Interactive Failover Simulator Dropdown in UI:** Added a non-intrusive selector in the top toolbar to trigger `X-Simulate-Failure` (`key_1=429`, `key_1=500`, `*=429`) directly from the browser for effortless live demos and QA testing.
- **Live Backend Health Pill & New Session Button:** Displays live key-pool count and cache state from `/health`, plus a one-click "New Session" button to test cross-session `LLM_CACHE` hits without reloading the page.
- **Per-Key Multi-Model Support (`LLM_MODELS`):** Supports both a single model across multiple keys (`LLM_MODEL`) and heterogeneous model failover across providers (`LLM_MODELS`).
