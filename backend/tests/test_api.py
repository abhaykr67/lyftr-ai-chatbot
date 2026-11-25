import time
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import KeyEntry, Settings
from app.database import Database
from app.llm_gateway import LLMGateway
from app.main import create_app


def _make_settings(enable_sim: bool = True, cache_enabled: bool = True) -> Settings:
    return Settings(
        key_pool=[
            KeyEntry(key_id="key_1", api_key="secret-test-key-1", model="gemini/gemini-2.0-flash"),
            KeyEntry(key_id="key_2", api_key="secret-test-key-2", model="gemini/gemini-2.0-flash"),
        ],
        default_model="gemini/gemini-2.0-flash",
        database_url="sqlite:///:memory:",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_sim,
        cache_enabled=cache_enabled,
        cache_type="local",
        disk_cache_dir="/tmp/test_cache",
        default_system_prompt="You are a helpful assistant.",
        request_timeout_s=10.0,
    )


def _mock_response(content: str, prompt_tokens: int = 25, completion_tokens: int = 8, cache_hit: bool = False):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        _hidden_params={"cache_hit": cache_hit},
    )


class _FakeLiteLLMCacheBehavior:
    """Simulates LiteLLM's built-in cache keyed on (model, messages) ignoring api_key."""

    def __init__(self, fail_keys=None, delay_s: float = 0.02):
        self.store = {}
        self.fail_keys = set(fail_keys or [])
        self.delay_s = delay_s
        self.calls = []

    def __call__(self, **kwargs):
        model = kwargs["model"]
        messages = kwargs["messages"]
        api_key = kwargs["api_key"]
        caching = kwargs.get("caching", False)
        cache_key = (model, tuple((m["role"], m["content"]) for m in messages))

        self.calls.append({"api_key": api_key, "messages": list(messages), "caching": caching})

        if caching and cache_key in self.store:
            reply, pt, ct = self.store[cache_key]
            return _mock_response(reply, pt, ct, cache_hit=True)

        if api_key in self.fail_keys:
            err = RuntimeError("429 Rate limit exceeded")
            err.status_code = 429
            raise err

        time.sleep(self.delay_s)
        last_user = messages[-1]["content"]
        reply = f"Answer to: {last_user} (turns={len(messages)})"
        prompt_tokens = sum(len(m["content"]) for m in messages)
        completion_tokens = len(reply)

        if caching:
            self.store[cache_key] = (reply, prompt_tokens, completion_tokens)

        return _mock_response(reply, prompt_tokens, completion_tokens, cache_hit=False)


@pytest.fixture
def client_and_fake():
    settings = _make_settings(enable_sim=True, cache_enabled=True)
    db = Database("sqlite:///:memory:")
    gateway = LLMGateway(settings)
    app = create_app(settings_override=settings, db_override=db, gateway_override=gateway)
    fake = _FakeLiteLLMCacheBehavior()
    with patch.object(litellm, "completion", side_effect=fake):
        with TestClient(app) as client:
            yield client, fake


def test_health_endpoint(client_and_fake):
    client, _ = client_and_fake
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_validation_422_errors(client_and_fake):
    client, _ = client_and_fake

    # Empty message after trimming
    r1 = client.post("/chat", json={"session_id": "s-1", "message": "    "})
    assert r1.status_code == 422
    assert "detail" in r1.json()

    # Over 4000 chars
    r2 = client.post("/chat", json={"session_id": "s-1", "message": "a" * 4001})
    assert r2.status_code == 422

    # Missing session_id
    r3 = client.post("/chat", json={"message": "hello"})
    assert r3.status_code == 422

    # Malformed JSON
    r4 = client.post("/chat", content="{bad-json", headers={"Content-Type": "application/json"})
    assert r4.status_code == 422

    # Failed 422 requests must NOT create session history
    hist = client.get("/sessions/s-1/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


def test_cache_hit_across_new_sessions_and_faster_latency(client_and_fake):
    client, _ = client_and_fake

    r1 = client.post("/chat", json={"session_id": "sess-a", "message": "What is the capital of France?"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert d1["metadata"]["key_id"] == "key_1"
    assert d1["metadata"]["fallback_attempts"] == 0
    assert d1["metadata"]["input_tokens"] > 0
    assert d1["metadata"]["output_tokens"] > 0

    # Same first message in a brand-new session -> LLM_CACHE
    r2 = client.post("/chat", json={"session_id": "sess-b", "message": "What is the capital of France?"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["reply"] == d1["reply"]
    assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
    assert d2["metadata"]["key_id"] is None
    assert d2["metadata"]["fallback_attempts"] == 0
    assert d2["metadata"]["input_tokens"] == d1["metadata"]["input_tokens"]
    assert d2["metadata"]["output_tokens"] == d1["metadata"]["output_tokens"]
    assert d2["metadata"]["latency_ms"] < d1["metadata"]["latency_ms"]

    # Different message must NOT come from cache
    r3 = client.post("/chat", json={"session_id": "sess-c", "message": "What is the capital of Japan?"})
    assert r3.status_code == 200
    assert r3.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_multiturn_state_and_session_isolation(client_and_fake):
    client, fake = client_and_fake

    r1 = client.post("/chat", json={"session_id": "multi-1", "message": "My name is Ada."})
    assert r1.status_code == 200

    r2 = client.post("/chat", json={"session_id": "multi-1", "message": "What is my name?"})
    assert r2.status_code == 200

    # Second LLM call must include system + turn1 user + turn1 assistant + turn2 user
    second_call_messages = fake.calls[-1]["messages"]
    assert len(second_call_messages) == 4
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1] == {"role": "user", "content": "My name is Ada."}
    assert second_call_messages[2]["role"] == "assistant"
    assert second_call_messages[3] == {"role": "user", "content": "What is my name?"}

    # Verify stored messages endpoint excludes system prompt and preserves order
    hist = client.get("/sessions/multi-1/messages")
    assert hist.status_code == 200
    payload = hist.json()
    assert payload["session_id"] == "multi-1"
    assert len(payload["messages"]) == 4
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "user", "assistant"]
    assert payload["messages"][1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # Separate session does not see multi-1 history
    r_other = client.post("/chat", json={"session_id": "other-1", "message": "Hi there"})
    assert r_other.status_code == 200
    assert len(fake.calls[-1]["messages"]) == 2


def test_simulate_failure_header_failover_and_503(client_and_fake):
    client, _ = client_and_fake

    # key_1=429 -> fails over to key_2
    r1 = client.post(
        "/chat",
        json={"session_id": "sim-1", "message": "Test failover 429"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert r1.status_code == 200
    m1 = r1.json()["metadata"]
    assert m1["key_id"] == "key_2"
    assert m1["fallback_attempts"] == 1
    assert m1["resolution_source"] == "LIVE_INFERENCE"

    # key_1=500 -> fails over to key_2
    r2 = client.post(
        "/chat",
        json={"session_id": "sim-2", "message": "Test failover 500"},
        headers={"X-Simulate-Failure": "key_1=500"},
    )
    assert r2.status_code == 200
    assert r2.json()["metadata"]["key_id"] == "key_2"
    assert r2.json()["metadata"]["fallback_attempts"] == 1

    # *=429 -> 503 all providers unavailable, nothing saved to history
    r3 = client.post(
        "/chat",
        json={"session_id": "sim-all-fail", "message": "Should fail"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r3.status_code == 503
    assert r3.json() == {"detail": "all providers unavailable"}

    hist = client.get("/sessions/sim-all-fail/messages")
    assert hist.status_code == 404


def test_real_provider_429_failover():
    settings = _make_settings(enable_sim=True, cache_enabled=True)
    db = Database("sqlite:///:memory:")
    gateway = LLMGateway(settings)
    app = create_app(settings_override=settings, db_override=db, gateway_override=gateway)

    fake = _FakeLiteLLMCacheBehavior(fail_keys={"secret-test-key-1"})
    with patch.object(litellm, "completion", side_effect=fake):
        with TestClient(app) as client:
            res = client.post("/chat", json={"session_id": "real-fail", "message": "Hello"})
            assert res.status_code == 200
            meta = res.json()["metadata"]
            assert meta["key_id"] == "key_2"
            assert meta["fallback_attempts"] == 1
            assert "secret-test-key" not in str(res.json())


def test_failure_simulation_ignored_when_disabled():
    settings = _make_settings(enable_sim=False, cache_enabled=True)
    db = Database("sqlite:///:memory:")
    gateway = LLMGateway(settings)
    app = create_app(settings_override=settings, db_override=db, gateway_override=gateway)

    fake = _FakeLiteLLMCacheBehavior()
    with patch.object(litellm, "completion", side_effect=fake):
        with TestClient(app) as client:
            res = client.post(
                "/chat",
                json={"session_id": "disabled-sim", "message": "Hello"},
                headers={"X-Simulate-Failure": "*=429"},
            )
            assert res.status_code == 200
            assert res.json()["metadata"]["key_id"] == "key_1"
