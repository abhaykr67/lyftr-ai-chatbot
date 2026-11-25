import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import litellm

try:
    from litellm.caching.caching import Cache
except ImportError:
    from litellm.caching import Cache  # type: ignore

from app.config import KeyEntry, Settings

logger = logging.getLogger(__name__)

VALID_SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails to produce a response."""


@dataclass
class GatewayResult:
    reply: str
    resolution_source: str  # "LIVE_INFERENCE" or "LLM_CACHE"
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


def parse_simulate_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    """Parse X-Simulate-Failure header like 'key_1=429,key_2=500' or '*=429'."""
    if not header_value:
        return {}

    rules: Dict[str, int] = {}
    for pair in header_value.split(","):
        part = pair.strip()
        if "=" not in part:
            continue
        target, raw_status = part.split("=", 1)
        target = target.strip()
        raw_status = raw_status.strip()
        if not target:
            continue
        try:
            status_code = int(raw_status)
        except ValueError:
            continue
        if status_code in VALID_SIMULATED_STATUSES:
            rules[target] = status_code
    return rules


def _extract_reply_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()

    return str(content).strip() if content is not None else ""


def _extract_usage(
    response: Any, model: str, messages: List[Dict[str, str]], reply: str
) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens = None
    completion_tokens = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

    if prompt_tokens is None:
        try:
            prompt_tokens = int(litellm.token_counter(model=model, messages=messages))
        except Exception:
            prompt_tokens = max(1, sum(len(m["content"].split()) for m in messages))

    if completion_tokens is None:
        try:
            completion_tokens = int(litellm.token_counter(model=model, text=reply))
        except Exception:
            completion_tokens = max(1, len(reply.split()))

    return int(prompt_tokens), int(completion_tokens)


def _is_cache_hit(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("cache_hit") is True:
        return True
    if isinstance(response, dict):
        hidden_dict = response.get("_hidden_params")
        if isinstance(hidden_dict, dict) and hidden_dict.get("cache_hit") is True:
            return True
    return False


def _is_failover_exception(exc: Exception) -> bool:
    """Determine whether a provider exception should trigger key failover."""
    failover_types = tuple(
        t
        for t in (
            getattr(litellm, "RateLimitError", None),
            getattr(litellm, "InternalServerError", None),
            getattr(litellm, "ServiceUnavailableError", None),
            getattr(litellm, "BadGatewayError", None),
            getattr(litellm, "Timeout", None),
            getattr(litellm, "APIConnectionError", None),
            getattr(litellm, "AuthenticationError", None),
            getattr(litellm, "PermissionDeniedError", None),
            getattr(litellm, "BudgetExceededError", None),
        )
        if t is not None
    )
    if failover_types and isinstance(exc, failover_types):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500 or status_code in {401, 403}):
        return True

    msg = str(exc).lower()
    keywords = (
        "429",
        "rate limit",
        "ratelimit",
        "quota",
        "resource_exhausted",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "service unavailable",
        "bad gateway",
        "timeout",
        "temporarily unavailable",
    )
    return any(kw in msg for kw in keywords)


class LLMGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.key_pool: List[KeyEntry] = list(settings.key_pool)
        self.cache_enabled = settings.cache_enabled

        # Silence verbose provider debug traces while keeping clean app logs
        litellm.suppress_debug_info = True
        litellm.drop_params = True

        if self.cache_enabled:
            if settings.cache_type == "disk":
                os.makedirs(settings.disk_cache_dir, exist_ok=True)
                litellm.cache = Cache(
                    type="disk",
                    disk_cache_dir=settings.disk_cache_dir,
                )
            else:
                litellm.cache = Cache(type="local")
        else:
            litellm.cache = None

    def complete(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> GatewayResult:
        if not self.key_pool:
            raise AllProvidersUnavailableError("No LLM API keys configured")

        simulated_failures: Dict[str, int] = {}
        if self.settings.enable_failure_simulation and simulate_failure_header:
            simulated_failures = parse_simulate_failure_header(simulate_failure_header)

        # When X-Simulate-Failure is active for this request, bypass cache so the
        # failover path is exercised deterministically rather than returning a cached reply.
        use_cache = self.cache_enabled and not bool(simulated_failures)

        start = time.perf_counter()
        fallback_attempts = 0
        last_error: Optional[Exception] = None

        for key_entry in self.key_pool:
            # 1. Check per-request simulated failure hook
            simulated_status = simulated_failures.get("*") or simulated_failures.get(key_entry.key_id)
            if simulated_status is not None:
                logger.warning(
                    "Simulated HTTP %s failure for %s (model=%s)",
                    simulated_status,
                    key_entry.key_id,
                    key_entry.model,
                )
                fallback_attempts += 1
                last_error = RuntimeError(
                    f"Simulated failure HTTP {simulated_status} on {key_entry.key_id}"
                )
                continue

            # 2. Invoke LiteLLM completion (with built-in cache when enabled)
            call_kwargs: Dict[str, Any] = {
                "model": key_entry.model,
                "messages": messages,
                "api_key": key_entry.api_key,
                "timeout": self.settings.request_timeout_s,
                "num_retries": 0,  # Let our key pool handle failover immediately
                "caching": use_cache,
            }
            if not use_cache:
                call_kwargs["cache"] = {"no-cache": True, "no-store": True}

            try:
                response = litellm.completion(**call_kwargs)
                reply_text = _extract_reply_text(response)
                if not reply_text:
                    raise RuntimeError(f"Empty completion returned by {key_entry.key_id}")

                elapsed_ms = int((time.perf_counter() - start) * 1000)
                input_tokens, output_tokens = _extract_usage(
                    response, key_entry.model, messages, reply_text
                )
                cache_hit = use_cache and _is_cache_hit(response)

                if cache_hit:
                    return GatewayResult(
                        reply=reply_text,
                        resolution_source="LLM_CACHE",
                        model=key_entry.model,
                        latency_ms=elapsed_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

                return GatewayResult(
                    reply=reply_text,
                    resolution_source="LIVE_INFERENCE",
                    model=key_entry.model,
                    latency_ms=elapsed_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=key_entry.key_id,
                    fallback_attempts=fallback_attempts,
                )

            except Exception as exc:
                last_error = exc
                if _is_failover_exception(exc):
                    logger.warning(
                        "Failover triggered on %s (%s): %s",
                        key_entry.key_id,
                        type(exc).__name__,
                        str(exc)[:200],
                    )
                    fallback_attempts += 1
                    continue

                logger.error(
                    "Provider error on %s (%s): %s",
                    key_entry.key_id,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                fallback_attempts += 1
                continue

        logger.error("All %d configured keys failed. Last error: %s", len(self.key_pool), last_error)
        raise AllProvidersUnavailableError("all providers unavailable")
