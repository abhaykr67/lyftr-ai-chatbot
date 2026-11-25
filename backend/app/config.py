import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class KeyEntry:
    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    key_pool: List[KeyEntry]
    default_model: str
    database_url: str
    allowed_origins: List[str]
    enable_failure_simulation: bool
    cache_enabled: bool
    cache_type: str
    disk_cache_dir: str
    default_system_prompt: str
    request_timeout_s: float


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    raw_keys = os.getenv("LLM_API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    default_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip() or "gemini/gemini-2.0-flash"
    raw_models = os.getenv("LLM_MODELS", "")
    models = [m.strip() for m in raw_models.split(",") if m.strip()]

    key_pool: List[KeyEntry] = []
    for idx, api_key in enumerate(keys, start=1):
        model_for_key = models[idx - 1] if idx - 1 < len(models) else default_model
        key_pool.append(
            KeyEntry(
                key_id=f"key_{idx}",
                api_key=api_key,
                model=model_for_key,
            )
        )

    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip()
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
    )
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    enable_failure_simulation = _parse_bool(
        os.getenv("ENABLE_FAILURE_SIMULATION"), default=False
    )
    cache_enabled = _parse_bool(os.getenv("CACHE_ENABLED"), default=True)
    cache_type = os.getenv("CACHE_TYPE", "local").strip().lower()
    if cache_type not in {"local", "disk"}:
        cache_type = "local"

    disk_cache_dir = os.getenv("DISK_CACHE_DIR", "/data/litellm_cache").strip()
    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    ).strip()
    request_timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    return Settings(
        key_pool=key_pool,
        default_model=default_model,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_failure_simulation,
        cache_enabled=cache_enabled,
        cache_type=cache_type,
        disk_cache_dir=disk_cache_dir,
        default_system_prompt=default_system_prompt,
        request_timeout_s=request_timeout_s,
    )
