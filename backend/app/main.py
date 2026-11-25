import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.database import Database
from app.llm_gateway import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
    StoredMessage,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    settings_override: Optional[Settings] = None,
    db_override: Optional[Database] = None,
    gateway_override: Optional[LLMGateway] = None,
) -> FastAPI:
    settings = settings_override or load_settings()
    db = db_override or Database(settings.database_url)
    gateway = gateway_override or LLMGateway(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db.init_schema()
        logger.info(
            "Startup complete: keys_configured=%d cache_enabled=%s failure_simulation=%s",
            len(settings.key_pool),
            gateway.cache_enabled,
            settings.enable_failure_simulation,
        )
        yield

    app = FastAPI(
        title="Lyftr AI LLM Chatbot Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.db = db
    app.state.gateway = gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if not app.state.db.check_health():
            raise HTTPException(status_code=503, detail="database unavailable")
        if len(app.state.gateway.key_pool) == 0:
            raise HTTPException(status_code=503, detail="no api keys configured")
        return HealthResponse(
            status="ok",
            keys_configured=len(app.state.gateway.key_pool),
            cache_enabled=app.state.gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        # Resolve effective system prompt (request override -> stored session prompt -> default)
        stored_system_prompt = app.state.db.get_session_system_prompt(payload.session_id)
        effective_system_prompt = (
            payload.system_prompt
            if payload.system_prompt is not None
            else (stored_system_prompt or app.state.settings.default_system_prompt)
        )

        # Build full LLM message array: system prompt + prior SQLite history + new user message
        history = app.state.db.get_messages(payload.session_id)
        llm_messages: List[Dict[str, str]] = [
            {"role": "system", "content": effective_system_prompt}
        ]
        for msg in history:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = app.state.gateway.complete(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError:
            # Never store failed requests in session history
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata_obj = ChatMetadata(
            resolution_source=result.resolution_source,  # type: ignore[arg-type]
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        # Persist conversation turn only after successful inference/cache hit
        app.state.db.save_turn(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=result.reply,
            assistant_metadata=metadata_obj.model_dump(),
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata_obj,
        )

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    def get_session_messages(session_id: str) -> SessionMessagesResponse:
        rows = app.state.db.get_messages(session_id)
        if not rows:
            raise HTTPException(status_code=404, detail="session not found")

        messages = [
            StoredMessage(
                role=row["role"],
                content=row["content"],
                metadata=row.get("metadata"),
            )
            for row in rows
        ]
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    return app


app = create_app()
