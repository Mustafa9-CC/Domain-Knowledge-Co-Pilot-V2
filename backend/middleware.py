"""
Production-grade observability middleware.

Provides:
1. Structured JSON logging (no secrets leaked)
2. Request-ID generation and propagation (X-Request-ID header)
3. Per-request timing (X-Response-Time-Ms header)
4. Contextvar-based request ID for use in any logger
"""

import logging
import time
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Context variable: accessible from any module via get_request_id()
# ---------------------------------------------------------------------------
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request ID (or '-' if not inside a request)."""
    return _request_id_ctx.get()


# ---------------------------------------------------------------------------
# Structured JSON log formatter
# ---------------------------------------------------------------------------
class StructuredFormatter(logging.Formatter):
    """Emit one-line JSON log records.

    Fields: timestamp, level, logger, request_id, message.
    Secrets (Authorization, Cookie, API keys) are never included.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "rid": get_request_id(),
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Replace the root logger's handlers with a single structured handler.

    Call this once during application startup (before any log is emitted).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------
class RequestIdTimingMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID and measure response time.

    Incoming:
        If the client sends X-Request-ID, it is reused (useful for
        distributed tracing). Otherwise a new UUID4 is generated.

    Outgoing:
        X-Request-ID: <id>
        X-Response-Time-Ms: <elapsed_ms>

    The request ID is stored in a contextvar so every log line
    emitted during that request automatically includes it.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = _request_id_ctx.set(request_id)

        logger = logging.getLogger("backend.middleware")

        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                f"{request.method} {request.url.path} → 500 ({elapsed_ms:.0f}ms)"
            )
            raise
        finally:
            _request_id_ctx.reset(token)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Attach headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        # Structured access log (skip /health to reduce noise)
        if request.url.path != "/api/health":
            logger.info(
                f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.0f}ms)"
            )

        return response
