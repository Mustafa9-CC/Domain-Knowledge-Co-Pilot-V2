import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import STARTUP_WARMUP
from backend.database import init_db
from backend.middleware import RequestIdTimingMiddleware, configure_logging
from backend.routers import health, auth, corpora, documents, chat, cross_corpus_chat, diagnostics
from backend.services.document_worker import DocumentWorker

logger = logging.getLogger(__name__)

# Module-level worker instance so lifespan can start/stop it
_document_worker = DocumentWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    if STARTUP_WARMUP:
        from backend.services.vector_store import vector_store
        from backend.services.bm25_index import bm25_index
        logger.info("Startup warmup enabled — pre-loading embedding model, ChromaDB, and BM25 indices")
        vector_store.warmup()
        bm25_index.warmup()
    _document_worker.start()
    yield
    _document_worker.stop()
    from backend.services.llm_service import close_client
    close_client()


app = FastAPI(title="Domain Knowledge Co-Pilot", lifespan=lifespan)

# Request-ID + timing middleware (runs before CORS)
app.add_middleware(RequestIdTimingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(corpora.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(cross_corpus_chat.router, prefix="/api")
app.include_router(diagnostics.router, prefix="/api")
