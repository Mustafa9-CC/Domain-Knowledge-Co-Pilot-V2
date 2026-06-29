from datetime import datetime
from pydantic import BaseModel


class SignupRequest(BaseModel):
    email: str
    username: str
    password: str


class SignupResponse(BaseModel):
    id: int
    email: str
    username: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CorpusCreate(BaseModel):
    name: str
    description: str = ""


class CorpusResponse(BaseModel):
    id: int
    name: str
    description: str
    doc_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int | None = None
    process_error: str | None = None
    processing_stage: str | None = None
    processing_progress: float | None = None
    processing_attempts: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionCreate(BaseModel):
    title: str = ""


class ChatSessionResponse(BaseModel):
    id: int
    title: str
    message_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources_json: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    question: str
    session_id: int | None = None
    debug: bool = False


class CrossCorpusChatRequest(BaseModel):
    question: str
    corpus_ids: list[int] | str | None = None  # None or "all" -> all user corpora
    session_id: int | None = None
    debug: bool = False
    retrieval_mode: str = "hybrid"  # "dense" | "bm25" | "hybrid"


class Citation(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    corpus_id: int | None = None
    corpus_name: str | None = None


class RetrievedChunk(BaseModel):
    filename: str
    chunk_index: int
    chunk_text: str
    score: float
    corpus_id: int | None = None
    corpus_name: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    session_id: int
    retrieved_chunks: list[RetrievedChunk] = []
    timing_ms: dict[str, float] | None = None
    corpus_ids: list[int] | None = None  # which corpora were searched


# ---------------------------------------------------------------------------
# Diagnostics models (internal APIs)
# ---------------------------------------------------------------------------

class DocumentDiagnostics(BaseModel):
    """Full health report for a single document."""
    document_id: int
    filename: str
    status: str
    file_type: str
    file_size: int
    file_exists: bool
    chunk_count_sql: int | None
    vector_count_chroma: int
    consistent: bool
    missing_indices: list[int] = []
    extra_indices: list[int] = []
    process_error: str | None = None
    processing_stage: str | None = None
    processing_progress: float | None = None
    processing_attempts: int | None = None
    processing_max_attempts: int | None = None
    created_at: datetime
    updated_at: datetime


class CorpusDiagnostics(BaseModel):
    """Aggregate health report for a corpus."""
    corpus_id: int
    corpus_name: str
    total_documents: int
    documents_ready: int
    documents_failed: int
    documents_processing: int
    total_chunks_sql: int
    total_vectors_chroma: int
    storage_consistent: bool
    inconsistent_documents: list[int] = []
    created_at: datetime
