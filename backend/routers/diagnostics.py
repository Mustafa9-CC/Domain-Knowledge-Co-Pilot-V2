"""
Diagnostics router — internal health-check APIs.

Provides deep inspection of document and corpus health, including
cross-system consistency between SQLite and ChromaDB.

These endpoints are authenticated (require valid JWT) but are intended
for operators and debugging, not for end-user UI.
"""

import os
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Corpus, Document
from backend.schemas import DocumentDiagnostics, CorpusDiagnostics
from backend.auth import get_current_user
from backend.services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


# ---------------------------------------------------------------------------
# Document-level diagnostics
# ---------------------------------------------------------------------------

@router.get("/documents/{document_id}", response_model=DocumentDiagnostics)
def document_diagnostics(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full health report for a single document.

    Checks:
    - SQLite record exists and is owned by user
    - File exists on disk
    - chunk_count in SQLite matches vector count in ChromaDB
    - All expected chunk indices are present in ChromaDB
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # File existence
    file_exists = bool(document.file_path and os.path.exists(document.file_path))

    # Vector store verification
    verification = vector_store.verify_document(
        document_id=document.id,
        expected_chunks=document.chunk_count,
    )

    return DocumentDiagnostics(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        file_type=document.file_type,
        file_size=document.file_size,
        file_exists=file_exists,
        chunk_count_sql=document.chunk_count,
        vector_count_chroma=verification.get("vector_count", -1),
        consistent=verification.get("consistent", False),
        missing_indices=verification.get("missing_indices", []),
        extra_indices=verification.get("extra_indices", []),
        process_error=document.process_error,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


# ---------------------------------------------------------------------------
# Corpus-level diagnostics
# ---------------------------------------------------------------------------

@router.get("/corpora/{corpus_id}", response_model=CorpusDiagnostics)
def corpus_diagnostics(
    corpus_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate health report for a corpus.

    Checks:
    - Document counts by status (ready, failed, processing)
    - Total chunk_count from SQLite vs total vectors in ChromaDB
    - Per-document consistency (flags inconsistent document IDs)
    """
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus not found",
        )

    documents = db.query(Document).filter(Document.corpus_id == corpus_id).all()

    total = len(documents)
    ready = sum(1 for d in documents if d.status == "ready")
    failed = sum(1 for d in documents if d.status == "failed")
    processing = sum(1 for d in documents if d.status in ("uploaded", "processing"))

    # Sum expected chunks from SQLite
    total_chunks_sql = sum(d.chunk_count or 0 for d in documents)

    # Actual vectors from ChromaDB
    total_vectors_chroma = vector_store.count_by_corpus(corpus_id)

    # Per-document vector counts for consistency check
    doc_vector_counts = vector_store.corpus_doc_vector_counts(corpus_id)
    inconsistent = []
    for doc in documents:
        if doc.status == "ready":
            expected = doc.chunk_count or 0
            actual = doc_vector_counts.get(doc.id, 0)
            if expected != actual:
                inconsistent.append(doc.id)

    storage_consistent = (
        total_chunks_sql == total_vectors_chroma
        and len(inconsistent) == 0
    )

    return CorpusDiagnostics(
        corpus_id=corpus.id,
        corpus_name=corpus.name,
        total_documents=total,
        documents_ready=ready,
        documents_failed=failed,
        documents_processing=processing,
        total_chunks_sql=total_chunks_sql,
        total_vectors_chroma=total_vectors_chroma,
        storage_consistent=storage_consistent,
        inconsistent_documents=inconsistent,
        created_at=corpus.created_at,
    )


# ---------------------------------------------------------------------------
# Retrieval diagnostics (standalone probe)
# ---------------------------------------------------------------------------

@router.post("/corpora/{corpus_id}/retrieval-probe")
def retrieval_probe(
    corpus_id: int,
    query: str = "test",
    top_k: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Test retrieval without calling the LLM.

    Returns raw retrieval diagnostics: chunks, scores, timing,
    document coverage, and a prompt token estimate.
    """
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus not found",
        )

    # Count total documents for coverage calculation
    total_docs = db.query(Document).filter(
        Document.corpus_id == corpus_id,
        Document.status == "ready",
    ).count()

    t0 = time.perf_counter()
    chunks = vector_store.query(
        query_text=query,
        corpus_id=corpus_id,
        top_k=top_k,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Calculate coverage and score statistics
    unique_docs = set()
    scores = []
    total_context_chars = 0
    for c in chunks:
        unique_docs.add(c["document_id"])
        scores.append(c["score"])
        total_context_chars += len(c.get("chunk_text", ""))

    # Rough token estimate (1 token ≈ 4 chars for English)
    prompt_token_estimate = total_context_chars // 4

    return {
        "query": query,
        "corpus_id": corpus_id,
        "top_k": top_k,
        "chunks_retrieved": len(chunks),
        "unique_documents": len(unique_docs),
        "document_ids": sorted(unique_docs),
        "total_ready_documents": total_docs,
        "coverage": f"{len(unique_docs)}/{total_docs}",
        "scores": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": round(sum(scores) / len(scores), 4) if scores else None,
        },
        "prompt_token_estimate": prompt_token_estimate,
        "retrieval_latency_ms": round(retrieval_ms, 1),
        "chunks": [
            {
                "document_id": c["document_id"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
                "text_length": len(c.get("chunk_text", "")),
            }
            for c in chunks
        ],
    }
