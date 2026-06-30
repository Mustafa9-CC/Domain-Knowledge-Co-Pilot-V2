"""
Document CRUD router with background processing pipeline.

Upload flow (Phase 5.1):
  save file → create Document record → create DocumentProcessingJob → return HTTP 202.
  Background worker (document_worker.py) processes: extract → chunk → embed → store.

Transaction safety guarantees (Phase 1):
- Zero chunks → marked as FAILED
- ChromaDB failure during upload → compensating cleanup by worker
- Deletion order: ChromaDB → disk → SQL (abort on vector failure)
- process_error persisted for every failure
"""

import os
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Corpus, Document, DocumentProcessingJob
from backend.schemas import DocumentResponse
from backend.auth import get_current_user
from backend.config import UPLOAD_DIR, DOCUMENT_JOB_MAX_ATTEMPTS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def _get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _build_response(doc: Document) -> DocumentResponse:
    """Single source of truth for building a DocumentResponse.

    Avoids field omission bugs that occur when each endpoint
    manually constructs the response dict.
    """
    # Pull processing job info if available
    job = doc.processing_job
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        status=doc.status,
        chunk_count=doc.chunk_count,
        process_error=doc.process_error,
        processing_stage=job.stage if job else None,
        processing_progress=job.progress if job else None,
        processing_attempts=job.attempts if job else None,
        created_at=doc.created_at,
    )


@router.post(
    "/corpora/{corpus_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_document(
    corpus_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify corpus exists and belongs to user
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")

    # Validate file extension
    ext = _get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read file contents
    contents = file.file.read()
    file_size = len(contents)

    # Save file to disk
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Release upload bytes from memory
    del contents

    # Create document record (status: processing)
    document = Document(
        corpus_id=corpus_id,
        user_id=current_user.id,
        filename=file.filename,
        file_type=ext,
        file_size=file_size,
        file_path=file_path,
        status="processing",
    )
    db.add(document)
    db.flush()  # Get the document ID without committing

    # Create background processing job
    job = DocumentProcessingJob(
        document_id=document.id,
        status="queued",
        stage="queued",
        progress=0.0,
        attempts=0,
        max_attempts=DOCUMENT_JOB_MAX_ATTEMPTS,
    )
    db.add(job)
    db.commit()
    db.refresh(document)

    logger.info(
        f"Document {document.id} ({file.filename}): queued for background processing"
    )
    return _build_response(document)


@router.get("/corpora/{corpus_id}/documents", response_model=list[DocumentResponse])
def list_documents(
    corpus_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify corpus exists and belongs to user
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")

    documents = db.query(Document).filter(Document.corpus_id == corpus_id).all()
    return [_build_response(doc) for doc in documents]


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return _build_response(document)


@router.get("/documents/{document_id}/preview")
def preview_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the first 500 characters of extracted text from a document."""
    from backend.services.document_processor import extract_text

    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if document.status != "ready":
        return {"preview": "", "message": f"Document status is '{document.status}', not ready for preview."}

    try:
        text = extract_text(document.file_path, document.file_type)
        preview = text[:500].strip()
        return {"preview": preview}
    except Exception as e:
        return {"preview": "", "message": f"Failed to extract preview: {e}"}


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id,
    ).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Cancel any pending processing job before deletion
    job = db.query(DocumentProcessingJob).filter(
        DocumentProcessingJob.document_id == document_id,
    ).first()
    if job and job.status in ("queued", "processing"):
        logger.info(f"Document {document_id}: cancelling processing job {job.id}")
        job.status = "cancelled"
        job.stage = "cancelled"
        db.commit()

    # --- Safe deletion order: ChromaDB → disk → SQL ---
    #
    # 1. Delete ChromaDB vectors FIRST. If this fails, we abort
    #    entirely so the DB record and file remain (consistent state,
    #    user can retry).
    # 2. Delete file from disk (non-critical — orphaned files are
    #    harmless and can be cleaned up later).
    # 3. Delete SQL record last — this is the "point of no return".
    if document.status == "ready":
        try:
            from backend.services.vector_store import vector_store
            from backend.services.bm25_index import bm25_index
            vector_store.delete_by_document(document_id)
            try:
                bm25_index.remove_document(document.corpus_id, document_id)
            except Exception as e:
                logger.warning(f"Document {document_id}: BM25 removal failed: {e}")
        except Exception as e:
            logger.exception(f"Document {document_id}: vector deletion failed, aborting delete")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete document vectors: {e}. "
                       f"Document was not deleted. Please try again.",
            )

    # Disk cleanup (best-effort — log but don't abort)
    if document.file_path and os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except OSError as e:
            logger.warning(f"Document {document_id}: failed to delete file {document.file_path}: {e}")

    db.delete(document)
    db.commit()
    return {"ok": True}
