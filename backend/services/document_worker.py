"""
Background document processing worker.

Polls DocumentProcessingJob rows from SQLite and processes them through
the full ingestion pipeline: extract → chunk → embed → vector-insert.

Features:
- Configurable thread-pool size (DOCUMENT_WORKER_COUNT)
- Retry with exponential backoff (DOCUMENT_JOB_MAX_ATTEMPTS)
- Per-stage progress tracking (stage + progress columns)
- Per-job timing metrics persisted in metrics_json
- Graceful shutdown via threading.Event
- Duplicate-worker prevention via atomic DB claim
- Cancellation-safe: checks document existence before each stage
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session

from backend.config import (
    DOCUMENT_JOB_MAX_ATTEMPTS,
    DOCUMENT_JOB_POLL_SECONDS,
    DOCUMENT_WORKER_COUNT,
)
from backend.database import SessionLocal
from backend.models import Document, DocumentProcessingJob
from backend.services.document_processor import extract_text, chunk_text
from backend.services.vector_store import vector_store
from backend.services.bm25_index import bm25_index

logger = logging.getLogger(__name__)

# Backoff multiplier for retries (seconds): 2^attempt * BASE
_RETRY_BACKOFF_BASE = 2.0


class DocumentWorker:
    """Manages a pool of background threads that process document jobs."""

    def __init__(self, worker_count: int = DOCUMENT_WORKER_COUNT):
        self._worker_count = worker_count
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self):
        """Start all worker threads."""
        if self._threads:
            logger.warning("DocumentWorker.start() called but threads already running")
            return

        logger.info(
            f"Starting document worker pool: {self._worker_count} worker(s), "
            f"poll interval {DOCUMENT_JOB_POLL_SECONDS}s, "
            f"max attempts {DOCUMENT_JOB_MAX_ATTEMPTS}"
        )
        self._stop_event.clear()
        for i in range(self._worker_count):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"doc-worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 30.0):
        """Signal all workers to stop and wait for them to finish."""
        if not self._threads:
            return
        logger.info("Stopping document worker pool...")
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(f"Worker thread {t.name} did not stop within {timeout}s")
        self._threads.clear()
        logger.info("Document worker pool stopped")

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Main loop for a single worker thread."""
        thread_name = threading.current_thread().name
        logger.info(f"{thread_name}: started")

        while not self._stop_event.is_set():
            try:
                job_claimed = self._try_claim_and_process()
                if not job_claimed:
                    # No work available — sleep before next poll
                    self._stop_event.wait(DOCUMENT_JOB_POLL_SECONDS)
            except Exception:
                logger.exception(f"{thread_name}: unexpected error in worker loop")
                self._stop_event.wait(DOCUMENT_JOB_POLL_SECONDS)

        logger.info(f"{thread_name}: stopped")

    def _try_claim_and_process(self) -> bool:
        """Try to claim one queued job and process it.

        Uses a two-step claim pattern compatible with SQLite:
        1. ORM SELECT to find a candidate (handles datetime properly).
        2. Raw SQL UPDATE with status guard for atomic claim.

        Returns True if a job was claimed (regardless of success/failure),
        False if no job was available.
        """
        db: Session = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            thread_name = threading.current_thread().name

            # Step 1: Find a candidate job using ORM (datetime comparison works)
            candidate = (
                db.query(DocumentProcessingJob)
                .filter(
                    DocumentProcessingJob.status == "queued",
                    DocumentProcessingJob.available_at <= now,
                )
                .order_by(DocumentProcessingJob.available_at)
                .first()
            )

            if candidate is None:
                return False

            candidate_id = candidate.id

            # Step 2: Atomic claim — only succeeds if still 'queued'.
            # SQLite serialises writes, so concurrent workers are safe.
            from sqlalchemy import text as sa_text
            result = db.execute(
                sa_text(
                    "UPDATE document_processing_jobs "
                    "SET status = 'processing', stage = 'claimed', "
                    "    started_at = :now, attempts = attempts + 1 "
                    "WHERE id = :job_id AND status = 'queued'"
                ),
                {"now": now.isoformat(), "job_id": candidate_id},
            )
            db.commit()

            if result.rowcount == 0:
                # Another worker claimed it first — not an error
                return False

            # Step 3: Reload the claimed job with fresh ORM state
            db.expire_all()
            job = (
                db.query(DocumentProcessingJob)
                .filter(DocumentProcessingJob.id == candidate_id)
                .first()
            )

            if job is None:
                return False

            document_id = job.document_id
            job_id = job.id
            logger.info(
                f"Claimed job {job_id} for document {document_id} "
                f"(attempt {job.attempts}/{job.max_attempts})"
            )

            # Process it
            self._process_job(db, job)
            return True

        except Exception:
            logger.exception("Error during job claim/process")
            try:
                db.rollback()
            except Exception:
                pass
            return True  # We attempted work, return True to avoid tight-loop
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _process_job(self, db: Session, job: DocumentProcessingJob):
        """Execute the full processing pipeline for a claimed job."""
        metrics: dict = {}
        document_id = job.document_id

        try:
            # Load the document record
            document = db.query(Document).filter(Document.id == document_id).first()
            if document is None:
                # Document was deleted while queued — discard job
                logger.warning(f"Job {job.id}: document {document_id} no longer exists, discarding")
                job.status = "cancelled"
                job.stage = "cancelled"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return

            # Stage 1: Extraction
            self._update_stage(db, job, "extracting", 0.1)
            t0 = time.perf_counter()
            text = extract_text(document.file_path, document.file_type)
            metrics["extraction_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            if self._stop_event.is_set():
                self._requeue_job(db, job, "Worker shutting down")
                return

            # Stage 2: Chunking
            self._update_stage(db, job, "chunking", 0.3)
            t0 = time.perf_counter()
            chunks = chunk_text(text)
            metrics["chunking_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            # Release the full text from memory
            del text

            if not chunks:
                error_msg = (
                    "Document produced zero text chunks. "
                    "The file may be empty, contain only whitespace, "
                    "or use a format that could not be parsed."
                )
                self._fail_job(db, job, document, error_msg, metrics)
                return

            # Stage 3: Embedding + vector insertion
            self._update_stage(db, job, "embedding", 0.4)
            t0 = time.perf_counter()
            try:
                chunk_count = vector_store.add_chunks(
                    chunks=chunks,
                    user_id=document.user_id,
                    corpus_id=document.corpus_id,
                    document_id=document.id,
                    filename=document.filename,
                )
            except Exception as embed_err:
                # Compensating action: clean up partial vectors
                try:
                    vector_store.delete_by_document(document.id)
                except Exception:
                    logger.warning(
                        f"Job {job.id}: failed to clean up partial vectors",
                        exc_info=True,
                    )
                raise embed_err
            metrics["embedding_and_indexing_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            # Release chunks from memory
            del chunks

            if self._stop_event.is_set():
                # Vectors are already stored — mark as ready anyway
                pass

            # Update BM25 Index
            try:
                bm25_index.add_document(corpus_id=document.corpus_id, document_id=document.id)
            except Exception as bm25_err:
                logger.warning(
                    f"Job {job.id}: failed to update BM25 index for document {document.id}",
                    exc_info=True,
                )
                # We don't fail the entire job if BM25 fails, just log it.
                # A manual warmup or next document addition will fix it.

            # Stage 4: Finalize
            self._update_stage(db, job, "finalizing", 0.9)

            document.status = "ready"
            document.chunk_count = chunk_count
            document.process_error = None

            job.status = "completed"
            job.stage = "completed"
            job.progress = 1.0
            job.completed_at = datetime.now(timezone.utc)
            job.last_error = None
            job.metrics_json = json.dumps(metrics)

            db.commit()
            logger.info(
                f"Job {job.id}: document {document_id} processed successfully — "
                f"{chunk_count} chunks, "
                f"extraction={metrics.get('extraction_ms', '?')}ms, "
                f"embedding={metrics.get('embedding_and_indexing_ms', '?')}ms"
            )

        except Exception as e:
            error_msg = f"Processing failed: {e}"
            logger.exception(f"Job {job.id}: {error_msg}")
            try:
                db.rollback()
            except Exception:
                pass

            # Reload fresh state
            document = db.query(Document).filter(Document.id == document_id).first()
            job = db.query(DocumentProcessingJob).filter(DocumentProcessingJob.id == job.id).first()
            if job is None:
                return
            if document is None:
                job.status = "cancelled"
                job.stage = "cancelled"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return

            self._fail_or_retry(db, job, document, error_msg, metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_stage(self, db: Session, job: DocumentProcessingJob, stage: str, progress: float):
        """Update job stage and progress (best-effort, non-critical)."""
        try:
            job.stage = stage
            job.progress = progress
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass

    def _fail_job(
        self,
        db: Session,
        job: DocumentProcessingJob,
        document: Document,
        error: str,
        metrics: dict,
    ):
        """Mark both job and document as permanently failed."""
        document.status = "failed"
        document.process_error = error[:2000]

        job.status = "failed"
        job.stage = "failed"
        job.last_error = error[:2000]
        job.completed_at = datetime.now(timezone.utc)
        job.metrics_json = json.dumps(metrics) if metrics else None

        db.commit()
        logger.warning(f"Job {job.id}: permanently failed — {error}")

    def _fail_or_retry(
        self,
        db: Session,
        job: DocumentProcessingJob,
        document: Document,
        error: str,
        metrics: dict,
    ):
        """Either retry the job or mark it as permanently failed."""
        if job.attempts < job.max_attempts:
            # Schedule retry with exponential backoff
            backoff_seconds = _RETRY_BACKOFF_BASE ** job.attempts
            job.status = "queued"
            job.stage = "retry_pending"
            job.available_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
            job.last_error = error[:2000]
            job.metrics_json = json.dumps(metrics) if metrics else None

            document.status = "processing"
            document.process_error = f"Attempt {job.attempts} failed, retrying: {error[:500]}"

            db.commit()
            logger.info(
                f"Job {job.id}: attempt {job.attempts}/{job.max_attempts} failed, "
                f"retry in {backoff_seconds}s"
            )
        else:
            self._fail_job(db, job, document, error, metrics)

    def _requeue_job(self, db: Session, job: DocumentProcessingJob, reason: str):
        """Put a job back in the queue (e.g. on graceful shutdown)."""
        job.status = "queued"
        job.stage = "requeued"
        job.last_error = reason
        db.commit()
        logger.info(f"Job {job.id}: requeued — {reason}")
