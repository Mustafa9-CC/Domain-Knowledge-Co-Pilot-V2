import os
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Corpus, Document
from backend.schemas import CorpusCreate, CorpusResponse
from backend.auth import get_current_user
from backend.config import UPLOAD_DIR
from backend.services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpora", tags=["corpora"])


@router.post("", response_model=CorpusResponse, status_code=status.HTTP_201_CREATED)
def create_corpus(
    body: CorpusCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = Corpus(
        user_id=current_user.id,
        name=body.name,
        description=body.description,
    )
    db.add(corpus)
    db.commit()
    db.refresh(corpus)
    return CorpusResponse(
        id=corpus.id,
        name=corpus.name,
        description=corpus.description,
        doc_count=0,
        created_at=corpus.created_at,
    )


@router.get("", response_model=list[CorpusResponse])
def list_corpora(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpora = db.query(Corpus).filter(Corpus.user_id == current_user.id).all()
    results = []
    for c in corpora:
        doc_count = db.query(Document).filter(Document.corpus_id == c.id).count()
        results.append(CorpusResponse(
            id=c.id,
            name=c.name,
            description=c.description,
            doc_count=doc_count,
            created_at=c.created_at,
        ))
    return results


@router.get("/{corpus_id}", response_model=CorpusResponse)
def get_corpus(
    corpus_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")
    doc_count = db.query(Document).filter(Document.corpus_id == corpus.id).count()
    return CorpusResponse(
        id=corpus.id,
        name=corpus.name,
        description=corpus.description,
        doc_count=doc_count,
        created_at=corpus.created_at,
    )


@router.delete("/{corpus_id}", status_code=status.HTTP_200_OK)
def delete_corpus(
    corpus_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")

    # --- Safe deletion order: ChromaDB → disk → SQL ---
    #
    # 1. Delete ChromaDB vectors FIRST.  If this fails we abort
    #    so SQL records stay intact (user can retry).
    try:
        from backend.services.bm25_index import bm25_index
        vector_store.delete_by_corpus(corpus_id)
        bm25_index.remove_corpus(corpus_id)
    except Exception as e:
        logger.exception(f"Corpus {corpus_id}: vector deletion failed, aborting delete")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete corpus vectors: {e}. "
                   f"Corpus was not deleted. Please try again.",
        )

    # 2. Disk cleanup — best-effort (orphaned files are harmless)
    documents = db.query(Document).filter(Document.corpus_id == corpus_id).all()
    for doc in documents:
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except OSError as e:
                logger.warning(f"Corpus {corpus_id}: failed to delete file {doc.file_path}: {e}")

    # 3. SQL deletion last — SQLAlchemy cascade handles:
    #    documents, chat_sessions, chat_messages
    db.delete(corpus)
    db.commit()
    return {"ok": True}

