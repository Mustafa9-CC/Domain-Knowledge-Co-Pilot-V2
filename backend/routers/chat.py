"""
Chat router — RAG pipeline, sessions, and message history.

Handles: chat (POST), chat/stream (POST SSE), sessions (GET, POST, DELETE), messages (GET).
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Corpus, ChatSession, ChatMessage
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageResponse,
    Citation,
    RetrievedChunk,
)
from backend.auth import get_current_user
from backend import config
from backend.services.vector_store import vector_store
from backend.services.llm_service import (
    INSUFFICIENT_CONTEXT_ANSWER,
    call_groq,
    call_groq_streaming,
    prepare_prompt,
    validate_answer_citations,
)
from backend.routers.cross_corpus_chat import execute_chat_logic, execute_chat_stream_logic

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Chat endpoint (4.2, 4.6, 4.7)
# ---------------------------------------------------------------------------

@router.post("/corpora/{corpus_id}/chat", response_model=ChatResponse)
def chat(
    corpus_id: int,
    body: ChatRequest,
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

    return execute_chat_logic([corpus_id], body.question, body.session_id, body.debug, db, current_user)


# ---------------------------------------------------------------------------
# Streaming chat endpoint (5.6)
# ---------------------------------------------------------------------------

@router.post("/corpora/{corpus_id}/chat/stream")
def chat_stream(
    corpus_id: int,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stream LLM tokens via Server-Sent Events."""
    # Verify corpus
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")

    return execute_chat_stream_logic([corpus_id], body.question, body.session_id, db, current_user)


# ---------------------------------------------------------------------------
# Session endpoints (4.3, 4.4, 4.5)
# ---------------------------------------------------------------------------

@router.get("/corpora/{corpus_id}/sessions", response_model=list[ChatSessionResponse])
def list_sessions(
    corpus_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all chat sessions for a corpus (4.4)."""
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")

    sessions = (
        db.query(ChatSession)
        .filter(
            ChatSession.corpus_id == corpus_id,
            ChatSession.user_id == current_user.id,
        )
        .order_by(ChatSession.updated_at.desc())
        .all()
    )

    results = []
    for s in sessions:
        msg_count = db.query(ChatMessage).filter(ChatMessage.session_id == s.id).count()
        results.append(ChatSessionResponse(
            id=s.id,
            title=s.title,
            message_count=msg_count,
            created_at=s.created_at,
            updated_at=s.updated_at,
        ))
    return results


@router.post("/corpora/{corpus_id}/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    corpus_id: int,
    body: ChatSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new chat session."""
    corpus = db.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.user_id == current_user.id,
    ).first()
    if not corpus:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")

    session = ChatSession(
        user_id=current_user.id,
        corpus_id=corpus_id,
        title=body.title or "New Chat",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        message_count=0,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a session and cascade to messages (4.5)."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # SQLAlchemy cascade handles message deletion
    db.delete(session)
    db.commit()
    return {"ok": True}


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
def get_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Load all messages for a session (4.3)."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return [
        ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources_json=msg.sources_json,
            created_at=msg.created_at,
        )
        for msg in messages
    ]
