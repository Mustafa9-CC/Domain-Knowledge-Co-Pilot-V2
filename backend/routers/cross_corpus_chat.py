import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, insert

from backend.database import get_db, engine
from backend.models import User, Corpus, ChatSession, ChatMessage, session_corpora
from backend.schemas import (
    CrossCorpusChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cross-corpus-chat"])

def _resolve_corpus_ids(requested_ids: list[int] | str | None, current_user: User, db: Session) -> list[int]:
    """Resolve requested corpus IDs, validating ownership."""
    if requested_ids == "all":
        corpora = db.query(Corpus.id).filter(Corpus.user_id == current_user.id).all()
        return [c.id for c in corpora]
    
    if not requested_ids:
        return []

    # Validate they exist and belong to user
    valid_corpora = db.query(Corpus.id).filter(
        Corpus.id.in_(requested_ids),
        Corpus.user_id == current_user.id
    ).all()
    
    valid_ids = [c.id for c in valid_corpora]
    if len(valid_ids) != len(requested_ids):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more corpora not found or access denied")
    
    return valid_ids

def _get_corpus_names(corpus_ids: list[int], db: Session) -> dict[int, str]:
    if not corpus_ids:
        return {}
    corpora = db.query(Corpus).filter(Corpus.id.in_(corpus_ids)).all()
    return {c.id: c.name for c in corpora}

def execute_chat_logic(
    corpus_ids: list[int],
    question: str,
    session_id: int | None,
    debug: bool,
    db: Session,
    current_user: User,
) -> ChatResponse:
    t_start = time.perf_counter()
    timing: dict[str, float] = {}

    # Step 1: Resolve or create session
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        ).first()
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        # Ensure the session has the requested corpus_ids? Or we use the session's corpus_ids.
        # For simplicity, if session_id is provided, we just append to it. 
        # But wait, what if the user queries different corpora in an existing session?
        # The design says we use the associated corpora of the session if it's already set.
    else:
        title = question[:40].strip()
        if len(question) > 40:
            title += "..."
        session = ChatSession(
            user_id=current_user.id,
            corpus_id=None,  # Nullable for cross-corpus
            title=title,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Populate session_corpora
        if corpus_ids:
            values = [{"session_id": session.id, "corpus_id": cid} for cid in corpus_ids]
            db.execute(insert(session_corpora).values(values))
            db.commit()

    # Step 2: Store user message
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    db.commit()

    # Step 3: Load last 5 conversation turns
    history_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    prior_messages = history_messages[:-1][-10:]
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in prior_messages
    ]

    corpus_names = _get_corpus_names(corpus_ids, db)

    # Step 4 & 5: Query ChromaDB
    t0 = time.perf_counter()
    retrieved_chunks = []
    if corpus_ids:
        retrieved_chunks = vector_store.query(
            query_text=question,
            corpus_ids=corpus_ids,
            top_k=config.TOP_K,
        )
        for chunk in retrieved_chunks:
            chunk["corpus_name"] = corpus_names.get(chunk.get("corpus_id", 0), "Unknown")
            
    timing["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 6: Build prompt
    t0 = time.perf_counter()
    prompt = prepare_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        conversation_history=conversation_history,
    )
    timing["prompt_build_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timing["prompt_token_estimate"] = prompt.estimated_input_tokens
    timing["context_chunks_included"] = len(prompt.included_chunks)

    # Step 7: Call Groq API
    t0 = time.perf_counter()
    if not prompt.included_chunks:
        answer = INSUFFICIENT_CONTEXT_ANSWER
        used_chunks = []
    else:
        try:
            raw_answer = call_groq(prompt.messages)
        except ValueError as e:
            logger.error(f"LLM call failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
            )
        answer, used_chunks = validate_answer_citations(
            raw_answer,
            prompt.included_chunks,
        )
    timing["llm_call_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 8: Build citations
    citations = [
        Citation(
            document_id=chunk["document_id"],
            filename=chunk["filename"],
            chunk_index=chunk["chunk_index"],
            corpus_id=chunk.get("corpus_id"),
            corpus_name=chunk.get("corpus_name"),
        )
        for chunk in used_chunks
    ]
    sources_json = json.dumps([c.model_dump() for c in citations])

    # Step 9: Store assistant message
    t0 = time.perf_counter()
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources_json=sources_json,
    )
    db.add(assistant_msg)
    db.commit()
    timing["db_store_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

    # Logging
    unique_docs = set(c["document_id"] for c in retrieved_chunks)
    scores = [c["score"] for c in retrieved_chunks]
    logger.info(
        f"CrossCorpusChat session={session.id} "
        f"chunks={len(retrieved_chunks)} docs={len(unique_docs)} "
        f"scores=[{min(scores):.3f}..{max(scores):.3f}] "
        f"total={timing['total_ms']:.0f}ms"
        if scores else
        f"CrossCorpusChat session={session.id} "
        f"chunks=0 total={timing['total_ms']:.0f}ms"
    )

    response = ChatResponse(
        answer=answer,
        citations=citations,
        session_id=session.id,
        corpus_ids=corpus_ids,
    )

    if debug:
        response.retrieved_chunks = [
            RetrievedChunk(
                filename=chunk["filename"],
                chunk_index=chunk["chunk_index"],
                chunk_text=chunk["chunk_text"],
                score=chunk["score"],
                corpus_id=chunk.get("corpus_id"),
                corpus_name=chunk.get("corpus_name"),
            )
            for chunk in retrieved_chunks
        ]
        response.timing_ms = timing

    return response

def execute_chat_stream_logic(
    corpus_ids: list[int],
    question: str,
    session_id: int | None,
    db: Session,
    current_user: User,
):
    if not config.ENABLE_STREAMING:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming is disabled. Set ENABLE_STREAMING=true to enable.",
        )

    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        ).first()
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    else:
        title = question[:40].strip()
        if len(question) > 40:
            title += "..."
        session = ChatSession(
            user_id=current_user.id,
            corpus_id=None,
            title=title,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        if corpus_ids:
            values = [{"session_id": session.id, "corpus_id": cid} for cid in corpus_ids]
            db.execute(insert(session_corpora).values(values))
            db.commit()

    user_msg = ChatMessage(session_id=session.id, role="user", content=question)
    db.add(user_msg)
    db.commit()

    history_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    prior_messages = history_messages[:-1][-10:]
    conversation_history = [
        {"role": msg.role, "content": msg.content} for msg in prior_messages
    ]

    corpus_names = _get_corpus_names(corpus_ids, db)

    retrieved_chunks = []
    if corpus_ids:
        retrieved_chunks = vector_store.query(
            query_text=question, corpus_ids=corpus_ids, top_k=config.TOP_K
        )
        for chunk in retrieved_chunks:
            chunk["corpus_name"] = corpus_names.get(chunk.get("corpus_id", 0), "Unknown")

    prompt = prepare_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        conversation_history=conversation_history,
    )

    s_id = session.id
    included_chunks = prompt.included_chunks

    def event_generator():
        try:
            if not included_chunks:
                answer = INSUFFICIENT_CONTEXT_ANSWER
                yield f"event: token\ndata: {json.dumps({'token': answer})}\n\n"
                yield f"event: done\ndata: {json.dumps({'answer': answer, 'citations': [], 'session_id': s_id, 'corpus_ids': corpus_ids})}\n\n"
                _store_assistant_message(s_id, answer, [])
                return

            full_answer = ""
            for token in call_groq_streaming(prompt.messages):
                full_answer += token
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            validated_answer, used_chunks = validate_answer_citations(
                full_answer, included_chunks
            )

            citations = [
                {
                    "document_id": c["document_id"],
                    "filename": c["filename"],
                    "chunk_index": c["chunk_index"],
                    "corpus_id": c.get("corpus_id"),
                    "corpus_name": c.get("corpus_name"),
                }
                for c in used_chunks
            ]

            yield f"event: done\ndata: {json.dumps({'answer': validated_answer, 'citations': citations, 'session_id': s_id, 'corpus_ids': corpus_ids})}\n\n"

            _store_assistant_message(s_id, validated_answer, citations)

        except Exception as e:
            logger.exception("Streaming chat error")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

def _store_assistant_message(session_id: int, answer: str, citations: list[dict]):
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        sources_json = json.dumps(citations)
        msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources_json=sources_json,
        )
        db.add(msg)
        db.commit()
    except Exception:
        logger.exception("Failed to store streaming assistant message")
        db.rollback()
    finally:
        db.close()

@router.post("/chat", response_model=ChatResponse)
def cross_corpus_chat(
    body: CrossCorpusChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Resolve corpora
    if body.session_id:
        # If continuing a session, fetch its corpora
        session = db.query(ChatSession).filter(
            ChatSession.id == body.session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
        # Load associated corpora
        if session.corpus_id is not None:
            corpus_ids = [session.corpus_id]
        else:
            rows = db.execute(select(session_corpora.c.corpus_id).where(session_corpora.c.session_id == body.session_id)).fetchall()
            corpus_ids = [r[0] for r in rows]
            
    else:
        corpus_ids = _resolve_corpus_ids(body.corpus_ids, current_user, db)
        
    return execute_chat_logic(corpus_ids, body.question, body.session_id, body.debug, db, current_user)

@router.post("/chat/stream")
def cross_corpus_chat_stream(
    body: CrossCorpusChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == body.session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
        if session.corpus_id is not None:
            corpus_ids = [session.corpus_id]
        else:
            rows = db.execute(select(session_corpora.c.corpus_id).where(session_corpora.c.session_id == body.session_id)).fetchall()
            corpus_ids = [r[0] for r in rows]
    else:
        corpus_ids = _resolve_corpus_ids(body.corpus_ids, current_user, db)

    return execute_chat_stream_logic(corpus_ids, body.question, body.session_id, db, current_user)

@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_cross_corpus_session(
    body: ChatSessionCreate,
    corpus_ids: list[int] | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = ChatSession(
        user_id=current_user.id,
        corpus_id=None,
        title=body.title or "New Chat",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    if corpus_ids:
        valid_corpora = _resolve_corpus_ids(corpus_ids, current_user, db)
        if valid_corpora:
            values = [{"session_id": session.id, "corpus_id": cid} for cid in valid_corpora]
            db.execute(insert(session_corpora).values(values))
            db.commit()

    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        message_count=0,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )

@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_all_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
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
