"""
Chat view — RAG chat interface with session management.

Implements Phase 5 tasks:
  5.1  Session list sidebar, message input, message history
  5.2  Display answer with numbered citation markers [1] [2]
  5.3  Citation pills below answer — click to expand chunk text
  5.4  Session history sidebar — list sessions, click to reload
  5.5  New session button
  5.6  Delete session button
  5.7  Debug toggle — collapsible "Retrieved Chunks" section
  5.8  Loading spinner during chat request
"""

import json
import streamlit as st
from frontend.api_client import api_client


def show_chat_page():
    try:
        all_corpora = api_client.list_corpora()
    except Exception as e:
        st.error(f"Failed to load corpora: {_extract_error(e)}")
        return

    if not all_corpora:
        st.warning("No corpora available. Go back to Corpora to create one.")
        if st.button("← Back to Corpora"):
            st.session_state["page"] = "corpora"
            st.rerun()
        return

    corpus_map = {c["id"]: c["name"] for c in all_corpora}

    # --- Initialize chat-specific session state ---
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = None
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False
    if "last_debug_chunks" not in st.session_state:
        st.session_state["last_debug_chunks"] = []

    # ===================================================================
    # SIDEBAR — Session list (5.4), New session (5.5), Delete (5.6)
    # ===================================================================
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Sessions")

        # 5.5 — New session button
        if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state["chat_session_id"] = None
            st.session_state["chat_messages"] = []
            st.session_state["last_debug_chunks"] = []
            st.rerun()

        # 5.4 — Load and display all sessions
        try:
            sessions = api_client.list_all_sessions()
        except Exception as e:
            st.error(f"Failed to load sessions: {_extract_error(e)}")
            sessions = []

        current_session_id = st.session_state.get("chat_session_id")

        if not sessions:
            st.caption("No sessions yet. Ask a question to start!")
        else:
            for sess in sessions:
                col_title, col_del = st.columns([5, 1])
                with col_title:
                    # Highlight the active session
                    label = sess["title"] or "Untitled"
                    is_active = sess["id"] == current_session_id
                    btn_label = f"{'▶ ' if is_active else ''}{label}"
                    if st.button(
                        btn_label,
                        key=f"sess_{sess['id']}",
                        use_container_width=True,
                    ):
                        _load_session(sess["id"])
                        st.rerun()

                # 5.6 — Delete session button
                with col_del:
                    if st.button("🗑️", key=f"del_sess_{sess['id']}"):
                        try:
                            api_client.delete_session(sess["id"])
                            # If we deleted the active session, reset
                            if sess["id"] == current_session_id:
                                st.session_state["chat_session_id"] = None
                                st.session_state["chat_messages"] = []
                                st.session_state["last_debug_chunks"] = []
                            st.rerun()
                        except Exception as e:
                            st.error(_extract_error(e))

    # ===================================================================
    # MAIN AREA — Chat interface
    # ===================================================================

    # --- Header with back button ---
    col_back, col_title, col_docs = st.columns([1, 5, 1])
    with col_back:
        if st.button("← Back"):
            st.session_state["page"] = "documents"
            st.session_state.pop("chat_session_id", None)
            st.session_state.pop("chat_messages", None)
            st.session_state.pop("last_debug_chunks", None)
            st.rerun()
    with col_title:
        session_title = _get_session_title()
        st.title(f"💬 {session_title}")
    with col_docs:
        if st.button("📄 Docs"):
            st.session_state["page"] = "documents"
            st.rerun()

    # Cross-corpus selectors
    default_selected = [st.session_state["corpus_id"]] if st.session_state.get("corpus_id") in corpus_map else []
    selected_corpus_ids = st.multiselect(
        "Search Scope (Corpora):",
        options=list(corpus_map.keys()),
        format_func=lambda x: corpus_map[x],
        default=st.session_state.get("selected_corpus_ids", default_selected)
    )
    st.session_state["selected_corpus_ids"] = selected_corpus_ids

    retrieval_mode = st.radio("Retrieval Mode:", ["hybrid", "dense", "bm25"], horizontal=True, index=0)
    st.session_state["retrieval_mode"] = retrieval_mode

    # 5.7 — Debug toggle
    debug_mode = st.toggle("🐛 Debug Mode", value=st.session_state.get("debug_mode", False), key="debug_toggle")
    st.session_state["debug_mode"] = debug_mode

    st.divider()

    # --- Display chat history (5.1, 5.2, 5.3) ---
    _render_messages(st.session_state.get("chat_messages", []))

    # 5.7 — Debug: show retrieved chunks from last response
    if debug_mode and st.session_state.get("last_debug_chunks"):
        _render_debug_chunks(st.session_state["last_debug_chunks"])

    st.divider()

    # --- Chat input (5.1, 5.8) ---
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Immediately show the user message
        st.session_state["chat_messages"].append({
            "role": "user",
            "content": question,
            "sources_json": None,
        })

        # Try streaming if enabled, fallback to non-streaming
        try:
            _handle_chat_request(selected_corpus_ids, question, debug_mode, retrieval_mode)
        except Exception as e:
            st.error(f"Chat failed: {_extract_error(e)}")

        st.rerun()


def _handle_chat_request(corpus_ids: list[int], question: str, debug_mode: bool, retrieval_mode: str):
    """Handle a chat request, using streaming if available."""
    session_id = st.session_state.get("chat_session_id")

    # Try streaming first
    try:
        streamed = False
        full_answer = ""
        final_data = None

        with st.chat_message("assistant"):
            placeholder = st.empty()
            for event_type, data in api_client.cross_corpus_chat_stream(question, corpus_ids, session_id, retrieval_mode=retrieval_mode):
                if event_type == "token":
                    full_answer += data.get("token", "")
                    placeholder.markdown(full_answer + "▌")
                    streamed = True
                elif event_type == "done":
                    final_data = data
                    placeholder.markdown(data.get("answer", full_answer))
                elif event_type == "error":
                    st.error(data.get("error", "Streaming error"))
                    return

        if streamed and final_data:
            st.session_state["chat_session_id"] = final_data.get("session_id")
            citations = final_data.get("citations", [])
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": final_data.get("answer", full_answer),
                "sources_json": json.dumps(citations) if citations else None,
            })
            st.session_state["last_debug_chunks"] = []
            return

    except Exception:
        # Streaming not available or failed — fall back to non-streaming
        pass

    # Non-streaming fallback
    with st.spinner("🤔 Thinking..."):
        response = api_client.cross_corpus_chat(
            question=question,
            corpus_ids=corpus_ids,
            session_id=session_id,
            debug=debug_mode,
            retrieval_mode=retrieval_mode,
        )

        st.session_state["chat_session_id"] = response["session_id"]
        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": response["answer"],
            "sources_json": json.dumps(
                [c for c in response.get("citations", [])]
            ) if response.get("citations") else None,
        })

        if debug_mode and response.get("retrieved_chunks"):
            st.session_state["last_debug_chunks"] = response["retrieved_chunks"]
        else:
            st.session_state["last_debug_chunks"] = []


# ---------------------------------------------------------------------------
# Helper: Load session messages from backend (5.4)
# ---------------------------------------------------------------------------

def _load_session(session_id: int):
    """Load all messages for a session and set as current."""
    try:
        messages = api_client.get_messages(session_id)
        st.session_state["chat_session_id"] = session_id
        st.session_state["chat_messages"] = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "sources_json": msg.get("sources_json"),
            }
            for msg in messages
        ]
        st.session_state["last_debug_chunks"] = []
    except Exception as e:
        st.error(f"Failed to load session: {_extract_error(e)}")


# ---------------------------------------------------------------------------
# Helper: Get current session title
# ---------------------------------------------------------------------------

def _get_session_title() -> str:
    session_id = st.session_state.get("chat_session_id")
    if not session_id:
        return "New Chat"
    # Try to find title from loaded sessions
    try:
        sessions = api_client.list_all_sessions()
        for s in sessions:
            if s["id"] == session_id:
                return s["title"] or "Untitled"
    except Exception:
        pass
    return "Chat"


# ---------------------------------------------------------------------------
# Helper: Render message list (5.1, 5.2, 5.3)
# ---------------------------------------------------------------------------

def _render_messages(messages: list):
    """Render the full conversation with citation support."""
    if not messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <div class="empty-state-title">Start a research chat</div>
                <div class="empty-state-desc">Ask questions about your uploaded documents. The assistant will retrieve relevant snippets and provide answers with citations.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            # 5.2 — Display content (citation markers like [1] [2] are already in the text)
            st.markdown(content)

            # 5.3 — Citation pills for assistant messages
            if role == "assistant" and msg.get("sources_json"):
                _render_citations(msg["sources_json"], idx)


# ---------------------------------------------------------------------------
# Helper: Render citation pills (5.3)
# ---------------------------------------------------------------------------

def _render_citations(sources_json: str, msg_idx: int):
    """Render expandable citation pills below an assistant message."""
    try:
        citations = json.loads(sources_json)
    except (json.JSONDecodeError, TypeError):
        return

    if not citations:
        return

    st.markdown("<div style='margin-top: 0.75rem; margin-bottom: 0.25rem; font-size: 13px; color: #6366F1; font-weight: 500;'>📎 Sources:</div>", unsafe_allow_html=True)

    num_cites = len(citations)
    cols = st.columns(max(num_cites, 4))

    for cite_idx, citation in enumerate(citations):
        filename = citation.get("filename", "Unknown")
        chunk_index = citation.get("chunk_index", 0)
        doc_id = citation.get("document_id", 0)

        corpus_name = citation.get("corpus_name")
        corpus_badge = f" ({corpus_name})" if corpus_name else ""
        short_name = filename[:15] + "..." if len(filename) > 18 else filename
        label = f"[{cite_idx + 1}] {short_name}{corpus_badge}"

        with cols[cite_idx]:
            with st.expander(label, expanded=False):
                if corpus_name:
                    st.markdown(f"**Corpus:** {corpus_name}")
                st.markdown(f"**Document:** {filename}")
                st.markdown(f"**Chunk Index:** {chunk_index}")
                st.markdown(f"**Document ID:** {doc_id}")
                try:
                    preview = api_client.preview_document(doc_id)
                    preview_text = preview.get("preview", "")
                    if preview_text:
                        st.markdown("<span style='font-size: 12px; color: #9CA3AF;'>Preview (first 500 chars):</span>", unsafe_allow_html=True)
                        st.text_area("Preview", value=preview_text[:500], height=120, disabled=True, key=f"cite_prev_{msg_idx}_{cite_idx}")
                except Exception:
                    st.caption("Preview not available.")


# ---------------------------------------------------------------------------
# Helper: Render debug chunks (5.7)
# ---------------------------------------------------------------------------

def _render_debug_chunks(chunks: list):
    """Render retrieved chunks in a collapsible debug section."""
    with st.expander("🐛 Retrieved Chunks (Debug)", expanded=False):
        if not chunks:
            st.info("No chunks retrieved.")
            return

        for i, chunk in enumerate(chunks):
            st.markdown(f"### Chunk {i + 1}")
            col1, col2 = st.columns([3, 1])
            with col1:
                if chunk.get("corpus_name"):
                    st.markdown(f"**Corpus:** {chunk.get('corpus_name')}")
                st.markdown(f"**File:** {chunk.get('filename', 'Unknown')}")
                st.markdown(f"**Chunk Index:** {chunk.get('chunk_index', 'N/A')}")
            with col2:
                score = chunk.get("score", 0)
                st.metric("Similarity Score", f"{score:.4f}")

            st.text_area(
                "Chunk Text",
                value=chunk.get("chunk_text", ""),
                height=150,
                disabled=True,
                key=f"debug_chunk_{i}",
            )
            if i < len(chunks) - 1:
                st.divider()


# ---------------------------------------------------------------------------
# Helper: Extract error message from httpx exceptions
# ---------------------------------------------------------------------------

def _extract_error(e: Exception) -> str:
    try:
        body = e.response.json() if hasattr(e, "response") else {}
        return body.get("detail", str(e))
    except Exception:
        return str(e)
