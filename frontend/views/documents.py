import time
import streamlit as st
from frontend.api_client import api_client


def show_documents_page():
    corpus_id = st.session_state.get("corpus_id")
    corpus_name = st.session_state.get("corpus_name", "Corpus")

    if not corpus_id:
        st.warning("No corpus selected. Go back to Corpora.")
        if st.button("← Back to Corpora"):
            st.session_state["page"] = "corpora"
            st.rerun()
        return

    # --- Header with back button ---
    col_back, col_title, col_chat = st.columns([1, 5, 1])
    with col_back:
        if st.button("← Back"):
            st.session_state["page"] = "corpora"
            st.session_state.pop("corpus_id", None)
            st.session_state.pop("corpus_name", None)
            st.session_state.pop("_poll_until", None)
            st.rerun()
    with col_title:
        st.title(f"📄 {corpus_name}")
    with col_chat:
        if st.button("💬 Chat", key="docs_to_chat"):
            st.session_state["page"] = "chat"
            st.rerun()

    # --- Upload document ---
    with st.expander("📤 Upload Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
            key=f"uploader_{corpus_id}",
        )
        if uploaded_file is not None:
            if st.button("Upload", use_container_width=True):
                try:
                    with st.spinner("Uploading document..."):
                        # Write to a temp file so api_client can send it
                        import tempfile, os
                        tmp_dir = tempfile.mkdtemp()
                        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        result = api_client.upload_document(corpus_id, tmp_path)
                        # Clean up temp file
                        os.remove(tmp_path)
                        os.rmdir(tmp_dir)

                    st.success(f"📤 '{uploaded_file.name}' uploaded — processing in background.")
                    # Start polling for up to 60 seconds
                    st.session_state["_poll_until"] = time.time() + 60
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Upload failed: {_extract_error(e)}")

    st.divider()

    # --- List documents ---
    try:
        documents = api_client.list_documents(corpus_id)
    except Exception as e:
        st.error(f"Failed to load documents: {_extract_error(e)}")
        return

    if not documents:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">📄</div>
                <div class="empty-state-title">No documents uploaded</div>
                <div class="empty-state-desc">Upload PDF, DOCX, TXT, or MD documents using the upload section above to start embedding knowledge.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Check if any documents are still processing
    any_processing = any(doc["status"] in ("processing", "uploaded") for doc in documents)

    for doc in documents:
        with st.container(border=True):
            # Select file type icon
            ftype = doc["file_type"].lower()
            icon = "📄"
            if ftype == "pdf":
                icon = "📕"
            elif ftype == "docx":
                icon = "📘"
            elif ftype == "txt":
                icon = "📝"
            elif ftype == "md":
                icon = "Ⓜ️"
            
            cols = st.columns([3.5, 1.2, 1.5, 1, 0.8])
            
            # Filename
            cols[0].markdown(f"**{icon} {doc['filename']}**")
            
            # Type & Size
            cols[1].markdown(f"<span style='font-size: 13px; color: #9CA3AF;'>{ftype.upper()} ({_format_size(doc['file_size'])})</span>", unsafe_allow_html=True)
            
            # Status Badge
            status = doc["status"]
            if status == "ready":
                cols[2].markdown('<span class="status-badge status-ready">Ready</span>', unsafe_allow_html=True)
            elif status == "processing":
                stage = doc.get("processing_stage") or "processing"
                progress = doc.get("processing_progress")
                progress_str = f" ({int(progress * 100)}%)" if progress is not None else ""
                cols[2].markdown(f'<span class="status-badge status-processing">{stage.capitalize()}{progress_str}</span>', unsafe_allow_html=True)
            elif status == "failed":
                cols[2].markdown('<span class="status-badge status-failed">Failed</span>', unsafe_allow_html=True)
            else:
                cols[2].markdown(f'<span class="status-badge status-processing">{status.capitalize()}</span>', unsafe_allow_html=True)
            
            # Show processing error reason for failed documents
            if status == "failed" and doc.get("process_error"):
                st.error(f"⚠️ {doc['process_error']}")
            
            # Chunk count
            chunks = doc.get("chunk_count")
            chunks_str = f"{chunks} chunks" if chunks is not None else "— chunks"
            cols[3].markdown(f"<span style='font-size: 13px; color: #E5E7EB; font-weight: 500;'>{chunks_str}</span>", unsafe_allow_html=True)
            
            # Delete button
            if cols[4].button("🗑️", key=f"deldoc_{doc['id']}", use_container_width=True):
                try:
                    api_client.delete_document(doc["id"])
                    st.success(f"'{doc['filename']}' deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(_extract_error(e))
            
            # Preview section inside card
            if doc["status"] == "ready":
                with st.expander("🔍 Preview Content", expanded=False):
                    try:
                        preview_data = api_client.preview_document(doc["id"])
                        preview_text = preview_data.get("preview", "")
                        if preview_text:
                            st.text_area("Preview (first 500 characters)", value=preview_text[:500], height=150, disabled=True, key=f"prev_text_{doc['id']}")
                        else:
                            msg = preview_data.get("message", "No preview available.")
                            st.info(msg)
                    except Exception as e:
                        st.error(f"Failed to load preview: {_extract_error(e)}")

    # --- Auto-refresh polling for processing documents ---
    poll_until = st.session_state.get("_poll_until", 0)
    if any_processing and time.time() < poll_until:
        time.sleep(2)
        st.rerun()
    elif not any_processing and poll_until > 0:
        # All done — clear polling state
        st.session_state.pop("_poll_until", None)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _extract_error(e: Exception) -> str:
    try:
        body = e.response.json() if hasattr(e, "response") else {}
        return body.get("detail", str(e))
    except Exception:
        return str(e)
