import streamlit as st
from frontend.api_client import api_client


def show_corpora_page():
    st.title("📚 Corpora")

    # --- Create corpus form ---
    with st.expander("➕ Create New Corpus", expanded=False):
        with st.form("create_corpus_form"):
            name = st.text_input("Corpus Name")
            description = st.text_area("Description (optional)", height=80)
            submitted = st.form_submit_button("Create Corpus", use_container_width=True)
            if submitted:
                if not name.strip():
                    st.error("Corpus name is required.")
                else:
                    try:
                        api_client.create_corpus(name=name.strip(), description=description.strip())
                        st.success(f"Corpus '{name.strip()}' created!")
                        st.rerun()
                    except Exception as e:
                        st.error(_extract_error(e))

    st.divider()

    # --- List corpora ---
    try:
        corpora = api_client.list_corpora()
    except Exception as e:
        st.error(f"Failed to load corpora: {_extract_error(e)}")
        return

    if not corpora:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">📚</div>
                <div class="empty-state-title">No corpora yet</div>
                <div class="empty-state-desc">Create your first document collection using the expander above to start uploading and analyzing knowledge files.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Render in a 3-column grid
    cols_per_row = 3
    for i in range(0, len(corpora), cols_per_row):
        row_corpora = corpora[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, corpus in enumerate(row_corpora):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"### 📁 {corpus['name']}")
                    
                    doc_count = corpus.get('doc_count', 0)
                    st.markdown(f"<span style='color: #6366F1; font-weight: 600; font-size: 14px;'>{doc_count} documents</span>", unsafe_allow_html=True)
                    
                    if corpus.get("description"):
                        st.markdown(f"<div style='font-size: 13px; color: #9CA3AF; min-height: 40px; margin-top: 0.5rem; margin-bottom: 0.5rem;'>{corpus['description']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size: 13px; color: #6B7280; font-style: italic; min-height: 40px; margin-top: 0.5rem; margin-bottom: 0.5rem;'>No description.</div>", unsafe_allow_html=True)
                    
                    # Format created date if present
                    created_val = corpus.get("created_at")
                    date_str = ""
                    if isinstance(created_val, str):
                        try:
                            date_str = created_val.split("T")[0]
                        except Exception:
                            date_str = created_val
                    elif created_val:
                        date_str = created_val.strftime("%Y-%m-%d")
                    
                    if date_str:
                        st.markdown(f"<div style='font-size: 11px; color: #4B5563;'>📅 Created: {date_str}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                    
                    col_open, col_del = st.columns([4, 1.2])
                    with col_open:
                        if st.button("Open", key=f"open_{corpus['id']}", use_container_width=True, type="primary"):
                            st.session_state["page"] = "documents"
                            st.session_state["corpus_id"] = corpus["id"]
                            st.session_state["corpus_name"] = corpus["name"]
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_{corpus['id']}", use_container_width=True):
                            try:
                                api_client.delete_corpus(corpus["id"])
                                st.success(f"Corpus '{corpus['name']}' deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(_extract_error(e))


def _extract_error(e: Exception) -> str:
    try:
        body = e.response.json() if hasattr(e, "response") else {}
        return body.get("detail", str(e))
    except Exception:
        return str(e)
