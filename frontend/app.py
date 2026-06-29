import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from frontend.views.login import show_login_page
from frontend.views.corpora import show_corpora_page
from frontend.views.documents import show_documents_page
from frontend.views.chat import show_chat_page

st.set_page_config(page_title="Domain Knowledge Co-Pilot", layout="wide")

from frontend.style import inject_custom_css
inject_custom_css()

if "token" not in st.session_state:
    st.session_state["token"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "corpora"

token = st.session_state["token"]

if not token:
    show_login_page()
    st.stop()

# --- Connection guard: verify backend is reachable ---
try:
    from frontend.api_client import api_client
    api_client.token = token
    api_client.health()
except Exception:
    st.error("⚠️ Cannot connect to the backend server. Please ensure it is running on http://localhost:8000")
    st.stop()

st.sidebar.title("Domain Co-Pilot")

st.sidebar.write(f"Welcome, **{st.session_state['user']['username']}**")

page = st.session_state.get("page", "corpora")

if st.sidebar.button(
    "📚 Corpora",
    type="primary" if page == "corpora" else "secondary",
    use_container_width=True
):
    st.session_state["page"] = "corpora"
    st.session_state.pop("corpus_id", None)
    st.session_state.pop("corpus_name", None)
    st.session_state.pop("chat_session_id", None)
    st.session_state.pop("chat_messages", None)
    st.session_state.pop("last_debug_chunks", None)
    st.rerun()

# Show context-sensitive buttons when a corpus is selected
if st.session_state.get("corpus_id"):
    corpus_name = st.session_state.get("corpus_name", "Corpus")
    st.sidebar.caption(f"📁 {corpus_name}")

    if st.sidebar.button(
        "📄 Documents",
        key="nav_docs",
        type="primary" if page == "documents" else "secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "documents"
        st.session_state.pop("chat_session_id", None)
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("last_debug_chunks", None)
        st.rerun()

    if st.sidebar.button(
        "💬 Chat",
        key="nav_chat",
        type="primary" if page == "chat" else "secondary",
        use_container_width=True
    ):
        st.session_state["page"] = "chat"
        st.session_state.pop("chat_session_id", None)
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("last_debug_chunks", None)
        st.rerun()

if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
    from frontend.api_client import api_client
    api_client.token = None
    st.session_state["token"] = None
    st.session_state["user"] = None
    st.session_state["page"] = "corpora"
    st.session_state.pop("corpus_id", None)
    st.session_state.pop("corpus_name", None)
    st.session_state.pop("chat_session_id", None)
    st.session_state.pop("chat_messages", None)
    st.session_state.pop("last_debug_chunks", None)
    st.rerun()

# --- Page routing ---
page = st.session_state.get("page", "corpora")

if page == "chat":
    show_chat_page()
elif page == "documents":
    show_documents_page()
else:
    show_corpora_page()
