import streamlit as st
from frontend.api_client import api_client


def show_login_page():
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0.2rem;'>Domain Knowledge Co-Pilot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #9CA3AF; margin-bottom: 1.5rem;'>Precision dark platform for document intelligence and RAG research</p>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if not email or not password:
                        st.error("Please fill in all fields")
                    else:
                        try:
                            api_client.login(email=email, password=password)
                            st.session_state["token"] = api_client.token
                            st.session_state["user"] = api_client.me()
                            st.rerun()
                        except Exception as e:
                            detail = _extract_error(e)
                            st.error(detail)

        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email")
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign Up", use_container_width=True)
                if submitted:
                    if not new_email or not new_username or not new_password:
                        st.error("Please fill in all fields")
                    else:
                        try:
                            api_client.signup(email=new_email, username=new_username, password=new_password)
                            st.success("Account created! Switch to the Login tab and sign in.")
                        except Exception as e:
                            detail = _extract_error(e)
                            st.error(detail)


def _extract_error(e: Exception) -> str:
    import json
    try:
        body = e.response.json() if hasattr(e, "response") else {}
        return body.get("detail", str(e))
    except Exception:
        return str(e)
