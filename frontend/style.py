import streamlit as st

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap');

        /* Global Font & Backgrounds */
        html, body, [class*="css"], .stApp {
            font-family: 'Geist', sans-serif !important;
            background-color: #0B1020 !important;
            color: #FFFFFF !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #374151 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            padding-top: 1.5rem !important;
        }

        /* Active Navigation Tab Left Border & Tint */
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            border-left: 4px solid #6366F1 !important;
            background-color: rgba(99, 102, 241, 0.1) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            text-align: left !important;
            padding-left: 12px !important;
            border-top: none !important;
            border-right: none !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }

        /* Inactive Navigation Tab */
        section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
            background-color: transparent !important;
            border: 1px solid transparent !important;
            color: #9CA3AF !important;
            border-left: 4px solid transparent !important;
            border-radius: 8px !important;
            text-align: left !important;
            padding-left: 12px !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
            background-color: #1F2937 !important;
            color: #FFFFFF !important;
        }

        /* Forms, Elevated Cards, Containers */
        div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stExpander"] {
            background-color: #1F2937 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease-in-out !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #6366F1 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1) !important;
        }
        
        div[data-testid="stExpander"] {
            margin-bottom: 1rem !important;
        }

        div[data-testid="stExpander"] details summary {
            background-color: #1F2937 !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
        }

        /* Collection Card Styling for Grid Elements */
        div.collection-card {
            background-color: #1F2937;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 1.5rem;
            height: 100%;
            transition: all 0.2s ease-in-out;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        div.collection-card:hover {
            border-color: #6366F1;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        }

        /* Buttons Styling */
        button[data-testid="baseButton-secondary"] {
            background-color: #1F2937 !important;
            border: 1px solid #374151 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            transition: all 0.2s ease-in-out !important;
        }

        button[data-testid="baseButton-secondary"]:hover {
            border-color: #6366F1 !important;
            background-color: #2D3748 !important;
        }

        button[data-testid="baseButton-primary"] {
            background-color: #6366F1 !important;
            border: 1px solid #6366F1 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
        }

        button[data-testid="baseButton-primary"]:hover {
            background-color: #4F46E5 !important;
            border-color: #4F46E5 !important;
        }

        /* Inputs & Textareas styling */
        div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
            background-color: #111827 !important;
            border: 1px solid #374151 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
        }

        div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 1px #6366F1 !important;
        }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-ready {
            background-color: rgba(16, 185, 129, 0.1);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-processing {
            background-color: rgba(245, 158, 11, 0.1);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .status-failed {
            background-color: rgba(239, 68, 68, 0.1);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Chat Input Styling */
        div[data-testid="stChatInput"] {
            border-radius: 9999px !important;
            background-color: #111827 !important;
            border: 1px solid #374151 !important;
            padding: 4px 8px !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            border: none !important;
            color: #FFFFFF !important;
        }

        /* Chat Message Styling */
        div[data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0.75rem 0 !important;
        }

        /* User Message Container right alignment */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]),
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]),
        div[data-testid="stChatMessage"]:has(img[alt="user avatar"]) {
            background-color: #1F2937 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
            max-width: 80% !important;
            margin-left: auto !important;
        }

        /* Assistant Message bubble background and padding */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]),
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]),
        div[data-testid="stChatMessage"]:has(img[alt="assistant avatar"]) {
            background-color: transparent !important;
            max-width: 100% !important;
        }

        /* Source Chips / Citation Pills */
        .source-chip {
            display: inline-flex;
            align-items: center;
            background-color: #1F2937;
            border: 1px solid #374151;
            color: #6366F1;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 500;
            margin: 4px;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
        }
        .source-chip:hover {
            border-color: #6366F1;
            background-color: rgba(99, 102, 241, 0.1);
        }

        /* Force horizontal layout for citation columns inside chat messages */
        div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 8px !important;
        }

        /* Empty States */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 1.5rem;
            text-align: center;
            border: 1px dashed #374151;
            border-radius: 8px;
            background-color: #111827;
            margin: 1.5rem 0;
        }
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 1rem;
            color: #4B5563;
        }
        .empty-state-title {
            font-size: 18px;
            font-weight: 500;
            color: #F9FAFB;
            margin-bottom: 0.5rem;
        }
        .empty-state-desc {
            font-size: 14px;
            color: #9CA3AF;
            max-width: 350px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
