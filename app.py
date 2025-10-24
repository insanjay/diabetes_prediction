import streamlit as st
import requests # Need requests for initial check maybe? Or maybe not needed here.

# --- Remove all database imports ---
# from database.database_setup import SessionLocal, create_db_and_tables # REMOVED

# --- Import only the UI modules ---
from ui.login_register import show_login_register_forms
from ui.form_and_ouput import show_main_dashboard # Corrected filename if needed
from ui.LLM_chat import show_llm_chat_page # Corrected filename if needed

# --- 1. INITIALIZATION & SETUP ---

st.set_page_config(page_title="Diabetic Risk Check", layout="centered")

# --- REMOVED Database setup ---
# create_db_and_tables() # REMOVED
# db = SessionLocal() # REMOVED

# --- REMOVED Model Loading ---
# Models are now loaded by the backend API.
# The @st.cache_resource function and calls are removed.

# --- 2. SESSION STATE & UI ROUTING ---

# Initialize session state if it doesn't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Main App Logic ---
if not st.session_state.logged_in:
    # Show login/register page (no longer needs 'db')
    show_login_register_forms()
else:
    # If logged in, show the sidebar navigation and main app
    st.sidebar.title(f"Welcome, {st.session_state.get('user_name', '')}")

    page = st.sidebar.radio("Navigation", ["Prediction Dashboard", "AI Health Chat"])

    if page == "Prediction Dashboard":
        # Show dashboard (no longer needs 'db' or 'models')
        show_main_dashboard()

    elif page == "AI Health Chat":
        # Show chat page (no longer needs 'db')
        show_llm_chat_page()

    # Logout button - Ensure key is unique if used elsewhere
    if st.sidebar.button("Logout", key="app_logout_button"):
        # Clear all session state keys to log out
        for key in list(st.session_state.keys()): # Use list() to avoid RuntimeError
            del st.session_state[key]
        st.rerun()
