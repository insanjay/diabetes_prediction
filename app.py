import streamlit as st
import joblib

# Database components
from database.database_setup import SessionLocal, create_db_and_tables

# --- Import all our new UI modules ---
from ui.login_register import show_login_register_forms
from ui.form_and_ouput import show_main_dashboard
from ui.LLM_chat import show_llm_chat_page

# --- 1. INITIALIZATION & SETUP ---

st.set_page_config(page_title="Diabetic Risk Check", layout="centered")

# This is a one-time setup to create the database tables if they don't exist.
create_db_and_tables()

# Create a single database session when the script starts.
db = SessionLocal()

# --- 2. MODEL LOADING ---

@st.cache_resource
def load_models_and_preprocessors():
    """
    Loads all models and preprocessors from disk once at startup.
    """
    try:
        # Load male model components
        with open("models/male_model/diabetes_stacking_ensemble_model.pkl", "rb") as f:
            male_model = joblib.load(f)
        with open("models/male_model/diabetes_label_encoder_final.pkl", "rb") as f:
            male_encoder = joblib.load(f)
        
        # Load female model components
        with open("models/female_model/female_final_ensemble_model.pkl", "rb") as f:
            female_model = joblib.load(f)
        with open("models/female_model/female_feature_scaler.pkl", "rb") as f:
            female_scaler = joblib.load(f)
            
        return male_model, male_encoder, female_model, female_scaler
    except FileNotFoundError:
        st.error("One or more model files were not found. Please ensure the 'models' directory is correct.")
        return None, None, None, None

models = load_models_and_preprocessors()

# --- 3. SESSION STATE & UI ROUTING ---

# Initialize session state if it doesn't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Main App Logic ---
if not st.session_state.logged_in:
    # If not logged in, show the login/register page
    show_login_register_forms(db)
else:
    # If logged in, show the sidebar navigation and main app
    st.sidebar.title(f"Welcome, {st.session_state.get('user_name', '')}")
    
    page = st.sidebar.radio("Navigation", ["Prediction Dashboard", "AI Health Chat"])
    
    if page == "Prediction Dashboard":
        if models[0]: # Check if models loaded successfully
            show_main_dashboard(db, models)
    
    elif page == "AI Health Chat":
        show_llm_chat_page(db)

    # --- THIS IS THE FIX ---
    # We add a unique key to the logout button to prevent the duplicate ID error.
    if st.sidebar.button("Logout", key="main_logout_button"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

