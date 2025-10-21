import os
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# --- SMART CONNECTION LOGIC ---
# This block determines which database to connect to.

SQLALCHEMY_DATABASE_URL = None

# Check if running in a deployed Streamlit Cloud environment
if hasattr(st, 'secrets') and "database" in st.secrets:
    SQLALCHEMY_DATABASE_URL = st.secrets["database"]["url"]
# Check if running in a deployed AWS Lambda environment
elif 'DATABASE_URL' in os.environ:
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
else:
    # Fallback to local Docker database for development
    db_password = os.getenv("POSTGRES_PASSWORD")
    if db_password:
        SQLALCHEMY_DATABASE_URL = f"postgresql://diabetes_app_user:{db_password}@localhost/diabetes_db"

# Proceed only if a database URL was determined
if SQLALCHEMY_DATABASE_URL:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    # If no database is configured, raise an error to stop the app
    raise ValueError("Database URL not configured. Please check your .env file or deployment secrets.")


def create_db_and_tables():
    """This function will be used to create the database tables."""
    if engine:
        Base.metadata.create_all(bind=engine)