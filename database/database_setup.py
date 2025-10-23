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

# --- MODIFIED BLOCK ---
# Priority 1: Check if running in a deployed Streamlit Cloud environment
# and if the new individual database secrets are present.
if (hasattr(st, 'secrets') and
    "DB_HOST" in st.secrets and
    "DB_USER" in st.secrets and
    "DB_PASSWORD" in st.secrets):

    # Retrieve credentials from Streamlit Secrets
    host = st.secrets["DB_HOST"]
    port = st.secrets.get("DB_PORT", "5432")  # Use .get() for optional keys with a default
    dbname = st.secrets.get("DB_NAME", "postgres") # Default db name for RDS is often 'postgres'
    user = st.secrets["DB_USER"]
    password = st.secrets["DB_PASSWORD"]

    # Build the SQLAlchemy Database URL from the individual secrets
    SQLALCHEMY_DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Priority 2: Check for a single DATABASE_URL environment variable (e.g., for AWS Lambda)
elif 'DATABASE_URL' in os.environ:
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Priority 3: Fallback to local Docker database for development
else:
    db_password = os.getenv("POSTGRES_PASSWORD")
    if db_password:
        SQLALCHEMY_DATABASE_URL = f"postgresql://diabetes_app_user:{db_password}@localhost/diabetes_db"
# --- END OF MODIFIED BLOCK ---


# Proceed only if a database URL was determined
if SQLALCHEMY_DATABASE_URL:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    # If no database is configured, raise an error to stop the app
    raise ValueError("Database URL not configured. Please check your .env file or deployment secrets.")


# --- ADD THIS FUNCTION ---
# This function will be used by FastAPI to provide a database
# session to your API endpoints (this fixes the AttributeError).
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# --- END OF ADDED FUNCTION ---


def create_db_and_tables():
    """This function will be used to create the database tables."""
    if engine:
        Base.metadata.create_all(bind=engine)

