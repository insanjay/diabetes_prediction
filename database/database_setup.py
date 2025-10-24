import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# --- DECOUPLED CONNECTION LOGIC ---
# This "pure" backend logic ONLY reads from environment variables.
# AWS Lambda will provide these variables.
# Your local .env file will provide them for local testing.
# Streamlit/st.secrets logic has been completely removed.

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if DB_HOST and DB_USER and DB_PASSWORD:
    # Build the SQLAlchemy Database URL
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

else:
    # If no database is configured, raise an error to stop the app
    raise ValueError("Database credentials (DB_HOST, DB_USER, DB_PASSWORD) not found in environment variables.")


# This is the dependency that our API endpoints (auth.py, etc.) will use
def get_db():
    """
    Dependency to get a new database session per request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

