from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base # Import the Base from our models.py file

# Define the database file path. 
# 'sqlite:///./diabetes_app.db' means the file will be created in the same directory.
SQLALCHEMY_DATABASE_URL = "sqlite:///./diabetes_app.db"

# Create the SQLAlchemy engine. 
# The 'check_same_thread' argument is needed only for SQLite.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class. Each instance of a SessionLocal will be a database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This function will be used to create the database tables.
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)