from sqlalchemy.orm import Session
from . import models, schemas
from api import auth  # This import is correct

# --- User Functions ---

def get_user_by_email(db: Session, email: str):
    """
    Retrieve a single user from the database by their email address.
    """
    return db.query(models.User).filter(models.User.email == email).first()

# --- NEW FUNCTION ---
def get_user_by_id(db: Session, user_id: int):
    """
    Retrieve a single user by their ID.
    This was missing and caused an error in auth.py.
    """
    return db.query(models.User).filter(models.User.id == user_id).first()
# --- END NEW FUNCTION ---

def create_user(db: Session, user: schemas.UserCreate):
    """
    Create a new user and save them to the database.
    FIX: This function now handles password hashing.
    """
    # Hash the password from the schema
    hashed_password = auth.get_password_hash(user.password)
    
    # Create the database model object
    db_user = models.User(
        name=user.name,
        email=user.email,
        gender=user.gender,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Health Reading Functions ---

def create_health_reading(db: Session, reading: schemas.HealthReadingCreate, user_id: int):
    """
    Create a new health reading record associated with a user and save it.
    """
    reading_data = reading.model_dump()
    db_reading = models.HealthReading(**reading_data, user_id=user_id)
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    return db_reading

def get_readings_for_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    """
    Retrieve all health readings for a specific user.
    """
    # Sort by timestamp descending to get the newest first
    return db.query(models.HealthReading)\
        .filter(models.HealthReading.user_id == user_id)\
        .order_by(models.HealthReading.timestamp.desc())\
        .offset(skip).limit(limit).all()

# --- Chat History Functions ---

def create_chat_message(db: Session, chat_data: schemas.ChatHistoryCreate, user_id: int):
    """
    Create a new chat history record associated with a user and save it.
    FIX: Changed to accept ChatHistoryCreate schema.
    """
    db_chat = models.ChatHistory(
        user_id=user_id,
        user_input=chat_data.user_input,
        llm_response=chat_data.llm_response
    )
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_chat_history_for_user(db: Session, user_id: int):
    """
    Retrieve all chat history for a specific user, ordered by timestamp.
    """
    return db.query(models.ChatHistory)\
        .filter(models.ChatHistory.user_id == user_id)\
        .order_by(models.ChatHistory.timestamp).all()
