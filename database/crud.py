# diabetes_project/database/crud.py

from sqlalchemy.orm import Session
from . import models
from . import schemas # Note the .. to go up one directory level
import auth 

def get_user_by_email(db: Session, email: str):
    """
    Retrieve a single user from the database by their email address.
    """
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate, hashed_password: str):
    """
    Create a new user and save them to the database.
    """
    db_user = models.User(
        name=user.name,
        email=user.email, 
        hashed_password=hashed_password, 
        gender=user.gender
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_health_reading(db: Session, reading: schemas.HealthReadingCreate, user_id: int):
    """
    Create a new health reading record associated with a user and save it.
    """
    # Create a dictionary from the Pydantic model and add the user_id
    # .model_dump() is the Pydantic v2 equivalent of .dict()
    reading_data = reading.model_dump() 
    reading_data['user_id'] = user_id
    
    db_reading = models.HealthReading(**reading_data)
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    return db_reading

def get_readings_for_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    """
    Retrieve all health readings for a specific user.
    """
    return db.query(models.HealthReading).filter(models.HealthReading.user_id == user_id).offset(skip).limit(limit).all()



# To work on this later

# def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate):
#     """
#     Update a user's details, such as name or password.
#     """
#     db_user = db.query(models.User).filter(models.User.id == user_id).first()
#     if not db_user:
#         return None

#     # Update name if provided
#     if user_update.name is not None:
#         db_user.name = user_update.name

#     # Update password if provided
#     if user_update.password is not None:
#         # We will use a function from auth.py to get the new hash
#         new_hashed_password = auth.get_password_hash(user_update.password)
#         db_user.hashed_password = new_hashed_password
    
#     db.commit()
#     db.refresh(db_user)
#     return db_user

def create_chat_message(db: Session, chat_data: schemas.ChatHistory, user_id: int):
    """
    Create a new chat history record associated with a user and save it.
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
    return db.query(models.ChatHistory).filter(models.ChatHistory.user_id==user_id).order_by(models.ChatHistory.timestamp).all()