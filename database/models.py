from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
import pytz
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

# At the top, define India timezone
india_tz = pytz.timezone('Asia/Kolkata')

# Create a base class for our declarative class definitions
Base = declarative_base()

# Define the User table as a Python class
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=False, index=False, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    gender = Column(String, nullable=False) # 'Male' or 'Female'
    created_at = Column(DateTime, default=lambda: datetime.now(india_tz))


    # This creates a relationship to the HealthReading table
    readings = relationship("HealthReading", back_populates="owner")

    # --- THIS IS THE FIX ---
    # This adds the missing relationship to the ChatHistory table.
    chats = relationship("ChatHistory", back_populates="owner")

# Define the HealthReading table as a Python class
class HealthReading(Base):
    __tablename__ = "health_readings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(india_tz))
    
    # Model features - common to both
    age = Column(Integer)
    bmi = Column(Float)
    
    family_diabetes = Column(Integer, nullable=False)

    # Female-specific features (will be NULL for male users)
    Pregnancies = Column(Integer, nullable=True)

    # Prediction results
    prediction_result = Column(String, nullable=False)
    prediction_score = Column(Float, nullable=False)

    # This creates a relationship back to the User table
    owner = relationship("User", back_populates="readings")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(india_tz))
    user_input = Column(Text, nullable=False)
    llm_response = Column(Text, nullable=False)

    # This creates the relationship back to the User table
    owner = relationship("User", back_populates="chats")

