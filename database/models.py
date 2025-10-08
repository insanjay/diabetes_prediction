from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
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

# Define the HealthReading table as a Python class
class HealthReading(Base):
    __tablename__ = "health_readings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(india_tz))
    # 
    # Model features - common to both
    age = Column(Integer)
    bmi = Column(Float)
    
    family_diabetes = Column(Integer, nullable=False)

    # Female-specific features (will be NULL for male users)
    Pregnancies = Column(Integer, nullable=True)
    # DiabetesPedigreeFunction = Column(Float, nullable=True) kept this so that in future i may know that the female model and model doesn't have this feature name common

    # Prediction results
    prediction_result = Column(String, nullable=False)
    prediction_score = Column(Float, nullable=False)

    # This creates a relationship back to the User table
    owner = relationship("User", back_populates="readings")