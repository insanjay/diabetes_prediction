from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# --- User Schemas ---
class UserBase(BaseModel):
    email: str
    name: str
    gender: str

# Model for CREATING a user (what we expect from the request)
class UserCreate(UserBase):
    password: str

# Model for RESPONDING with a user (what we send back)
# This is the 'User' model that was missing.
# Notice it does NOT include the password.
class User(UserBase):
    id: int

    class Config:
        from_attributes = True  # Pydantic v2 syntax

# --- Health Reading Schemas ---
class HealthReadingBase(BaseModel):
    age: float
    bmi: float
    family_diabetes: int
    Pregnancies: Optional[float] = None
    prediction_result: str
    prediction_score: float

class HealthReadingCreate(HealthReadingBase):
    pass

class HealthReading(HealthReadingBase):
    id: int
    user_id: int  # <-- FIX: Changed from owner_id to match models.py
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Chat History Schemas ---
class ChatHistoryBase(BaseModel):
    user_input: str
    llm_response: str

class ChatHistoryCreate(ChatHistoryBase):
    pass

class ChatHistory(ChatHistoryBase):
    id: int
    user_id: int  # <-- FIX: Changed from owner_id to match models.py
    timestamp: datetime

    class Config:
        from_attributes = True

