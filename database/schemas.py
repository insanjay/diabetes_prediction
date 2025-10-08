# diabetes_project/schemas.py

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Annotated

# pydantic model to update the user's info
class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None

# Pydantic model for creating a health reading
class HealthReadingCreate(BaseModel):
    age: Annotated[int, Field(..., gt=0, description="Age of the user")]
    bmi: Annotated[float, Field(..., gt=0, description="BMI of the user")]
    family_diabetes: Annotated[int, Field(default=0, ge=0, le=1, description="User family had diabetes- yes:1, no:0")]
    Pregnancies: Optional[int] = Field(None, ge=0, description="Number of times a user (valid for females only) has been pregnant")
    prediction_result: Annotated[str, Field(..., description="Prediction result from the model(e.g. Low risk, high risk)")]
    prediction_score: Annotated[float, Field(..., description="Confidence score of the model on certain prediction(e.g.0.86/86%)")]

# Pydantic model for creating a new user
class UserCreate(BaseModel):
    name: Annotated[str, Field(..., description="Name of the user")]
    email: Annotated[EmailStr, Field(..., description="Email address of the user(e.g. user@mail.com)")]
    password: Annotated[str, Field(..., description="User's password to login")]
    gender: Annotated[str, Field(..., description="Gender of the user(e.g. either male or female)")]