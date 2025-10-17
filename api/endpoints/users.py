from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# Note the relative imports to go up one directory level
from ..dependencies import get_db
# Import from the top-level directories
from database import crud, schemas, models
import auth

# Create a new router object. 
# We can use this to group all user-related endpoints.
router = APIRouter()

# Define the response model for reading a user.
# We don't want to expose the hashed_password.
class User(schemas.UserCreate):
    id: int
    created_at: models.datetime

    class Config:
        orm_mode = True

@router.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    API endpoint to register a new user.
    It receives user data, validates it using the UserCreate schema,
    and saves the new user to the database.
    """
    # Check if a user with this email already exists
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Email already registered"
        )
    
    # Hash the password before saving
    hashed_password = auth.get_password_hash(user.password)
    
    # Use the existing CRUD function to create the user
    created_user = crud.create_user(db=db, user=user, hashed_password=hashed_password)
    return created_user
