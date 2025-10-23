from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from database import crud, schemas, database_setup
from api import auth

router = APIRouter()

# --- User Registration Endpoint ---
@router.post("/users", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_new_user(user: schemas.UserCreate, db: Session = Depends(database_setup.get_db)):
    """
    Endpoint to create a new user.
    FIX 1: Changed response_model to schemas.User (Hides password).
    FIX 2: Simplified logic to pass the user schema directly to crud.
           This fixes the 'gender' ValidationError.
    """
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Pass the user schema directly to crud.
    # The crud function will handle hashing and saving.
    return crud.create_user(db=db, user=user)

# --- User Login Endpoint (for getting a token) ---
@router.post("/token", response_model=schemas.Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(database_setup.get_db)
):
    """
    Endpoint to handle user login.
    It authenticates the user and returns a JWT access token.
    FIX: Added response_model=schemas.Token for clarity.
    """
    user = auth.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": str(user.id)},  # 'sub' is the user ID
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

