import os
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
from database import crud, schemas, database_setup
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer

# --- Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_default_key_for_local_dev")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# FIX: The tokenUrl must match the full path from the root
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/token")

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- User Authentication ---
def authenticate_user(db: Session, email: str, password: str):
    """
    Finds a user by email and verifies their password.
    """
    user = crud.get_user_by_email(db, email=email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# --- JWT Token Creation ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Token Verification (for secured endpoints) ---
def get_current_user(
    db: Session = Depends(database_setup.get_db), 
    token: str = Depends(oauth2_scheme)
):
    """
    A dependency that decodes the token, validates it, 
    and returns the user. This will be used to protect
    all other endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # This function was missing from crud.py, but is now added.
        user = crud.get_user_by_id(db, user_id=int(user_id))
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

