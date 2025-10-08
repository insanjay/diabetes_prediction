# diabetes_project/auth.py

from datetime import datetime, timezone, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext

# --- Configuration ---

# This creates a context for hashing passwords using the bcrypt algorithm.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# These should be kept secret and ideally loaded from an environment file.
# For now, we will define them here.
SECRET_KEY = "a_very_secret_key_change_this" # Replace with a real, random secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # A user's session will last 30 minutes

# --- Functions ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain password against a stored hash.
    Returns True if they match, False otherwise.
    """

    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hashes a plain password.
    Returns the hashed password as a string.
    """
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a new JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt