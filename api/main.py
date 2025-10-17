from fastapi import FastAPI
from .endpoints import users

# Create the main FastAPI application instance
app = FastAPI(
    title="Diabetic Risk Assessment API",
    description="An API to manage users, health predictions, and chat history.",
    version="1.0.0",
)

# Include the router from the users.py file.
# All endpoints defined in that file will now be part of the main app.
# The prefix ensures that all user-related routes will start with /api/v1
# e.g., http://localhost:8000/api/v1/users/
app.include_router(users.router, prefix="/api/v1", tags=["Users"])

# A simple root endpoint to confirm the API is running
@app.get("/")
def read_root():
    """
    Root endpoint to provide a welcome message.
    """
    return {"message": "Welcome to the Diabetes Prediction API"}

