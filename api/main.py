import os
from fastapi import FastAPI
from database import database_setup, models
from .endpoints import users, LLM_Chat, predictions

# --- THIS IS THE FIX ---
from mangum import Mangum
# --- END OF FIX ---

# This command creates all the database tables if they don't exist
models.Base.metadata.create_all(bind=database_setup.engine)

# Create the main FastAPI application instance
app = FastAPI(
    title="Diabetic Risk Assessment API",
    description="An API to manage users, health predictions, and chat history.",
    version="2.0.0",
)

# --- Include all the new routers from our endpoint files ---

# Include the router from users.py (for login and registration)
app.include_router(users.router, prefix="/api/v1", tags=["Authentication"])

# Include the router from LLM_Chat.py (for the AI assistant)
app.include_router(LLM_Chat.router, prefix="/api/v1", tags=["AI Chat"])

# Include the router from predictions.py (for the health predictions)
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])


# A simple root endpoint to confirm the API is running
@app.get("/")
def read_root():
    """
    Root endpoint to provide a welcome message.
    """
    return {"message": "Welcome to the Diabetes Prediction API V2"}

stage = os.environ.get("STAGE", "default")
api_gateway_base_path = f"/{stage}"


# This creates the "handler" that Lambda understands.
# It wraps our FastAPI app in the Mangum translator.
handler = Mangum(app, api_gateway_base_path=api_gateway_base_path)
# --- END OF FIX ---

