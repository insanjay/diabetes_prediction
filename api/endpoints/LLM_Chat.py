import os
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from typing import List

# Import all the necessary components
from database import crud, schemas, database_setup
from api import auth  # We can now import this directly since it's in the api/ folder

# model name
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

@lru_cache(maxsize=1)
def get_hf_client():
    """
    Loads the HF client once and caches it.
    This is called by the endpoint, not at init.
    """
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        # This will cause a clean 500 error if the secret is missing
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    return InferenceClient(token=HF_TOKEN)



# --- Pydantic Models for Request/Response ---
# This defines what the frontend must send to us
class ChatRequest(BaseModel):
    prompt: str

# This defines what we will send back to the frontend
class ChatResponse(BaseModel):
    response: str

# Initialize the router
router = APIRouter()

# --- NEW: Chat Endpoint ---
@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def handle_chat_message(
    chat_request: ChatRequest, 
    db: Session = Depends(database_setup.get_db),
    current_user: schemas.user = Depends(auth.get_current_user),
    client: InferenceClient = Depends(get_hf_client)
):
    """
    Endpoint to handle a user's chat message.
    1. Requires a valid token (gets current_user).
    2. Calls the Hugging Face LLM.
    3. Saves the conversation to the database.
    4. Returns the LLM's response.
    """
    if not client:
        raise HTTPException(status_code=503, detail="AI service is not available")

    user_prompt = chat_request.prompt
    
    try:
        # 1. Call the Hugging Face LLM
        # We can build a more complex history later, for now, just send the prompt
        response = client.text_generation(
            prompt=user_prompt,
            model=MODEL_NAME,
            max_new_tokens=250
        )
        llm_response_text = response.strip()

        # 2. Save the conversation to the database
        chat_to_save = schemas.ChatHistoryCreate(
            user_input=user_prompt, 
            llm_response=llm_response_text
        )
        crud.create_chat_message(db=db, chat_data=chat_to_save, user_id=current_user.id)

        # 3. Return the LLM's response
        return ChatResponse(response=llm_response_text)

    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the AI response.")

@router.get("/chat", response_model=List[schemas.ChatHistory])
def get_chat_history(
    db: Session = Depends(database_setup.get_db),
    current_user: schemas.User = Depends(auth.get_current_user)
    ):  
    """
    Endpoint to retrieve the chat history for the authenticated user.
    """
    history = crud.get_chat_history_for_user(db=db, user_id=current_user.id)
    if not history:
        # Return an empty list, not a 404, if there's no history
        return []
    return history