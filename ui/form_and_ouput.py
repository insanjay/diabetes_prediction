import os
import joblib
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import pandas as pd
from typing import List                     # <-- Added
from sqlalchemy.orm import Session         # <-- Added
from fastapi import Depends                # <-- Already exists, ensure it's there
from database import database_setup, crud, schemas  # <-- Added/Consolidated
from api import auth                       # <-- Added/Consolidated
from functools import lru_cache            # <-- Added for lazy loading


# --- Pydantic Models for Request/Response ---

# This model defines the data we EXPECT from the frontend form.
class PredictionRequest(BaseModel):
    gender: str
    age: float
    height_cm: float
    weight_kg: float
    family_history: int  # 0 or 1
    pregnancies: int     # 0 for males, >= 0 for females

# This model defines what we will send BACK to the frontend.
class PredictionResponse(BaseModel):
    prediction_result: str
    prediction_score: float


# --- Model Loading (Lazy Loading) ---

@lru_cache(maxsize=1)
def load_prediction_models():
    """
    Loads all models and preprocessors from disk once when first called
    and caches the result. Handles potential errors during loading.
    """
    MODEL_PATH = "models/" # Relative to project root, should be correct in container
    MALE_MODEL_PATH = os.path.join(MODEL_PATH, "male_model/diabetes_stacking_ensemble_model.pkl")
    MALE_ENCODER_PATH = os.path.join(MODEL_PATH, "male_model/diabetes_label_encoder_final.pkl")
    FEMALE_MODEL_PATH = os.path.join(MODEL_PATH, "female_model/female_final_ensemble_model.pkl")
    FEMALE_SCALER_PATH = os.path.join(MODEL_PATH, "female_model/female_feature_scaler.pkl")

    try:
        print("Attempting to load models...") # Add logging
        male_model = joblib.load(MALE_MODEL_PATH)
        male_encoder = joblib.load(MALE_ENCODER_PATH)
        female_model = joblib.load(FEMALE_MODEL_PATH)
        female_scaler = joblib.load(FEMALE_SCALER_PATH)
        print("Models loaded successfully.")
        return male_model, male_encoder, female_model, female_scaler
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Model file not found: {e}. Check paths relative to Docker context: {MODEL_PATH}")
        # Raise an exception to prevent the app from starting incorrectly
        raise RuntimeError(f"Model file not found: {e}") from e
    except Exception as e:
        print(f"FATAL ERROR loading models: {e}")
        raise RuntimeError(f"Error loading models: {e}") from e

# Initialize the router
router = APIRouter()

# --- Helper Function for BMI ---
def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculates BMI from height (cm) and weight (kg)."""
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    return 0.0

# --- Prediction Endpoint ---
@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def run_prediction(
    request_data: PredictionRequest,
    db: Session = Depends(database_setup.get_db),
    current_user: schemas.User = Depends(auth.get_current_user) # Correct schema
):
    """
    Endpoint to run a diabetes prediction.
    """
    # --- Trigger model loading on first request ---
    try:
        male_model, male_encoder, female_model, female_scaler = load_prediction_models()
    except RuntimeError as e:
         # If models failed to load at startup (or on first call), return error
         print(f"Prediction endpoint cannot proceed: {e}")
         raise HTTPException(status_code=503, detail=f"Prediction models are not available: {e}")


    try:
        # 1. Calculate BMI from raw inputs
        bmi = calculate_bmi(request_data.height_cm, request_data.weight_kg)
        if bmi == 0.0:
            raise HTTPException(status_code=400, detail="Invalid height or weight provided.")

        # 2. Create the base DataFrame
        model_input_dict = {
            "age": request_data.age,
            "bmi": bmi,
            "family_diabetes": request_data.family_history,
            "pregnancies": request_data.pregnancies
        }
        input_df = pd.DataFrame([model_input_dict])
        input_df.columns = [col.lower() for col in input_df.columns]

        # 3. Select model and preprocess based on gender
        if request_data.gender.lower() == 'male':
            male_features = ["age", "bmi", "family_diabetes"]
            if not all(f in input_df.columns for f in male_features):
                raise HTTPException(status_code=400, detail="Missing required features for male model.")

            input_df_male = input_df[male_features]

            prediction = male_model.predict(input_df_male)
            probability = male_model.predict_proba(input_df_male)

            prediction_result = male_encoder.inverse_transform(prediction)[0] + 't Diabetic'
            prediction_score = float(probability[0][prediction[0]]) # Ensure float

        elif request_data.gender.lower() == 'female':
            input_df['diabetespedigreefunction'] = input_df['family_diabetes']
            input_df = input_df.rename(columns={
                'pregnancies': 'Pregnancies',
                'bmi': 'BMI',
                'diabetespedigreefunction': 'DiabetesPedigreeFunction',
                'age': 'Age'
            })

            female_features = ["Pregnancies", "BMI", "DiabetesPedigreeFunction", "Age"]
            if not all(f in input_df.columns for f in female_features):
                raise HTTPException(status_code=400, detail="Missing required features for female model.")

            input_df_female = input_df[female_features]

            scaled_features = female_scaler.transform(input_df_female)

            prediction = female_model.predict(scaled_features)
            probability = female_model.predict_proba(scaled_features)

            prediction_result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
            prediction_score = float(probability[0][prediction[0]]) # Ensure float

        else:
            raise HTTPException(status_code=400, detail="Invalid gender specified.")

        # 4. Save the reading to the database
        reading_to_save = schemas.HealthReadingCreate(
            age=int(request_data.age), # Ensure int for DB if needed
            bmi=bmi,
            family_diabetes=request_data.family_history,
            Pregnancies=int(request_data.pregnancies) if request_data.gender.lower() == 'female' else None, # Ensure int
            prediction_result=prediction_result,
            prediction_score=prediction_score
        )
        crud.create_health_reading(db=db, reading=reading_to_save, user_id=current_user.id)

        # 5. Return the result to the frontend
        return PredictionResponse(
            prediction_result=prediction_result,
            prediction_score=prediction_score
        )

    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI/validation errors
    except Exception as e:
        print(f"Error during prediction: {type(e).__name__} - {e}") # Log error type
        # Be careful not to leak sensitive error details in production
        raise HTTPException(status_code=500, detail=f"An internal error occurred during prediction.")


# --- Get Readings History Endpoint ---
@router.get("/readings", response_model=List[schemas.HealthReading])
def get_health_readings(
    db: Session = Depends(database_setup.get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
    skip: int = 0, # Optional query parameters for pagination
    limit: int = 100
):
    """
    Endpoint to retrieve the health reading history for the authenticated user.
    """
    readings = crud.get_readings_for_user(db=db, user_id=current_user.id, skip=skip, limit=limit)
    if not readings:
        return []
    # Convert SQLAlchemy models to Pydantic models (or ensure your schema allows it)
    # This might require adjusting your HealthReading schema or how crud returns data
    return readings # Assuming crud function returns objects compatible with the schema

