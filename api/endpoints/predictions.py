import os
import joblib
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Import all the necessary components
from database import crud, schemas, database_setup
from api import auth  # For securing the endpoint

# --- Pydantic Models for Request/Response ---

# This model defines the data we EXPECT from the frontend form.
# It is now based on your ui/form_and_output.py file.
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

# --- Model Loading ---

# We load the models ONCE when the app starts.
# This is much more efficient than loading them on every request.
# The `lambda_package` must include the `models/` folder.
MODEL_PATH = "models/" # This path is relative to the root of the zip
MALE_MODEL_PATH = os.path.join(MODEL_PATH, "male_model/diabetes_stacking_ensemble_model.pkl")
MALE_ENCODER_PATH = os.path.join(MODEL_PATH, "male_model/diabetes_label_encoder_final.pkl")
FEMALE_MODEL_PATH = os.path.join(MODEL_PATH, "female_model/female_final_ensemble_model.pkl")
FEMALE_SCALER_PATH = os.path.join(MODEL_PATH, "female_model/female_feature_scaler.pkl")

try:
    male_model = joblib.load(MALE_MODEL_PATH)
    male_encoder = joblib.load(MALE_ENCODER_PATH)
    female_model = joblib.load(FEMALE_MODEL_PATH)
    female_scaler = joblib.load(FEMALE_SCALER_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model files not found. Check paths: {MODEL_PATH}")
    male_model, male_encoder, female_model, female_scaler = None, None, None, None
except Exception as e:
    print(f"Error loading models: {e}")
    male_model, male_encoder, female_model, female_scaler = None, None, None, None


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

# --- NEW: Prediction Endpoint ---
@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def run_prediction(
    request_data: PredictionRequest, 
    db: Session = Depends(database_setup.get_db),
    current_user: schemas.UserCreate = Depends(auth.get_current_user)
):
    """
    Endpoint to run a diabetes prediction.
    1. Requires a valid token.
    2. Calculates BMI.
    3. Loads the correct model based on gender.
    4. Preprocesses the data.
    5. Runs the prediction.
    6. Saves the reading to the database.
    7. Returns the result.
    """
    if not all([male_model, male_encoder, female_model, female_scaler]):
        raise HTTPException(status_code=503, detail="Prediction models are not loaded.")

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
            # Ensure columns exist before selection
            if not all(f in input_df.columns for f in male_features):
                raise HTTPException(status_code=400, detail="Missing required features for male model.")
            
            input_df_male = input_df[male_features]
            
            prediction = male_model.predict(input_df_male)
            probability = male_model.predict_proba(input_df_male)
            
            # Replicate exact frontend logic
            prediction_result = male_encoder.inverse_transform(prediction)[0] + 't Diabetic'
            prediction_score = float(probability[0][prediction[0]])

        elif request_data.gender.lower() == 'female':
            # Map family_diabetes to DiabetesPedigreeFunction
            input_df['diabetespedigreefunction'] = input_df['family_diabetes']

            # Rename columns to match the female model's expected input
            input_df = input_df.rename(columns={
                'pregnancies': 'Pregnancies',
                'bmi': 'BMI',
                'diabetespedigreefunction': 'DiabetesPedigreeFunction',
                'age': 'Age'
            })

            female_features = ["Pregnancies", "BMI", "DiabetesPedigreeFunction", "Age"]
            # Ensure columns exist before selection
            if not all(f in input_df.columns for f in female_features):
                raise HTTPException(status_code=400, detail="Missing required features for female model.")

            input_df_female = input_df[female_features]
            
            scaled_features = female_scaler.transform(input_df_female)
            
            prediction = female_model.predict(scaled_features)
            probability = female_model.predict_proba(scaled_features)
            
            # Replicate exact frontend logic
            prediction_result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
            prediction_score = float(probability[0][prediction[0]])
        
        else:
            raise HTTPException(status_code=400, detail="Invalid gender specified.")

        # 4. Save the reading to the database
        # We use the schema and function names from your frontend code for consistency
        reading_to_save = schemas.HealthReadingCreate(
            age=request_data.age,
            bmi=bmi,
            family_diabetes=request_data.family_history,
            Pregnancies=request_data.pregnancies if request_data.gender.lower() == 'female' else None,
            prediction_result=prediction_result,
            prediction_score=prediction_score
        )
        # Assuming the function name from your UI code is correct
        crud.create_health_reading(db=db, reading=reading_to_save, user_id=current_user.id)

        # 5. Return the result to the frontend
        return PredictionResponse(
            prediction_result=prediction_result,
            prediction_score=prediction_score
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Be careful not to leak sensitive error details
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

