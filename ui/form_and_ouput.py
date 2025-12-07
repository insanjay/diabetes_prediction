import streamlit as st
import pandas as pd
import numpy as np  # Added numpy for the array handling in logic
from database import schemas, crud

# Note: We need to pass the loaded models into the main function.
# This avoids reloading them on every interaction.

# --- HELPER FUNCTIONS ---

def get_prediction(gender: str, input_data: dict, models: tuple):
    """
    Selects the correct model, preprocesses data, and returns the prediction.
    Now includes Hybrid Guardrails for Male predictions.
    """
    male_model, male_encoder, female_model, female_scaler = models
    input_df = pd.DataFrame([input_data])

    # Ensure column names are lowercase for consistency
    input_df.columns = [col.lower() for col in input_df.columns]

    if gender == "Male":
        male_features = ["age", "bmi", "family_diabetes"]
        input_df_male = input_df[male_features]
        
        # --- 1. Get the Raw Model Probability ---
        # We need the probability of class 1 (Diabetes)
        raw_probability = male_model.predict_proba(input_df_male)[0][1]
        
        # --- 2. Apply Medical Guardrails (The Fix) ---
        # Extract values for logic check
        age_val = input_data['age']
        bmi_val = input_data['bmi']
        fam_val = input_data['family_diabetes']
        
        penalty = 0.0
        
        # RULE A: BMI Penalty (Add 1.5% risk for every point over 25)
        if bmi_val > 25:
            excess_bmi = bmi_val - 25
            penalty += (excess_bmi * 0.015)
            
        # RULE B: Age Penalty (Add 0.5% risk for every year over 50)
        if age_val > 50:
            excess_age = age_val - 50
            penalty += (excess_age * 0.005)
            
        # RULE C: Family History Penalty
        if fam_val == 1:
            penalty += 0.10

        # --- 3. Calculate Final Adjusted Score ---
        final_prob = raw_probability + penalty
        
        # Cap at 99% to avoid overflow
        final_prob = min(final_prob, 0.99)
        
        # --- 4. Determine Output based on Threshold ---
        # Using the optimized threshold of 0.4
        threshold = 0.4
        
        if final_prob >= threshold:
            prediction_text = "Diabetic"
        else:
            prediction_text = "Not Diabetic"
            
        prediction_score = final_prob

    elif gender == "Female":
        # As per the logic, map family_diabetes to DiabetesPedigreeFunction
        input_df['diabetespedigreefunction'] = input_df['family_diabetes']

        # Rename columns to match the female model's expected input
        input_df = input_df.rename(columns={
            'pregnancies': 'Pregnancies',
            'bmi': 'BMI',
            'diabetespedigreefunction': 'DiabetesPedigreeFunction',
            'age': 'Age'
        })

        female_features = ["Pregnancies", "BMI", "DiabetesPedigreeFunction", "Age"]
        input_df_female = input_df[female_features]
        
        scaled_features = female_scaler.transform(input_df_female)
        
        prediction = female_model.predict(scaled_features)
        probability = female_model.predict_proba(scaled_features)
        
        prediction_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        prediction_score = probability[0][prediction[0]]
    
    else:
        # Should not happen if gender is always 'Male' or 'Female'
        return "Error: Invalid gender", 0.0

    return prediction_text, prediction_score

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculates BMI from height (cm) and weight (kg)."""
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    return 0.0

# --- MAIN UI FUNCTION ---

def show_main_dashboard(db, models):
    """
    Displays the main prediction form and user history dashboard.
    """
    st.sidebar.title(f"Welcome, {st.session_state.user_name}")
    if st.sidebar.button("Logout"):
        # Clear all session state keys to log out
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.title("Diabetic Risk Prediction Dashboard")

    # --- Prediction Form ---
    st.header("Make a New Prediction")
    with st.form("prediction_form"):
        st.write("Please enter the following health metrics:")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        height_cm = st.number_input("Height (in cm)", min_value=50.0, max_value=250.0, step=0.5)
        weight_kg = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=0.1)
        family_history = st.selectbox("Family history of diabetes?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Default value for males; shown for females
        pregnancies = 0
        if st.session_state.user_gender == "Female":
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            bmi = calculate_bmi(height_cm, weight_kg)
            if bmi > 0:
                st.info(f"Your calculated BMI is: **{bmi}**")

                model_input = {
                    "age": age,
                    "bmi": bmi,
                    "family_diabetes": family_history,
                    "pregnancies": pregnancies
                }

                prediction_result_text, prediction_score_val = get_prediction(
                    gender=st.session_state.user_gender,
                    input_data=model_input,
                    models=models
                )
                
                # Show colored prediction result
                if prediction_result_text.lower() == "diabetic":
                    st.error(f"### Prediction Result: **{prediction_result_text}**")
                else:
                    st.success(f"### Prediction Result: **{prediction_result_text}**")

                # Confidence score stays green (success)
                st.info(f"Estimated Diabetes Risk: **{prediction_score_val*100:.2f}%**")

                # Save the reading to the database
                reading_data = schemas.HealthReadingCreate(
                    age=age,
                    bmi=bmi,
                    family_diabetes=family_history,
                    Pregnancies=pregnancies if st.session_state.user_gender == "Female" else None,
                    prediction_result=prediction_result_text,
                    prediction_score=prediction_score_val
                )
                crud.create_health_reading(db=db, reading=reading_data, user_id=st.session_state.user_id)
                st.info("Your prediction has been saved to your history.")
            else:
                st.error("Please enter valid height and weight.")

    # --- History Display ---
    st.header("Your Prediction History")
    readings = crud.get_readings_for_user(db, user_id=st.session_state.user_id)
    
    if readings:
        # Sort readings by timestamp, most recent first
        sorted_readings = sorted(readings, key=lambda r: r.timestamp, reverse=True)
        history_data = [
            {
                "Date": r.timestamp.strftime("%d-%m-%Y %H:%M"),
                "Age": r.age,
                "BMI": r.bmi,
                "Prediction": r.prediction_result,
                "Score": f"{r.prediction_score*100:.2f}%"
            }
            for r in sorted_readings
        ]
        st.table(history_data)
    else:
        st.write("You have no prediction history yet.")
