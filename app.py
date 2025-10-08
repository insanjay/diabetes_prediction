import streamlit as st
import pandas as pd
import joblib
# from datetime import timedelta # for future review

# --- Import all our custom modules ---
# Schemas and Authentication
from database import schemas
import auth
# Database components
from database.database_setup import SessionLocal, create_db_and_tables
from database import crud

# --- 1. INITIALIZATION & SETUP ---

# Set the title and layout for the browser tab
st.set_page_config(page_title="Diabetic Risk Check", layout="centered")

# This line creates the database and tables if they don't exist yet.
# It's safe to run this every time the app starts.
create_db_and_tables()

# Get a database session that will be used for all database operations
db = SessionLocal()

# --- 2. HELPER FUNCTIONS (for cleanliness) ---

@st.cache_resource
def load_models_and_preprocessors():
    """
    Loads all models and preprocessors from disk once at startup.
    The cache ensures this function only runs one time.
    """
    # Load male model components
    with open("models/male_model/diabetes_stacking_ensemble_model.pkl", "rb") as f:
        male_model = joblib.load(f)
    with open("models/male_model/diabetes_label_encoder_final.pkl", "rb") as f:
        male_encoder = joblib.load(f)
    
    # Load female model components
    with open("models/female_model/female_final_ensemble_model.pkl", "rb") as f:
        female_model = joblib.load(f)
    with open("models/female_model/female_feature_scaler.pkl", "rb") as f:
        female_scaler = joblib.load(f)
        
    return male_model, male_encoder, female_model, female_scaler

def get_prediction(gender: str, input_data: dict):
    """
    Selects the correct model, preprocesses data, and returns the prediction.
    """
    input_df = pd.DataFrame([input_data])

    input_df.columns = [col.lower() for col in input_df.columns]
    male_model, male_encoder, female_model, female_scaler = load_models_and_preprocessors()

    if gender == "Male":
        male_features = ["age", "bmi", "family_diabetes"]
        input_df_male = input_df[male_features]
        
        prediction = male_model.predict(input_df_male)
        probability = male_model.predict_proba(input_df_male)
        
        prediction_text = male_encoder.inverse_transform(prediction)[0] + 't Diabetic'
        prediction_score = probability[0][prediction[0]]

    elif gender == "Female":
        # As we decided, use the Family_diabetes value for DiabetesPedigreeFunction
        input_df['DiabetesPedigreeFunction'] = input_df['family_diabetes']

            # Convert to proper case for female model
        input_df = input_df.rename(columns={
            'pregnancies': 'Pregnancies',
            'bmi': 'BMI',
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
        return "Error", 0.0

    return prediction_text, prediction_score

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculates BMI from height (cm) and weight (kg)."""
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    return 0.0

# --- 3. SESSION STATE MANAGEMENT ---

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = ""
    st.session_state.user_email = ""
    st.session_state.user_gender = ""
    st.session_state.user_id = 0

# --- 4. UI SECTIONS (Login vs. Main App) ---

# --- AUTHENTICATION UI (if not logged in) ---
if not st.session_state.logged_in:
    st.title("Welcome to the Diabetic Risk Checker")
    st.write("Please log in or register to continue.")

    choice = st.selectbox("Choose an action", ["Login", "Register"])

    if choice == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user = crud.get_user_by_email(db, email=email)
                if user and auth.verify_password(password, user.hashed_password):
                    st.session_state.logged_in = True
                    st.session_state.user_name = user.name
                    st.session_state.user_email = user.email
                    st.session_state.user_gender = user.gender
                    st.session_state.user_id = user.id
                    st.rerun()
                else:
                    st.error("Incorrect email or password.")
    
    else: # Register
        with st.form("register_form"):
            new_name = st.text_input("Name")
            new_email = st.text_input("Email")
            new_gender = st.selectbox("Gender", ["Male", "Female"])
            new_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Register")

            if submitted:
                user = crud.get_user_by_email(db, email=new_email)
                if user:
                    st.error("Email already registered.")
                else:
                    user_data = schemas.UserCreate(name=new_name, email=new_email, password=new_password, gender=new_gender)
                    hashed_password = auth.get_password_hash(new_password)
                    crud.create_user(db=db, user=user_data, hashed_password=hashed_password)
                    st.success("Account created successfully! Please proceed to the Login tab.")

# --- MAIN APPLICATION UI (if logged in) ---
else:
    st.sidebar.title(f"Welcome, {st.session_state.user_name}")
    if st.sidebar.button("Logout"):
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
        
        pregnancies = 0 # Default value for males
        if st.session_state.user_gender == "Female":
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            bmi = calculate_bmi(height_cm, weight_kg)
            st.info(f"Your calculated BMI is: **{bmi}**")

            model_input = {
                "Age": age,
                "BMI": bmi,
                "Family_diabetes": family_history,
                "Pregnancies": pregnancies
            }

            prediction_result_text, prediction_score_val = get_prediction(
                gender=st.session_state.user_gender,
                input_data=model_input
            )
            
            st.success(f"### Prediction Result: **{prediction_result_text}**")
            st.success(f"Confidence Score: **{prediction_score_val*100:.2f}%**")

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

    # --- History Display ---
    st.header("Your Prediction History")
    readings = crud.get_readings_for_user(db, user_id=st.session_state.user_id)
    
    if readings:
        history_data = [
            {
                "Date": r.timestamp.strftime("%d-%m-%Y %H:%M"),
                "age": r.age,
                "bmi": r.bmi,
                "Prediction": r.prediction_result,
                "Score": f"{r.prediction_score*100:.2f}%"
            }
            for r in sorted(readings, key=lambda item: item.timestamp, reverse=True) # Show most recent first
        ]
        st.table(history_data)
    else:
        st.write("You have no prediction history yet.")

# Always close the database session at the end of the script run
db.close()
