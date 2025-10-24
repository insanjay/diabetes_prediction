import streamlit as st
import requests  # <-- Import requests
import pandas as pd # Still needed for displaying history table
from dateutil import parser # Need 'python-dateutil' in root requirements.txt

# --- MAIN UI FUNCTION ---

def show_main_dashboard(): # Removed db and models parameters
    """
    Displays the main prediction form and user history dashboard, using API calls.
    """
    # --- 1. Get API URL and Token ---
    try:
        API_BASE_URL = st.secrets["API_BASE_URL"]
        # Ensure the user is logged in and has a token
        if 'token' not in st.session_state or not st.session_state.get('logged_in'):
            st.error("You must be logged in to view the dashboard.")
            st.stop()
        token = st.session_state.token
        headers = {"Authorization": f"Bearer {token}"}
    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets. Cannot connect to backend.")
        st.stop()
    except Exception as e:
        st.error(f"Error accessing secrets or session state: {e}")
        st.stop()


    # --- Sidebar ---
    # Display user info loaded during login
    st.sidebar.title(f"Welcome, {st.session_state.get('user_name', 'User')}")

    # Logout button needs a unique key if it appears elsewhere
    if st.sidebar.button("Logout", key="dashboard_logout_button"):
        # Clear all session state keys to log out
        for key in list(st.session_state.keys()): # Use list() to avoid RuntimeError
            del st.session_state[key]
        st.rerun()

    st.title("Diabetic Risk Prediction Dashboard")

    # --- Prediction Form ---
    st.header("Make a New Prediction")
    with st.form("prediction_form"):
        st.write("Please enter the following health metrics:")
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30) # Add default value
        height_cm = st.number_input("Height (in cm)", min_value=50.0, max_value=250.0, step=0.5, value=170.0)
        weight_kg = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=0.1, value=70.0)
        family_history = st.selectbox("Family history of diabetes?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        pregnancies = 0
        user_gender = st.session_state.get('user_gender', 'Male') # Get gender from session state
        if user_gender == "Female":
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, value=0)

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            # --- NEW: Call the /predict endpoint ---
            predict_url = f"{API_BASE_URL}/api/v1/predict"
            # The API expects all fields defined in PredictionRequest
            payload = {
                "gender": user_gender,
                "age": float(age), # Ensure type consistency with Pydantic model
                "height_cm": float(height_cm),
                "weight_kg": float(weight_kg),
                "family_history": int(family_history),
                "pregnancies": int(pregnancies)
            }

            try:
                with st.spinner("Calculating prediction..."):
                    response = requests.post(predict_url, headers=headers, json=payload)

                    if response.status_code == 200:
                        prediction_data = response.json()
                        prediction_result_text = prediction_data.get("prediction_result")
                        prediction_score_val = prediction_data.get("prediction_score")

                        st.success(f"### Prediction Result: **{prediction_result_text}**")
                        st.success(f"Confidence Score: **{prediction_score_val*100:.2f}%**")
                        st.info("Your prediction has been saved to your history.")
                        # --- THIS IS THE FIX ---
                        # REMOVED st.rerun() which was causing the message to vanish
                        # --- END OF FIX ---

                    elif response.status_code == 400: # Bad Request (e.g., invalid input)
                         st.error(f"Prediction failed: {response.json().get('detail', 'Invalid input.')}")
                    elif response.status_code == 401: # Unauthorized
                         st.error("Authentication failed. Please log out and log back in.")
                    else:
                        st.error(f"Prediction failed. Status code: {response.status_code}")
                        try:
                            st.error(f"Details: {response.json().get('detail', 'No details provided.')}")
                        except: pass

            except requests.exceptions.RequestException as e:
                st.error(f"Network error during prediction: {e}")

    # --- History Display ---
    st.header("Your Prediction History")

    # --- NEW: Fetch history from the backend ---
    readings_url = f"{API_BASE_URL}/api/v1/readings"
    history_data = [] # Default to empty list

    try:
        with st.spinner("Loading prediction history..."):
            response = requests.get(readings_url, headers=headers)

            if response.status_code == 200:
                readings = response.json()
                if readings: # Check if readings is not empty
                    try:
                        # Sort readings using parsed timestamps
                        sorted_readings = sorted(readings, key=lambda r: parser.parse(r.get('timestamp', '1970-01-01T00:00:00Z')), reverse=True)

                        history_data = [
                            {
                                "Date": parser.parse(r.get('timestamp', '')).strftime("%d-%m-%Y %H:%M") if r.get('timestamp') else "N/A",
                                "Age": r.get("age", "N/A"),
                                "BMI": r.get("bmi", "N/A"),
                                "Prediction": r.get("prediction_result", "N/A"),
                                "Score": f"{r.get('prediction_score', 0)*100:.2f}%" if r.get('prediction_score') is not None else "N/A"
                            }
                            for r in sorted_readings
                        ]
                    except ImportError:
                        st.error("Dependency missing: Please add 'python-dateutil' to your requirements.txt to display dates correctly.")
                        history_data = readings # Show raw if parsing fails
                    except Exception as e:
                        st.error(f"Error processing history data: {e}")
                        history_data = readings # Show raw on other errors

            elif response.status_code == 401:
                 st.error("Authentication failed while fetching history.")
            # Do not warn on 404, just means no history yet.
            elif response.status_code != 404:
                 st.warning(f"Could not load prediction history (Error: {response.status_code}).")

    except requests.exceptions.RequestException as e:
        st.warning(f"Network error loading prediction history: {e}")

    # Display the table
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df.set_index("Date")) # Set Date as index for better display
    else:
        st.write("You have no prediction history yet.")

