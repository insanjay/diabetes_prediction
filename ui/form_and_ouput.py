import streamlit as st
import requests
import pandas as pd
try:
    from dateutil import parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

# --- MAIN UI FUNCTION ---

def show_main_dashboard():
    """
    Displays the main prediction form and user history dashboard, using API calls.
    """
    # --- 1. Get API URL and Token ---
    try:
        API_BASE_URL = st.secrets["API_BASE_URL"]
        if 'token' not in st.session_state or not st.session_state.get('logged_in'):
            st.error("You must be logged in to view the dashboard.")
            return
        token = st.session_state.token
        headers = {"Authorization": f"Bearer {token}"}
    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets.")
        return
    except Exception as e:
        st.error(f"Error accessing secrets or session state: {e}")
        return

    # --- Sidebar ---
    st.sidebar.title(f"Welcome, {st.session_state.get('user_name', 'User')}")
    if st.sidebar.button("Logout", key="dashboard_logout_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.title("Diabetic Risk Prediction Dashboard")

    # --- Prediction Form ---
    st.header("Make a New Prediction")

    with st.form("prediction_form"):
        st.write("Please enter the following health metrics:")
        # --- Form Inputs (using session state to preserve values) ---
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=st.session_state.get("form_age", 30))
        height_cm = st.number_input("Height (in cm)", min_value=50.0, max_value=250.0, step=0.5, value=st.session_state.get("form_height", 170.0))
        weight_kg = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=0.1, value=st.session_state.get("form_weight", 70.0))
        family_history = st.selectbox("Family history of diabetes?", [0, 1], index=st.session_state.get("form_family_history", 0), format_func=lambda x: "No" if x == 0 else "Yes")

        pregnancies = 0
        user_gender = st.session_state.get('user_gender', 'Male')
        if user_gender == "Female":
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, value=st.session_state.get("form_pregnancies", 0))

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
             # Store form values in session state
             st.session_state.form_age = age
             st.session_state.form_height = height_cm
             st.session_state.form_weight = weight_kg
             st.session_state.form_family_history = family_history
             if user_gender == "Female":
                 st.session_state.form_pregnancies = pregnancies

             # Clear previous results/errors before making new call
             st.session_state.pop("prediction_error", None)
             st.session_state.pop("last_prediction_result", None)
             st.session_state.pop("last_prediction_score", None)
             st.session_state.pop("last_prediction_message", None)


             # --- Call the /predict endpoint ---
             predict_url = f"{API_BASE_URL}/api/v1/predict"
             payload = {
                 "gender": user_gender, "age": float(age), "height_cm": float(height_cm),
                 "weight_kg": float(weight_kg), "family_history": int(family_history),
                 "pregnancies": int(pregnancies)
             }

             try:
                 with st.spinner("Calculating prediction..."):
                     response = requests.post(predict_url, headers=headers, json=payload)

                     if response.status_code == 200:
                         prediction_data = response.json()
                         # --- STORE result in session state ---
                         st.session_state.last_prediction_result = prediction_data.get("prediction_result")
                         st.session_state.last_prediction_score = prediction_data.get("prediction_score")
                         st.session_state.last_prediction_message = "Your prediction has been saved." # Simplified message

                     # --- Handle errors by storing them in session state ---
                     elif response.status_code == 400:
                          st.session_state.prediction_error = f"Prediction failed: {response.json().get('detail', 'Invalid input.')}"
                     elif response.status_code == 401:
                          st.session_state.prediction_error = "Authentication error. Please log out and log back in."
                     else:
                         detail = "No details provided."
                         try: detail = response.json().get('detail', detail)
                         except: pass
                         st.session_state.prediction_error = f"Prediction failed (Status {response.status_code}). Details: {detail}"

             except requests.exceptions.RequestException as e:
                 st.session_state.prediction_error = f"Network error during prediction: {e}"

             # Trigger rerun AFTER processing API response (success or error)
             st.rerun()

    # --- FIX: Define placeholder and display results AFTER the form ---
    prediction_result_placeholder = st.empty()
    # This block runs every time, including after a rerun triggered by submission
    if st.session_state.get("prediction_error"):
        prediction_result_placeholder.error(st.session_state.prediction_error)
        # Clear error after displaying once so it doesn't persist forever
        st.session_state.pop("prediction_error", None)
    elif st.session_state.get("last_prediction_result"):
        with prediction_result_placeholder.container():
            st.success(f"### Prediction Result: **{st.session_state.last_prediction_result}**")
            st.success(f"Confidence Score: **{st.session_state.last_prediction_score*100:.2f}%**")
            st.info(st.session_state.last_prediction_message)
        # Clear result after displaying once so it doesn't persist forever
        st.session_state.pop("last_prediction_result", None)
        st.session_state.pop("last_prediction_score", None)
        st.session_state.pop("last_prediction_message", None)


    # --- History Display ---
    st.header("Your Prediction History")

    readings_url = f"{API_BASE_URL}/api/v1/readings"
    history_data = []
    error_loading_history = None
    readings = [] # Define readings outside the `if` block

    try:
        with st.spinner("Loading prediction history..."):
            response = requests.get(readings_url, headers=headers)

            if response.status_code == 200:
                readings = response.json()
                if readings:
                    if not DATEUTIL_AVAILABLE:
                         st.error("Dependency missing: Please add 'python-dateutil' to your root requirements.txt to display dates correctly.")
                         history_data = readings # Show raw if parsing fails
                    else:
                        try:
                            # Ensure timestamp is string before parsing
                            sorted_readings = sorted(
                                readings,
                                key=lambda r: parser.parse(r.get('timestamp', '1970-01-01T00:00:00Z')) if isinstance(r.get('timestamp'), str) else datetime.min.replace(tzinfo=timezone.utc),
                                reverse=True
                            )
                            history_data = [
                                {
                                    "Date": parser.parse(r.get('timestamp', '')).strftime("%d-%m-%Y %H:%M") if r.get('timestamp') and isinstance(r.get('timestamp'), str) else "N/A",
                                    "Age": r.get("age", "N/A"), "BMI": r.get("bmi", "N/A"),
                                    "Prediction": r.get("prediction_result", "N/A"),
                                    "Score": f"{r.get('prediction_score', 0)*100:.2f}%" if r.get('prediction_score') is not None else "N/A"
                                }
                                for r in sorted_readings
                            ]
                        except Exception as e:
                            st.error(f"Error processing history data: {e}")
                            st.exception(e) # Show full traceback
                            error_loading_history = f"Error processing history: {e}"
                            history_data = readings # Show raw on processing error

            elif response.status_code == 401:
                 st.error("Authentication failed while fetching history.")
                 error_loading_history = "Authentication failed"
            elif response.status_code == 404:
                 error_loading_history = None # No history is not an error
            else:
                 st.warning(f"Could not load prediction history (Error: {response.status_code}). Details: {response.text}")
                 error_loading_history = f"API Error {response.status_code}"

    except requests.exceptions.RequestException as e:
        st.warning(f"Network error loading prediction history: {e}")
        error_loading_history = f"Network Error: {e}"
    except Exception as e: # Catch-all for other errors
        st.error(f"An unexpected error occurred while loading history: {e}")
        st.exception(e)
        error_loading_history = f"Unexpected Error: {e}"

    # Display the table or appropriate message
    if history_data:
        try:
            df = pd.DataFrame(history_data)
            st.dataframe(df.set_index("Date"))
        except Exception as e:
             st.error(f"Error displaying history dataframe: {e}")
             st.write("Raw history data:", history_data) # Show raw if df fails
    elif error_loading_history:
         st.warning(f"Could not display history due to previous error: {error_loading_history}")
    else:
        # Check status code exists before accessing
        response_status_code = response.status_code if 'response' in locals() else None
        if response_status_code == 404 or (response_status_code == 200 and not readings):
             st.write("You have no prediction history yet.")

