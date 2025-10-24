import streamlit as st
import requests  # <-- Import requests
import pandas as pd # Still needed for displaying history table
# We wrap dateutil import in a try-except to handle potential ImportError
try:
    from dateutil import parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


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
            # st.stop() # Use return instead of st.stop() in functions
            return
        token = st.session_state.token
        headers = {"Authorization": f"Bearer {token}"}
    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets. Cannot connect to backend.")
        # st.stop()
        return
    except Exception as e:
        st.error(f"Error accessing secrets or session state: {e}")
        # st.stop()
        return


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
    prediction_result_placeholder = st.empty() # Create placeholder for results

    with st.form("prediction_form"):
        st.write("Please enter the following health metrics:")
        # --- Form Inputs ---
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
             # Store form values in session state to persist them after rerun
             st.session_state.form_age = age
             st.session_state.form_height = height_cm
             st.session_state.form_weight = weight_kg
             st.session_state.form_family_history = family_history
             if user_gender == "Female":
                 st.session_state.form_pregnancies = pregnancies

             # --- Call the /predict endpoint ---
             predict_url = f"{API_BASE_URL}/api/v1/predict"
             payload = {
                 "gender": user_gender,
                 "age": float(age),
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

                         # Display results in the placeholder
                         with prediction_result_placeholder.container():
                             st.success(f"### Prediction Result: **{prediction_result_text}**")
                             st.success(f"Confidence Score: **{prediction_score_val*100:.2f}%**")
                             st.info("Your prediction has been saved to your history.")
                         # Clear form values from session state after successful submission? Optional.
                         # for key in ["form_age", "form_height", "form_weight", "form_family_history", "form_pregnancies"]:
                         #     if key in st.session_state: del st.session_state[key]

                     elif response.status_code == 400:
                          prediction_result_placeholder.error(f"Prediction failed: {response.json().get('detail', 'Invalid input.')}")
                     elif response.status_code == 401:
                          prediction_result_placeholder.error("Authentication failed. Please log out and log back in.")
                     else:
                         prediction_result_placeholder.error(f"Prediction failed. Status code: {response.status_code}")
                         try:
                             prediction_result_placeholder.error(f"Details: {response.json().get('detail', 'No details provided.')}")
                         except: pass

             except requests.exceptions.RequestException as e:
                 prediction_result_placeholder.error(f"Network error during prediction: {e}")

    # --- History Display ---
    st.header("Your Prediction History")
    st.write("Attempting to load history...") # DEBUG MESSAGE

    readings_url = f"{API_BASE_URL}/api/v1/readings"
    history_data = []
    error_loading_history = None # Track errors specifically

    try:
        with st.spinner("Loading prediction history..."):
            response = requests.get(readings_url, headers=headers)
            st.write(f"GET /readings status code: {response.status_code}") # DEBUG MESSAGE

            if response.status_code == 200:
                readings = response.json()
                st.write(f"Received {len(readings)} readings from API.") # DEBUG MESSAGE
                if readings:
                    if not DATEUTIL_AVAILABLE:
                         st.error("Dependency missing: Please add 'python-dateutil' to your root requirements.txt to display dates correctly.")
                         history_data = readings # Show raw if parsing fails
                    else:
                        try:
                            st.write("Attempting to parse and sort history...") # DEBUG MESSAGE
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
                            st.write("History processed successfully.") # DEBUG MESSAGE
                        except Exception as e:
                            st.error(f"Error processing history data: {e}")
                            st.exception(e) # Show full traceback for debugging
                            error_loading_history = f"Error processing history: {e}"
                            history_data = readings # Show raw on processing error

            elif response.status_code == 401:
                 st.error("Authentication failed while fetching history.")
                 error_loading_history = "Authentication failed"
            elif response.status_code == 404:
                 st.write("No prediction history found (404).") # Info message, not error
                 error_loading_history = None # Clear potential previous error
            else:
                 st.warning(f"Could not load prediction history (Error: {response.status_code}).")
                 error_loading_history = f"API Error {response.status_code}"

    except requests.exceptions.RequestException as e:
        st.warning(f"Network error loading prediction history: {e}")
        error_loading_history = f"Network Error: {e}"
    except Exception as e: # Catch any other unexpected errors
        st.error(f"An unexpected error occurred while loading history: {e}")
        st.exception(e)
        error_loading_history = f"Unexpected Error: {e}"


    # Display the table or appropriate message
    if history_data:
        st.write("Displaying history table...") # DEBUG MESSAGE
        try:
            df = pd.DataFrame(history_data)
            st.dataframe(df.set_index("Date")) # Set Date as index for better display
        except Exception as e:
             st.error(f"Error displaying history dataframe: {e}")
             st.write("Raw history data:", history_data) # Show raw if df fails
    elif error_loading_history:
         st.warning(f"Could not display history due to previous error: {error_loading_history}")
    else:
        # Only show this if there was no error and no data
        if response.status_code == 404 or (response.status_code == 200 and not readings) :
             st.write("You have no prediction history yet.")

