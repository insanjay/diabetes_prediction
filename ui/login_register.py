import streamlit as st
import requests  # <-- Make sure 'requests' is in your root requirements.txt

# Removed database and auth imports

def show_login_register_forms(): # Removed db parameter
    """
    Displays the login and registration forms, using API calls.
    """
    st.title("Welcome to the Diabetic Risk Checker")
    st.write("Please log in or register to continue.")

    # Get the API URL from secrets (MUST be set in Streamlit Cloud)
    try:
        API_BASE_URL = st.secrets["API_BASE_URL"]
    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets. Cannot connect to backend.")
        st.stop() # Stop the app if the URL is missing

    choice = st.radio("Choose an action", ["Login", "Register"])

    if choice == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                token_url = f"{API_BASE_URL}/api/v1/token"
                login_data = {"username": email, "password": password} # OAuth2 uses 'username'

                try:
                    response = requests.post(token_url, data=login_data)

                    if response.status_code == 200:
                        token_data = response.json()
                        token = token_data["access_token"]

                        # --- NEW: Call /users/me to get user details ---
                        headers = {"Authorization": f"Bearer {token}"}
                        me_url = f"{API_BASE_URL}/api/v1/users/me"
                        user_response = requests.get(me_url, headers=headers)

                        if user_response.status_code == 200:
                            user_data = user_response.json()
                            # --- Store ALL user info in session state ---
                            st.session_state.token = token # Store the token
                            st.session_state.logged_in = True
                            st.session_state.user_name = user_data.get("name")
                            st.session_state.user_email = user_data.get("email")
                            st.session_state.user_gender = user_data.get("gender")
                            st.session_state.user_id = user_data.get("id")
                            st.success("Login successful!")
                            st.rerun() # Rerun to show the main dashboard
                        else:
                            st.error(f"Login succeeded, but failed to retrieve user details (Error: {user_response.status_code}). Please try again.")

                    elif response.status_code == 401:
                         st.error("Incorrect email or password.")
                    else:
                        st.error(f"Login failed. Status code: {response.status_code}")
                        try:
                            st.error(f"Details: {response.json().get('detail', 'No details provided.')}")
                        except: pass

                except requests.exceptions.RequestException as e:
                    st.error(f"Network error during login: {e}")


    else: # Register
        with st.form("register_form"):
            new_name = st.text_input("Name")
            new_email = st.text_input("Email")
            new_gender = st.radio("Gender", ["Male", "Female"])
            new_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Register")

            if submitted:
                register_url = f"{API_BASE_URL}/api/v1/users"
                user_data_to_send = {
                    "name": new_name,
                    "email": new_email,
                    "gender": new_gender,
                    "password": new_password
                }

                try:
                    response = requests.post(register_url, json=user_data_to_send)

                    if response.status_code == 201: # 201 Created
                        st.success("Account created successfully! Please proceed to the Login tab.")
                    elif response.status_code == 400: # Bad Request (e.g., email exists)
                        st.error(response.json().get("detail", "Registration failed (e.g., email already exists)."))
                    else:
                        st.error(f"Registration failed. Status code: {response.status_code}")
                        try:
                            st.error(f"Details: {response.json().get('detail', 'No details provided.')}")
                        except: pass

                except requests.exceptions.RequestException as e:
                    st.error(f"Network error during registration: {e}")

    # No return value needed, session state handles control flow

