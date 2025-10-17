import streamlit as st
from database import schemas, crud
import auth

def show_login_register_forms(db):
    """
    Displays the login and registration forms.
    Returns True if login is successful, False otherwise.
    """
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
                    # Set session state on successful login
                    st.session_state.logged_in = True
                    st.session_state.user_name = user.name
                    st.session_state.user_email = user.email
                    st.session_state.user_gender = user.gender
                    st.session_state.user_id = user.id
                    st.rerun() # Rerun the app to show the main dashboard
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
                    # Note: We don't use the password from the schema directly for hashing
                    user_data = schemas.UserCreate(
                        name=new_name, 
                        email=new_email, 
                        password=new_password, 
                        gender=new_gender
                    )
                    hashed_password = auth.get_password_hash(new_password)
                    crud.create_user(db=db, user=user_data, hashed_password=hashed_password)
                    st.success("Account created successfully! Please proceed to the Login tab.")

    return False # Return False by default if login is not yet successful
