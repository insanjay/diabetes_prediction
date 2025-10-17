import streamlit as st
import google.generativeai as genai
from database import crud, schemas

# --- HELPER FUNCTIONS ---

def get_latest_reading(db, user_id):
    """Fetches the most recent health reading for a user."""
    readings = crud.get_readings_for_user(db=db, user_id=user_id)
    if readings:
        return sorted(readings, key=lambda r: r.timestamp, reverse=True)[0]
    return None

def generate_initial_prompt(user_name, reading):
    """Creates the initial, context-aware prompt for the LLM."""
    if reading:
        return (
            f"My name is {user_name}. My age is {reading.age}, my BMI is {reading.bmi}, and "
            f"a recent prediction showed my diabetic risk as '{reading.prediction_result}' with a confidence of {reading.prediction_score*100:.0f}%. "
            "Please briefly acknowledge that you have received this information, then let me know you are ready for my questions. "
            "You must strictly avoid giving any medical advice and only provide general health and wellness information."
        )
    else:
        return (
            "You are a helpful AI assistant focused on health and wellness. You must strictly avoid giving medical advice. "
            "Please greet me and let me know you are ready for my health-related questions."
        )

# --- MAIN UI FUNCTION ---

def show_llm_chat_page(db):
    """
    Displays the LLM chat interface and handles the conversation logic.
    """
    st.header("AI Health Assistant")

    # --- 1. API Configuration ---
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-pro')
    except Exception:
        st.error("Error configuring the AI service. Please check your API key in .streamlit/secrets.toml")
        st.stop()

    # --- 2. Initialize Chat History in Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Load existing chat history from the database first
        history = crud.get_chat_history_for_user(db=db, user_id=st.session_state.user_id)
        for chat in history:
            st.session_state.messages.append({"role": "user", "parts": [chat.user_input]})
            st.session_state.messages.append({"role": "model", "parts": [chat.llm_response]})
        
        # If the session is new and there's no history, send the initial context prompt
        if not st.session_state.messages:
            with st.spinner("Initializing AI Assistant..."):
                latest_reading = get_latest_reading(db, st.session_state.user_id)
                initial_prompt = generate_initial_prompt(st.session_state.user_name, latest_reading)
                
                # Send the initial prompt to the model to get the first response
                response = model.generate_content(initial_prompt)
                
                # Add both the (hidden) initial prompt and the first response to the history
                st.session_state.messages.append({"role": "user", "parts": [initial_prompt]})
                st.session_state.messages.append({"role": "model", "parts": [response.text]})
                st.rerun() # Rerun to display the initial message immediately

    # --- 3. Save Chat Button ---
    if st.sidebar.button("Save Chat Session"):
        with st.spinner("Saving..."):
            # Get the number of messages already saved in the DB
            db_history_count = len(crud.get_chat_history_for_user(db=db, user_id=st.session_state.user_id))
            
            # Get all user/model message pairs from the session state
            session_pairs = []
            temp_user_input = None
            for msg in st.session_state.messages:
                # We skip the long initial prompt
                if "Analyze this" in msg["parts"][0] or "act as a general health assistant" in msg["parts"][0]:
                    continue
                if msg['role'] == 'user':
                    temp_user_input = msg['parts'][0]
                elif msg['role'] == 'model' and temp_user_input:
                    session_pairs.append((temp_user_input, msg['parts'][0]))
                    temp_user_input = None
            
            # Save only the new pairs that are not in the database yet
            new_pairs_to_save = session_pairs[db_history_count:]
            for user_input, llm_response in new_pairs_to_save:
                chat_to_save = schemas.ChatHistoryCreate(user_input=user_input, llm_response=llm_response)
                crud.create_chat_message(db=db, chat_data=chat_to_save, user_id=st.session_state.user_id)
            st.sidebar.success("Chat history saved!")


    # --- 4. Display Chat History ---
    for message in st.session_state.messages:
        # Don't display the initial long context prompt to the user
        if "Analyze this" in message["parts"][0] or "act as a general health assistant" in message["parts"][0]:
            continue
        
        role = "You" if message["role"] == "user" else "AI Assistant"
        with st.chat_message(role):
            st.markdown(message["parts"][0])

    # --- 5. Handle New User Input ---
    if prompt := st.chat_input("Ask a health-related question..."):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("AI is thinking..."):
                response = model.generate_content(st.session_state.messages)
                response_text = response.text

            with st.chat_message("assistant"):
                st.markdown(response_text)
            
            st.session_state.messages.append({"role": "model", "parts": [response_text]})

        except Exception as e:
            st.error(f"An error occurred while communicating with the AI: {e}")

