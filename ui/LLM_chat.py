import streamlit as st
from huggingface_hub import InferenceClient
from database import crud, schemas

# --- CONSTANTS ---
# Define the model you want to use from Hugging Face
# Make sure it's a model compatible with the chat completions API
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Define the strings used in the initial prompt to make filtering more reliable
INITIAL_PROMPT_KEY_PHRASES = [
    "My name is",
    "my BMI is",
    "You are a helpful AI assistant"
]


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

def format_messages_for_api(messages):
    """Converts Streamlit's message format to the format required by the HF API."""
    formatted = []
    for msg in messages:
        # The API expects "assistant" for the model's role, not "model"
        role = "assistant" if msg["role"] == "model" else msg["role"]
        # The API expects a "content" key, not "parts"
        content = msg["parts"][0]
        formatted.append({"role": role, "content": content})
    return formatted

def is_initial_prompt(message_content):
    """Checks if a message is part of the initial, hidden context-setting prompt."""
    return any(phrase in message_content for phrase in INITIAL_PROMPT_KEY_PHRASES)


# --- MAIN UI FUNCTION (REFACTORED) ---

def show_llm_chat_page(db):
    """
    Displays the LLM chat interface and handles the conversation logic
    using the Hugging Face InferenceClient.
    """
    st.header("AI Health Assistant")

    # --- 1. API Configuration ---
    try:
        HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    except Exception:
        st.error("Error configuring the AI service. Please add your HUGGINGFACE_TOKEN to Streamlit's secrets.")
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

                # **CHANGED**: Call the new HF API for the first message
                completion = client.chat_completion(
                    messages=[{"role": "user", "content": initial_prompt}],
                    max_tokens=250, # Limit the initial response length
                    stream=False, # We don't need streaming for the initial message
                )
                initial_response = completion.choices[0].message.content

                # Add both the (hidden) initial prompt and the first response to the history
                st.session_state.messages.append({"role": "user", "parts": [initial_prompt]})
                st.session_state.messages.append({"role": "model", "parts": [initial_response]})
                st.rerun() # Rerun to display the initial message immediately

    # --- 3. Save Chat Button ---
    if st.sidebar.button("Save Chat Session"):
        with st.spinner("Saving..."):
            # Get the number of message pairs already saved in the DB
            db_history_count = len(crud.get_chat_history_for_user(db=db, user_id=st.session_state.user_id))
            
            # Extract user/model message pairs from the session state, skipping the initial prompt
            session_pairs = []
            temp_user_input = None
            for msg in st.session_state.messages:
                if is_initial_prompt(msg["parts"][0]):
                    continue # Skip the hidden prompt
                
                if msg['role'] == 'user':
                    temp_user_input = msg['parts'][0]
                elif msg['role'] == 'model' and temp_user_input is not None:
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
        if is_initial_prompt(message["parts"][0]):
            continue
        
        # Display role as "You" or "AI Assistant"
        role_display_name = "You" if message["role"] == "user" else "AI Assistant"
        with st.chat_message(role_display_name):
            st.markdown(message["parts"][0])

    # --- 5. Handle New User Input ---
    if prompt := st.chat_input("Ask a health-related question..."):
        # Add user's new message to state and display it
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("You"):
            st.markdown(prompt)

        # Get response from the AI
        try:
            with st.spinner("AI is thinking..."):
                # **CHANGED**: Format messages and call the new HF API
                formatted_api_messages = format_messages_for_api(st.session_state.messages)
                completion = client.chat_completion(
                    messages=formatted_api_messages,
                    max_tokens=500,
                    stream=False,
                )
                response_text = completion.choices[0].message.content

            # Display AI's response
            with st.chat_message("AI Assistant"):
                st.markdown(response_text)
            
            # Add AI's response to the session state
            st.session_state.messages.append({"role": "model", "parts": [response_text]})

        except Exception as e:
            st.error(f"An error occurred while communicating with the AI: {e}")
