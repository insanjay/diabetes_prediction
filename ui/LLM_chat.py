import streamlit as st
import requests  # <-- Import requests

# Note: Removed huggingface_hub, crud, schemas imports

# --- CONSTANTS ---
# Define the strings used in the initial prompt to make filtering more reliable
# We still need this to avoid displaying the hidden setup prompt if it's stored
INITIAL_PROMPT_KEY_PHRASES = [
    "My name is",
    "my BMI is",
    "You are a helpful AI assistant"
]

# --- HELPER FUNCTIONS (MODIFIED) ---

def is_initial_prompt(message_content):
    """Checks if a message is part of the initial, hidden context-setting prompt."""
    # Ensure content is a string before checking
    if not isinstance(message_content, str):
        return False
    return any(phrase in message_content for phrase in INITIAL_PROMPT_KEY_PHRASES)

# --- MAIN UI FUNCTION (REFACTORED) ---

def show_llm_chat_page(): # Removed db parameter
    """
    Displays the LLM chat interface and handles conversation logic
    by calling the backend API.
    """
    st.header("AI Health Assistant")

    # --- 1. Get API URL and Token ---
    try:
        API_BASE_URL = st.secrets["API_BASE_URL"]
        # Ensure the user is logged in and has a token
        if 'token' not in st.session_state:
            st.error("You must be logged in to use the chat.")
            st.stop()
        token = st.session_state.token
        headers = {"Authorization": f"Bearer {token}"}
    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets. Cannot connect to backend.")
        st.stop()

    # --- 2. Initialize Chat History in Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # --- NEW: Load existing chat history from the backend ---
        chat_history_url = f"{API_BASE_URL}/api/v1/chat"
        try:
            with st.spinner("Loading chat history..."):
                response = requests.get(chat_history_url, headers=headers)
                if response.status_code == 200:
                    history = response.json() # Assuming API returns list of {"user_input": ..., "llm_response": ...}
                    for chat in history:
                        st.session_state.messages.append({"role": "user", "parts": [chat.get("user_input")]})
                        st.session_state.messages.append({"role": "model", "parts": [chat.get("llm_response")]})
                elif response.status_code != 404: # Ignore 404 if no history exists
                     st.warning(f"Could not load chat history (Error: {response.status_code}).")

        except requests.exceptions.RequestException as e:
            st.warning(f"Network error loading chat history: {e}")

        # If still no messages after loading, display a welcome (no initial API call needed)
        if not st.session_state.messages:
             st.session_state.messages.append({"role": "model", "parts": ["Hello! How can I help you with your health and wellness questions today? Remember, I cannot give medical advice."]})
             # No st.rerun needed here, will display on first load

    # --- 3. Save Chat Button (REMOVED) ---
    # The backend /chat endpoint now saves automatically on each message.

    # --- 4. Display Chat History ---
    for message in st.session_state.messages:
        # Don't display the initial long context prompt if it exists
        if message["parts"] and message["parts"][0] and is_initial_prompt(message["parts"][0]):
            continue

        # Display role as "You" or "AI Assistant"
        role_display_name = "You" if message["role"] == "user" else "AI Assistant"
        with st.chat_message(role_display_name):
             # Handle potential None values if history loading failed partially
             content = message["parts"][0] if message["parts"] and message["parts"][0] else "*message not loaded*"
             st.markdown(content)


    # --- 5. Handle New User Input ---
    if prompt := st.chat_input("Ask a health-related question..."):
        # Add user's new message to state and display it
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("You"):
            st.markdown(prompt)

        # --- NEW: Get response from the backend API ---
        chat_api_url = f"{API_BASE_URL}/api/v1/chat"
        payload = {"prompt": prompt}

        try:
            with st.spinner("AI is thinking..."):
                response = requests.post(chat_api_url, headers=headers, json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get("response", "Error: No response text found.")

                    # Display AI's response
                    with st.chat_message("AI Assistant"):
                        st.markdown(response_text)

                    # Add AI's response to the session state
                    st.session_state.messages.append({"role": "model", "parts": [response_text]})

                else:
                     st.error(f"Error getting AI response (Status: {response.status_code}).")
                     try:
                         st.error(f"Details: {response.json().get('detail', 'No details provided.')}")
                     except: pass
                     # Remove the user's message if the API call failed? Optional.
                     # st.session_state.messages.pop()

        except requests.exceptions.RequestException as e:
            st.error(f"Network error communicating with the AI: {e}")
            # Remove the user's message if the API call failed? Optional.
            # st.session_state.messages.pop()
        
        # We need to rerun to ensure message display updates correctly after API call
        st.rerun()
