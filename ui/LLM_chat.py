import streamlit as st
import requests
import json

# --- CONSTANTS ---
INITIAL_PROMPT_KEY_PHRASES = [
    "My name is",
    "my BMI is",
    "You are a helpful AI assistant"
]

# --- HELPER FUNCTIONS ---

def is_initial_prompt(message_content):
    """Checks if a message is part of the initial, hidden context-setting prompt."""
    if not isinstance(message_content, str):
        return False
    return any(phrase in message_content for phrase in INITIAL_PROMPT_KEY_PHRASES)

# --- MAIN UI FUNCTION (REFACTORED) ---

def show_llm_chat_page():
    """
    Displays the LLM chat interface and handles conversation logic
    by calling the backend API.
    """
    st.header("AI Health Assistant")

    # --- 1. Get API URL and Token ---
    try:
        API_BASE_URL = st.secrets["API_BASE_URL"]
        if 'token' not in st.session_state:
            st.error("Authentication token not found. Please log in again.")
            return
        token = st.session_state.token
        headers = {"Authorization": f"Bearer {token}"}
        chat_history_url = f"{API_BASE_URL}/api/v1/chat"
        chat_api_url = f"{API_BASE_URL}/api/v1/chat"

    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets.")
        return
    except Exception as e:
        st.error(f"Error accessing secrets or session state: {e}")
        return

    # --- 2. Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.history_loaded = False
        st.session_state.history_load_error = None

        try:
            # Only attempt to load if not already loaded and no previous error persists wrongly
            if not st.session_state.history_loaded:
                # Clear any potentially stale error before attempting load
                st.session_state.history_load_error = None
                st.write(f"DEBUG: Attempting to GET history from: {chat_history_url}") # DEBUG
                with st.spinner("Loading chat history..."):
                    response = requests.get(chat_history_url, headers=headers, timeout=15)
                    st.write(f"DEBUG: GET /chat status code: {response.status_code}") # DEBUG
                    if response.status_code == 200:
                        history = response.json()
                        for chat in history:
                            user_input = chat.get("user_input")
                            llm_response = chat.get("llm_response")
                            if user_input is not None:
                                 st.session_state.messages.append({"role": "user", "parts": [user_input]})
                            if llm_response is not None:
                                 st.session_state.messages.append({"role": "model", "parts": [llm_response]})
                        st.session_state.history_loaded = True
                        st.session_state.history_load_error = None # Explicitly clear error on success
                    elif response.status_code == 405:
                         st.session_state.history_load_error = f"Error loading history: Method Not Allowed (GET failed to {chat_history_url}). Check backend/API Gateway."
                    elif response.status_code == 401:
                         st.session_state.history_load_error = "Authentication failed loading history. Please log in again."
                    elif response.status_code == 404: # No history found is okay
                        st.session_state.history_loaded = True
                        st.session_state.history_load_error = None # Explicitly clear error
                    else: # Other errors
                         st.session_state.history_load_error = f"Could not load chat history (Error: {response.status_code})."

        except requests.exceptions.Timeout:
             st.session_state.history_load_error = "Loading chat history timed out."
        except requests.exceptions.RequestException as e:
            st.session_state.history_load_error = f"Network error loading chat history: {e}"
        except Exception as e:
            st.session_state.history_load_error = f"Unexpected error during chat init: {e}"
            st.exception(e)

        # Add welcome message only if loading was attempted, succeeded (even if empty), and no messages exist
        if st.session_state.history_loaded and not st.session_state.messages and not st.session_state.history_load_error:
             st.session_state.messages.append({"role": "model", "parts": ["Hello! How can I help you with your health and wellness questions today? Remember, I cannot give medical advice."]})

    # --- Display History Loading Error ---
    # Show error if loading failed
    if st.session_state.get("history_load_error"):
         st.error(st.session_state.history_load_error)

    # --- 3. Display Chat History ---
    # Display messages regardless of loading status (allows current session chat)
    for message in st.session_state.messages[:]:
        if message["parts"] and message["parts"][0] and is_initial_prompt(message["parts"][0]):
            continue
        role_display_name = "You" if message["role"] == "user" else "AI Assistant"
        with st.chat_message(role_display_name):
             content = message["parts"][0] if message["parts"] and message["parts"][0] else "*message error*"
             st.markdown(content)

    # --- 4. Handle New User Input ---
    if prompt := st.chat_input("Ask a health-related question..."):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        # Set a flag indicating processing is needed
        st.session_state.needs_processing = True
        st.rerun() # Rerun to display user message immediately

    # --- 5. Process User Input if Flag is Set ---
    if st.session_state.get("needs_processing", False):
        # Reset the flag
        st.session_state.needs_processing = False

        last_user_prompt = st.session_state.messages[-1]["parts"][0]
        payload = {"prompt": last_user_prompt}
        api_error_message = None

        try:
            with st.spinner("AI is thinking..."):
                st.write(f"DEBUG: POSTing to {chat_api_url} with payload: {json.dumps(payload)}") # DEBUG
                response = requests.post(chat_api_url, headers=headers, json=payload, timeout=60)
                st.write(f"DEBUG: POST /chat status code: {response.status_code}") # DEBUG

                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get("response", "Error: No response text found.")
                    st.session_state.messages.append({"role": "model", "parts": [response_text]})
                    st.session_state.pop("chat_api_error", None)
                    # --- FIX: NO RERUN HERE ---
                    # Let Streamlit update the chat display automatically
                else:
                     error_detail = "No details provided."
                     try: error_detail = response.json().get('detail', error_detail)
                     except requests.exceptions.JSONDecodeError: error_detail = response.text
                     api_error_message = f"Error getting AI response (Status: {response.status_code}). Detail: {error_detail}"
                     st.session_state.chat_api_error = api_error_message
                     # Do not pop user message

        except requests.exceptions.Timeout:
            api_error_message = "The AI service timed out. Please try again."
            st.session_state.chat_api_error = api_error_message
        except requests.exceptions.RequestException as e:
            api_error_message = f"Network error communicating with the AI: {e}"
            st.session_state.chat_api_error = api_error_message
        except Exception as e:
             api_error_message = f"An unexpected error occurred: {e}"
             st.session_state.chat_api_error = api_error_message
             st.exception(e)

        # --- FIX: NO RERUN HERE EITHER ---
        # If there was an error, it will be displayed below without a rerun

    # --- Display persistent API error ---
    if st.session_state.get("chat_api_error"):
        st.error(st.session_state.chat_api_error)
        st.session_state.pop("chat_api_error", None) # Clear after showing

