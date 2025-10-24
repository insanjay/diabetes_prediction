import streamlit as st
import requests
import json # Import json for better error detail parsing

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
        chat_history_url = f"{API_BASE_URL}/api/v1/chat" # Define URL here
        chat_api_url = f"{API_BASE_URL}/api/v1/chat"     # Same URL for POST

    except KeyError:
        st.error("API_BASE_URL is not set in Streamlit secrets. Cannot connect to backend.")
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
            if not st.session_state.history_loaded:
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
                        st.session_state.history_load_error = None
                    elif response.status_code == 405:
                         st.session_state.history_load_error = f"Error loading history: Method Not Allowed (GET request failed to {chat_history_url}). Please check backend/API Gateway configuration."
                    elif response.status_code == 401:
                         st.session_state.history_load_error = "Authentication failed while loading chat history. Please log in again."
                    elif response.status_code != 404:
                         st.session_state.history_load_error = f"Could not load chat history (Error: {response.status_code})."
                    else: # Status is 404
                        st.session_state.history_loaded = True
                        st.session_state.history_load_error = None

        except requests.exceptions.Timeout:
             st.session_state.history_load_error = "Loading chat history timed out."
        except requests.exceptions.RequestException as e:
            st.session_state.history_load_error = f"Network error loading chat history: {e}"
        except Exception as e:
            st.session_state.history_load_error = f"An unexpected error occurred during chat initialization: {e}"
            st.exception(e)

        if st.session_state.history_loaded and not st.session_state.messages and not st.session_state.history_load_error:
             st.session_state.messages.append({"role": "model", "parts": ["Hello! How can I help you with your health and wellness questions today? Remember, I cannot give medical advice."]})

    # --- Display History Loading Error ---
    if not st.session_state.get("history_loaded", False) and st.session_state.get("history_load_error"):
         st.error(st.session_state.history_load_error)

    # --- 3. Display Chat History ---
    if st.session_state.get("history_loaded", False) or st.session_state.messages:
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
        st.rerun() # Rerun to display user message immediately

    # --- 5. Check if the last message needs processing ---
    needs_processing = (
        st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "model") # Process if it's the first message or follows a model response
    )

    api_call_succeeded = False # Flag to track API call outcome

    if needs_processing:
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
                    api_call_succeeded = True # Mark as success
                else:
                     error_detail = "No details provided."
                     try: error_detail = response.json().get('detail', error_detail)
                     except requests.exceptions.JSONDecodeError: error_detail = response.text
                     api_error_message = f"Error getting AI response (Status: {response.status_code}). Detail: {error_detail}"
                     st.session_state.chat_api_error = api_error_message

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

        # --- FIX: Only rerun on SUCCESS ---
        if api_call_succeeded:
            st.rerun() # Rerun AFTER successful response to update display
        # --- END FIX ---

    # --- Display persistent API error ---
    if st.session_state.get("chat_api_error"):
        st.error(st.session_state.chat_api_error)
        st.session_state.pop("chat_api_error", None) # Clear after showing

