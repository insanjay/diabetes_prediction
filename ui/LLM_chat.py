import streamlit as st
import requests

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
        try:
            with st.spinner("Loading chat history..."):
                response = requests.get(chat_history_url, headers=headers, timeout=10) # Add timeout
                if response.status_code == 200:
                    history = response.json()
                    for chat in history:
                        user_input = chat.get("user_input")
                        llm_response = chat.get("llm_response")
                        if user_input is not None:
                             st.session_state.messages.append({"role": "user", "parts": [user_input]})
                        if llm_response is not None:
                             st.session_state.messages.append({"role": "model", "parts": [llm_response]})
                # Explicitly handle 405 error if GET /chat somehow isn't allowed by backend
                elif response.status_code == 405:
                     st.error(f"Error loading history: Method Not Allowed (GET request failed). Please check backend configuration.")
                elif response.status_code == 401:
                     st.error("Authentication failed while loading chat history. Please log in again.")
                elif response.status_code != 404: # Ignore 404
                     st.warning(f"Could not load chat history (Error: {response.status_code}).")

        except requests.exceptions.Timeout:
             st.warning("Loading chat history timed out.")
        except requests.exceptions.RequestException as e:
            st.warning(f"Network error loading chat history: {e}")
        except Exception as e: # Catch any other unexpected errors during init
            st.error(f"An unexpected error occurred during chat initialization: {e}")


        # If still no messages after loading, display a welcome
        if not st.session_state.messages:
             st.session_state.messages.append({"role": "model", "parts": ["Hello! How can I help you with your health and wellness questions today? Remember, I cannot give medical advice."]})

    # --- 3. Display Chat History ---
    for message in st.session_state.messages:
        if message["parts"] and message["parts"][0] and is_initial_prompt(message["parts"][0]):
            continue

        role_display_name = "You" if message["role"] == "user" else "AI Assistant"
        with st.chat_message(role_display_name):
             content = message["parts"][0] if message["parts"] and message["parts"][0] else "*message error*"
             st.markdown(content)


    # --- 4. Handle New User Input ---
    if prompt := st.chat_input("Ask a health-related question..."):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("You"):
            st.markdown(prompt)

        # Get response from the backend API
        payload = {"prompt": prompt}

        try:
            with st.spinner("AI is thinking..."):
                response = requests.post(chat_api_url, headers=headers, json=payload, timeout=30) # Add timeout

                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get("response", "Error: No response text found.")

                    with st.chat_message("AI Assistant"):
                        st.markdown(response_text)
                    st.session_state.messages.append({"role": "model", "parts": [response_text]})

                # --- IMPROVED ERROR REPORTING ---
                else:
                     error_detail = "No details provided."
                     try: # Try to get detail from JSON response
                         error_detail = response.json().get('detail', error_detail)
                     except requests.exceptions.JSONDecodeError:
                         error_detail = response.text # Show raw text if not JSON
                     st.error(f"Error getting AI response (Status: {response.status_code}). Detail: {error_detail}")
                     # Remove the user's message if API call failed
                     st.session_state.messages.pop()


        except requests.exceptions.Timeout:
            st.error("The AI service timed out. Please try again.")
            st.session_state.messages.pop()
        except requests.exceptions.RequestException as e:
            st.error(f"Network error communicating with the AI: {e}")
            st.session_state.messages.pop()
        except Exception as e: # Catch unexpected errors
             st.error(f"An unexpected error occurred: {e}")
             st.session_state.messages.pop()

        # Rerun AFTER processing the response/error to update the display cleanly
        st.rerun()

