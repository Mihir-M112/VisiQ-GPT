import streamlit as st
import requests
import json
from pathlib import Path
import os

# Constants
API_URL = "http://localhost:8000"  # Remove /api suffix
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
SUPPORTED_DOC_TYPES = ["pdf"]

def init_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

def make_request(endpoint, data, files=None):
    """Make API request to backend"""
    try:
        # Ensure endpoint has /api prefix
        if not endpoint.startswith('/api/'):
            endpoint = f"/api/{endpoint.lstrip('/')}"
            
        url = f"{API_URL}{endpoint}"
        
        print(f"Making request to: {url}")
        
        # Always use form data for consistency
        form_data = {
            "prompt": data["prompt"],
            "temperature": str(data["temperature"]),
            "max_tokens": str(data["max_tokens"])
        }
        
        if data.get("session_id"):
            form_data["session_id"] = data["session_id"]
            
        headers = {
            'accept': 'application/json'
        }
        
        if files:
            response = requests.post(
                url,
                data=form_data,
                files=files,
                headers=headers
            )
        else:
            response = requests.post(
                url,
                data=form_data,  # Use form_data instead of json
                headers=headers
            )
        
        print(f"Request data: {form_data}")
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="VisiQ-GPT",
        page_icon="🤖",
        layout="wide"
    )

    init_session_state()

    st.title("VisiQ-GPT")
    st.subheader("Chat with your Documents and Images")

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Settings")
        
        uploaded_file = st.file_uploader(
            "Upload File (Images: png, jpg, jpeg | Documents: pdf)",
            type=SUPPORTED_IMAGE_TYPES + SUPPORTED_DOC_TYPES
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 100, 2000, 800)
        
        if uploaded_file:
            st.session_state.current_file = uploaded_file
            st.success(f"File uploaded: {uploaded_file.name}")

    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["prompt"])
            with st.chat_message("assistant"):
                st.write(message["response"])

    # Chat input
    prompt = st.chat_input("Ask a question...")
    
    if prompt:
        # Show user message
        with st.chat_message("user"):
            st.write(prompt)

        # Prepare API request
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "session_id": st.session_state.session_id
        }

        # Handle file upload if present
        files = None
        if st.session_state.current_file:
            files = {"file": st.session_state.current_file}

        # Make API request and show response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = make_request("chat", data, files)
                
                if result:
                    response = result["response"]
                    st.session_state.session_id = result["session_id"]
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "prompt": prompt,
                        "response": response
                    })
                    
                    st.write(response)

if __name__ == "__main__":
    main()