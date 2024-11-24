import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from dotenv import load_dotenv
import google.generativeai as genai
import sys

# Add project root to sys.path to import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}

# Function to detect the best available device
def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Function to load the selected model
def load_model(model_choice):
    """
    Load and cache the specified model.

    Args:
        model_choice (str): 'llama3.2', 'llama3.2-vision', or 'gemini'

    Returns:
        tuple: model, processor (if applicable), device (if applicable)
    """
    global _model_cache

    # Check if the model is already loaded
    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    try:
        # Handle Google Gemini model
        if model_choice == 'gemini':
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in .env file.")
                raise ValueError("GOOGLE_API_KEY not found in .env file.")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-002')
            logger.info("Google Gemini model loaded successfully.")
            _model_cache[model_choice] = (model, None, None)
            return _model_cache[model_choice]

        # Handle Llama 3.2 Vision model
        elif model_choice == 'llama3.2-vision':
            device = detect_device()
            model_id = "alpindale/Llama-3.2-11B-Vision-Instruct"

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_id)
            model.to(device)

            _model_cache[model_choice] = (model, processor, device)
            logger.info("Llama 3.2 Vision model loaded and cached.")
            return _model_cache[model_choice]

        else:
            logger.error(f"Invalid model choice: {model_choice}")
            raise ValueError("Invalid model choice. Choose 'llama3.2-vision' or 'gemini'.")

    except Exception as e:
        logger.error(f"Error loading model '{model_choice}': {str(e)}")
        raise

# Example function to handle model inference for Web UI
def generate_response(model_choice, input_data):
    """
    Generate a response from the specified model.

    Args:
        model_choice (str): 'llama3.2-vision' or 'gemini'
        input_data (str): User input or prompt data

    Returns:
        str: Generated response
    """
    model, processor, device = load_model(model_choice)

    if model_choice == 'gemini':
        response = model.generate_content(input_data)
        return response.text

    elif model_choice == 'llama3.2-vision':
        inputs = processor(text=input_data, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response

    else:
        logger.error(f"Unsupported model for response generation: {model_choice}")
        raise ValueError("Unsupported model for response generation.")

