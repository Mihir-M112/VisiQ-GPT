# models/model_loader.py
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from dotenv import load_dotenv
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
def load_model(model_choice, temperature=0.7):
    """
    Load the specified model and set temperature.

    Args:
        model_choice (str): 'llama3.2' or 'llama3.2-vision'
        temperature (float): Sampling temperature for model generation

    Returns:
        tuple: model, processor, device
    """
    global _model_cache

    # Check if the model is already loaded
    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    device = detect_device()
    logger.info(f"Loading model '{model_choice}' on {device}")

    try:
        if model_choice == 'llama3.2':
            model_id = "meta-llama/Llama-3.2-7B"
        elif model_choice == 'llama3.2-vision':
            model_id = "alpindale/Llama-3.2-11B-Vision-Instruct"
        else:
            raise ValueError("Invalid model choice. Choose 'llama3.2' or 'llama3.2-vision'.")

        # Load model and processor
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)

        # Store in cache
        _model_cache[model_choice] = (model, processor, device)
        logger.info(f"Model '{model_choice}' loaded successfully.")

        return _model_cache[model_choice]
    
    except Exception as e:
        logger.error(f"Error loading {model_choice}: {str(e)}")
        raise

# Main function for standalone testing
def main():
    # For testing purposes: Choose a model and set temperature
    model_choice = input("Enter model choice ('llama3.2' or 'llama3.2-vision'): ")
    temperature = float(input("Enter temperature (0.1 - 1.0): "))

    model, processor, device = load_model(model_choice, temperature)
    logger.info(f"{model_choice} is loaded with temperature {temperature}")

if __name__ == "__main__":
    main()
