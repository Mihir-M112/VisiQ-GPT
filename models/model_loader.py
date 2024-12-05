import sys
import os
import requests
import json
import base64
from typing import Optional
import time
import ollama


# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import get_logger

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        logger.error(f"Image read error: {e}")
        raise

def generate_response(
    prompt: str,
    model_name: str = "llama3.2-vision",
    image_path: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    system_prompt: str = "Carefully examine the prompt to identify key information relevant to the question in concise manner."  # Default system prompt
) -> str:
    """
    Generate response with optional image input, system prompt, and performance optimizations.
    
    Args:
        prompt (str): User's query
        model_name (str): Ollama model name
        image_path (str, optional): Path to image file
        max_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter
        top_k (int): Limits number of considered tokens
        system_prompt (str): Instruction to set the model's behavior
    
    Returns:
        str: Model's response
    """
    # Start timing
    start_time = time.time()

    # Combine system prompt and user prompt
    combined_prompt = f"{system_prompt}\n\n{prompt}"

    # Validate image path if provided
    if image_path and not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return "Error: Image file not found."
 

    # Prepare payload
    payload = {
        "model": model_name,
        "prompt": combined_prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
    }

    # Add image if provided
    if image_path:
        try:
            payload["images"] = [encode_image_to_base64(image_path)]
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return f"Error processing image: {e}"

    # Send request
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        
        # Log response details for debugging
        logger.info(f"Full API Response: {json.dumps({
            'status_code': response.status_code,
            'model': model_name,
            'total_time': time.time() - start_time
        }, indent=2)}")
        
        # Check response
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code}")
            return "API request failed"
        
        # Extract response
        response_data = response.json()
        
        # Log performance metrics
        logger.info(f"Performance Metrics: {json.dumps({
            'total_duration_ms': response_data.get('total_duration', 0) / 1_000_000,
            'prompt_eval_count': response_data.get('prompt_eval_count', 0),
            'eval_count': response_data.get('eval_count', 0)
        }, indent=2)}")
        
        return response_data.get('response', 'No response')
    


    except Exception as e:
        logger.error(f"Request error: {e}")
        return f"Error: {e}"



# Test script
if __name__ == "__main__":
    # Text query
    text_response = generate_response("What is the Machine learning? in short")
    print(text_response)

    # Image query (replace with your actual image path)
    image_path = "D:\\Code\\VisiQ-GPT\\data\\Vector.png"
    if os.path.exists(image_path):
        image_response = generate_response("Describe this image fast", image_path=image_path)
        print(image_response)
    else:
        print(f"Image not found at {image_path}")

   

