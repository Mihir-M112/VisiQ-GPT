import sys
import os
import time
import requests
import base64
from typing import Optional



# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import get_logger
# Logger setup
logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

def process_pdf(pdf_path: str) -> str:
    """Extract text from a PDF and return its content."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return ""

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
    pdf_path: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    system_prompt: str = "Carefully read the user prompt and provide a detailed response:"
) -> str:
    """Generate a response from the model using optional image or PDF input."""
    try:
        start_time = time.time()
        
        combined_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Process PDF if provided
        if pdf_path:
            pdf_content = process_pdf(pdf_path)
            if pdf_content:
                combined_prompt = f"Context: {pdf_content}\n\n{prompt}"
                logger.info("PDF content appended to the prompt.")
            else:
                logger.warning("No content extracted from the PDF.")
        
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
            payload["images"] = [encode_image_to_base64(image_path)]
            logger.info("Image added to the payload.")

        # Send request to Ollama server
        response = requests.post(OLLAMA_URL, json=payload)
        response_time = time.time() - start_time

        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return "API request failed."

        logger.info(f"Response received in {response_time:.2f} seconds.")
        response_data = response.json()
        return response_data.get('response', 'No response content.')

    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        return f"Request failed: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"An error occurred: {e}"


# Test script
if __name__ == "__main__":
    # Text query test
    try:
        text_response = generate_response("What is machine learning?")
        print("Text Response:", text_response)
    except Exception as e:
        logger.error(f"Text query test failed: {e}")

    # # Image query test
    # image_path = "D:\\Code\\VisiQ-GPT\\data\\Vector.png"
    # if os.path.exists(image_path):
    #     try:
    #         image_response = generate_response("Describe this image", image_path=image_path)
    #         print("Image Response:", image_response)
    #     except Exception as e:
    #         logger.error(f"Image query test failed: {e}")
    # else:
    #     logger.warning(f"Image not found at {image_path}")

    # # PDF query test
    # pdf_path = "D:\\Code\\VisiQ-GPT\\data\\Vector Database.pdf"
    # if os.path.exists(pdf_path):
    #     try:
    #         pdf_response = generate_response("How to make coffee", pdf_path=pdf_path)
    #         print("PDF Response:", pdf_response)
    #     except Exception as e:
    #         logger.error(f"PDF query test failed: {e}")
    # else:
    #     logger.warning(f"PDF not found at {pdf_path}")