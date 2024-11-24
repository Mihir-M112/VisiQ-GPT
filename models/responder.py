# models/responder.py

from models.model_loader import load_model
from dotenv import load_dotenv
from logger import get_logger
from PIL import Image
import os
import base64
import ollama

logger = get_logger(__name__)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(images, query, session_id, model_choice='llama-vision'):
    """
    Generates a response using the selected model based on the query and images.
    Returns: (response_text, used_images)
    """
    try:
        logger.info(f"Generating response using model '{model_choice}'.")
        
        # Ensure images are full paths
        valid_images = [os.path.join('static', img) if not img.startswith('static') else img for img in images]
        valid_images = [img for img in valid_images if os.path.exists(img)]
        
        if not valid_images:
            logger.warning("No valid images found for analysis.")
            return "No images could be loaded for analysis.", []

        # Handle different model cases
        if model_choice == 'gemini':
            model, _ = load_model('gemini')

            content = [query]  # Add the text query first
            for img_path in valid_images:
                try:
                    img = Image.open(img_path)
                    content.append(img)
                except Exception as e:
                    logger.error(f"Error opening image {img_path}: {e}")
            
            try:
                response = model.generate_content(content)
                generated_text = response.text if response.text else "No response generated."
                logger.info("Response generated using Gemini model.")
                return generated_text, valid_images
            except Exception as e:
                logger.error(f"Error in Gemini processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing: {str(e)}", []

        elif model_choice == 'llama-vision':
            try:
                # Use the first valid image
                message = {
                    'role': 'user',
                    'content': query,
                    'images': [valid_images[0]]  # Assuming single image handling for simplicity
                }
                
                response = ollama.chat(
                    model='llama3.2-vision',
                    messages=[message]
                )
                
                logger.info("Response generated using Ollama Llama Vision model.")
                return response['message']['content'], valid_images
                
            except Exception as e:
                logger.error(f"Error in Llama Vision processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing: {str(e)}", []
        else:
            logger.error(f"Invalid model choice: {model_choice}")
            return "Invalid model selected.", []

    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return f"An error occurred while generating the response: {str(e)}", []
