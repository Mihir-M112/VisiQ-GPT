import sys
import os
import time
import requests
import base64
from typing import Optional
from sentence_transformers import CrossEncoder  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.doc_embed import EmbeddingsProcessor 
from models.image_embed import ImageEmbeddingProcessor
import hashlib
import io
from pathlib import Path

# Add the root directory to the sys.path
from utils.logger import get_logger
# Logger setup
logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
IMAGE_CACHE_DIR = Path("./image_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

def process_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using embeddings processor."""
    try:
        processor = EmbeddingsProcessor()
        text_content, _ = processor.process_pdf(pdf_path)
        return text_content
    except Exception as e:
        logger.error(f"Error processing PDF with embeddings: {str(e)}")
        return ""

def encode_image_to_base64(image_path: str) -> str:
    """Get processed and cached image data."""
    try:
        processor = ImageEmbeddingProcessor()
        result = processor.process_image(image_path)
        if result and result.get('base64_image'):
            return result['base64_image']
        raise IOError("Failed to process image")
    except IOError as e:
        logger.error(f"Image read error: {e}")
        raise

def generate_response(
    prompt: str,
    model_name: str = "llama3.2-vision",
    image_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    max_tokens: int = 800,  # Increased max_tokens for detailed responses
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    detailed_response: bool = True,
    system_prompt: str = "Carefully read the user prompt and provide a detailed response:"
) -> str:
    """Generate a response from the model using optional image or PDF input."""
    try:
        start_time = time.time()
        
        # Process PDF if provided
        if pdf_path:
            pdf_content = process_pdf(pdf_path)
            if pdf_content:
                processor = EmbeddingsProcessor()
                relevant_text, is_relevant = processor.query_similar_content("", prompt)  # Empty query string as we're using prompt for search
                
                if not is_relevant:
                    return "I cannot find relevant information about this query in the provided document."
                
                detail_instruction = "Provide a comprehensive and detailed explanation with examples if available." if detailed_response else "Provide a brief and concise answer."
                combined_prompt = f"""System: Answer based on the following context. {detail_instruction}
                If the question cannot be answered from this context, say so.
                
                Context: {relevant_text}
                
                Question: {prompt}
                
                Remember to {'provide detailed explanations and examples' if detailed_response else 'keep the response concise'}."""
            else:
                logger.warning("No content extracted from the PDF.")
                return "Failed to extract content from the PDF."
        else:
            detail_instruction = "provide comprehensive and detailed explanations" if detailed_response else "be brief and concise"
            combined_prompt = f"System: Please {detail_instruction} in your response.\n\n{prompt}"
        
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

        # Add image processing with vision model and embeddings
        if image_path:
            processor = ImageEmbeddingProcessor()
            image_data = processor.process_image(image_path)
            if image_data:
                logger.info("Using cached image analysis and embeddings")
                enhanced_prompt = f"""System: I have analyzed this image in detail. Here's the comprehensive analysis:

{image_data['vision_analysis']}

Based on this detailed analysis, please address the following user query:
"{prompt}"

Consider all relevant aspects from the analysis when forming your response. If the query asks about specific details, refer to the appropriate sections of the analysis."""

                payload = {
                    "model": model_name,
                    "prompt": enhanced_prompt,
                    "images": [image_data['base64_image']],
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k
                    }
                }
                
                if image_data.get('embeddings'):
                    payload["embeddings"] = image_data['embeddings']
            else:
                # Fallback to basic image processing
                payload["images"] = [encode_image_to_base64(image_path)]

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
    # # Text query test
    # print("=========================================================================")
    # print("------------------------------Text Processing----------------------------")          
    # print("=========================================================================")
    
    # try:
    #     text_response = generate_response("What is machine learning?")
    #     print("Text Response:", text_response)
    # except Exception as e:
    #     logger.error(f"Text query test failed: {e}")

    print("=========================================================================")
    print("----------------------------Image Processing-----------------------------")          
    print("=========================================================================")

    # Image query test
    image_path = "D:\\Code\\VisiQ-GPT\\data\\phone.png"
    if os.path.exists(image_path):
        try:
            image_response = generate_response("Describe the image", image_path=image_path)
            print("Image Response:", image_response)
        except Exception as e:
            logger.error(f"Image query test failed: {e}")
    else:
        logger.warning(f"Image not found at {image_path}")

    # print("=========================================================================")
    # print("------------------------------PDF Processing-----------------------------")          
    # print("=========================================================================")

    # # PDF query test
    # pdf_path = "D:\\Code\\VisiQ-GPT\\data\\Vector Database.pdf"
    # if os.path.exists(pdf_path):
    #     try:
    #         # # Test with detailed response
    #         # pdf_response = generate_response(
    #         #     "How to make coffee",
    #         #     pdf_path=pdf_path,
    #         #     detailed_response=True
    #         # )
    #         # print("Detailed PDF Response:", pdf_response)
            
    #         # Test with concise response
    #         pdf_response = generate_response(
    #             "can you tell me page number of table 1?",
    #             pdf_path=pdf_path,
    #             detailed_response=False
    #         )
    #         print("Concise PDF Response:", pdf_response)
    #     except Exception as e:
    #         logger.error(f"PDF query test failed: {e}")
    # else:
    #     logger.warning(f"PDF not found at {pdf_path}")
    