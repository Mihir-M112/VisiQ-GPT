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

OLLAMA_TIMEOUT = 60  # Increased timeout
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5

def create_http_session():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[408, 429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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
    system_prompt: str = "Carefully read the user prompt and provide a detailed response:",
    session_id: Optional[str] = None
) -> str:
    """Generate a response from the model using optional image or PDF input."""
    try:
        start_time = time.time()
        logger.info("Starting response generation...")
        
        # Initialize MongoDB manager for session tracking
        db_manager = MongoDBManager()
        
        # Create new session if none exists
        if not session_id:
            session_id = db_manager.create_session()
            
        # Get last used file from session if no new file provided
        if not image_path and not pdf_path:
            session_file = db_manager.get_session_file(session_id)
            if session_file:
                if session_file['file_type'] == 'image':
                    image_path = session_file['file_path']
                elif session_file['file_type'] == 'pdf':
                    pdf_path = session_file['file_path']

        # Initialize payload with common settings
        payload = {
            "model": model_name,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": 2048
            }
        }

        response_text = ""
        
        if image_path or pdf_path:
            # Update session with current file
            if image_path:
                db_manager.update_session_file(session_id, image_path, 'image')
            elif pdf_path:
                db_manager.update_session_file(session_id, pdf_path, 'pdf')

            # Process file-based query
            if pdf_path:
                processor = EmbeddingsProcessor(use_mongodb=True)
                relevant_text, is_relevant = processor.query_similar_content("", prompt)
                
                if not is_relevant:
                    response_text = "I cannot find relevant information about this query in the provided document. Would you like me to answer this question generally?"
                else:
                    payload["prompt"] = f"""System: Answer based on the following context.
                    Context: {relevant_text}
                    Question: {prompt}"""
            # ... existing image processing code ...

        else:
            # Handle general query
            payload["prompt"] = f"System: {system_prompt}\n\n{prompt}"

        if not response_text:  # If no response set yet, call the API
            with create_http_session() as session:
                response = session.post(OLLAMA_URL, json=payload)
                response_data = response.json()
                response_text = response_data.get('response', 'No response content.')

        # Store conversation history
        db_manager.add_conversation_history(session_id, prompt, response_text)
        
        return response_text

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            'response': f"An error occurred: {str(e)}",
            'session_id': session_id
        }


# Test script
if __name__ == "__main__":
    # Text query test
    print("=========================================================================")
    print("------------------------------Text Processing----------------------------")          
    print("=========================================================================")
    
    try:
        text_response = generate_response("What is machine learning?")
        print("Text Response:", text_response)
    except Exception as e:
        logger.error(f"Text query test failed: {e}")

    # print("=========================================================================")
    # print("----------------------------Image Processing-----------------------------")          
    # print("=========================================================================")

    # # Image query test
    # image_path = "D:\\Code\\VisiQ-GPT\\data\\phone.png"
    # if os.path.exists(image_path):
    #     try:
    #         image_response = generate_response("Can you tell me what the battery left?", image_path=image_path)
    #         print("Image Response:", image_response)
    #     except Exception as e:
    #         logger.error(f"Image query test failed: {e}")
    # else:
    #     logger.warning(f"Image not found at {image_path}")

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
    #             "what is the summary of the document?",
    #             pdf_path=pdf_path,
    #             detailed_response=True
    #         )
    #         print("Concise PDF Response:", pdf_response)
    #     except Exception as e:
    #         logger.error(f"PDF query test failed: {e}")
    # else:
    #     logger.warning(f"PDF not found at {pdf_path}")