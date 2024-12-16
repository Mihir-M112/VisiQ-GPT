import os
import base64
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from models.db_manager import MongoDBManager
from utils.logger import get_logger

logger = get_logger(__name__)

class ImageEmbeddingProcessor:
    def __init__(self, 
                 cache_dir: str = "./image_cache", 
                 use_mongodb: bool = True,
                 custom_mongodb: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.embedding_url = "http://localhost:11434/api/embeddings"
        self.cache_duration = timedelta(days=7)
        self.use_mongodb = use_mongodb
        if use_mongodb:
            logger.info("Initializing MongoDB connection...")
            self.db_manager = MongoDBManager()
            if custom_mongodb:
                MongoDBManager.set_connection_string(custom_mongodb)
            if self.db_manager._check_connection():
                logger.info("MongoDB connected successfully")
            else:
                logger.warning("Failed to connect to MongoDB")
        
        # Configure session with retries but no timeout
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[408, 429, 500, 502, 503, 504]
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    def _get_image_hash(self, image_path: str) -> str:
        """Generate unique hash for image file."""
        image_stat = os.stat(image_path)
        return hashlib.md5(f"{image_path}{image_stat.st_mtime}".encode()).hexdigest()

    def _get_image_analysis(self, base64_image: str) -> Optional[str]:
        """Get detailed image analysis using llama3.2-vision."""
        try:
            response = self.session.post(
                self.ollama_url,
                json={
                    "model": "llama3.2-vision",
                    "prompt": "Briefly describe what you see in this image.",
                    "images": [base64_image],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 250,  # Reduced token count
                        "num_ctx": 2048  # Add context window limit
                    }
                }  # Removed timeout
            )
            
            if response.status_code == 200:
                try:
                    return response.json().get('response', '')
                except Exception as json_error:
                    logger.error(f"JSON parsing error: {json_error}")
                    return response.text.strip()
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None

    def _get_image_embedding(self, base64_image: str) -> Optional[list]:
        """Get embeddings using nomic-embed-text."""
        try:
            response = self.session.post(
                self.embedding_url,
                json={
                    "model": "nomic-embed-text",
                    "prompt": "",
                    "images": [base64_image]
                }  # Removed timeout
            )
            if response.status_code == 200:
                return response.json().get('embeddings', None)
            return None
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return None

    def process_image(self, image_path: str) -> Dict[str, any]:
        """Process image using vision model and embeddings."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None

            # Read and encode image first
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to read image file: {e}")
                return None

            # Create basic response with image data
            cache_data = {
                'base64_image': base64_image,
                'vision_analysis': None,
                'embeddings': None,
                'timestamp': datetime.now().isoformat(),
                'path': str(image_path)
            }

            # Try to get vision analysis
            vision_analysis = self._get_image_analysis(base64_image)
            if vision_analysis:
                cache_data['vision_analysis'] = vision_analysis
            
            # Try to get embeddings
            embeddings = self._get_image_embedding(base64_image)
            if embeddings:
                cache_data['embeddings'] = embeddings

            # Store in MongoDB if enabled
            if self.use_mongodb and self.db_manager is not None:
                try:
                    self.db_manager.store_image_embeddings(image_path, cache_data)
                except Exception as e:
                    logger.error(f"Failed to store in MongoDB: {e}")

            return cache_data

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None