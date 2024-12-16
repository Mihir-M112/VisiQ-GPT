import os
import base64
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict
import requests
from datetime import datetime, timedelta

class ImageEmbeddingProcessor:
    def __init__(self, cache_dir: str = "./image_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.embedding_url = "http://localhost:11434/api/embeddings"
        self.cache_duration = timedelta(days=7)

    def _get_image_hash(self, image_path: str) -> str:
        """Generate unique hash for image file."""
        image_stat = os.stat(image_path)
        return hashlib.md5(f"{image_path}{image_stat.st_mtime}".encode()).hexdigest()

    def _get_image_analysis(self, base64_image: str) -> Optional[str]:
        """Get detailed image analysis using llama3.2-vision."""
        detailed_prompt = """Analyze this image in extreme detail. Structure your analysis as follows:

1. General Overview:
   - Main subject/focus
   - Overall composition
   - Time of day/lighting conditions
   - Color palette

2. Key Elements:
   - Foreground elements and their details
   - Background elements and their details
   - Any text or symbols present
   - Notable patterns or textures

3. Technical Details:
   - Image quality and clarity
   - Perspective and depth
   - Lighting and shadows


4. Contextual Information:
   - Setting/environment
   - Mood/atmosphere
   - Apparent purpose or context
   - Any cultural or historical references

5. Additional Details:
   - Small or subtle elements
   - Interesting features
   - Any unique or unusual aspects

6. If textual image provided:
    - Analyze and process the text inside the image (if any) and its relevance to the overall image and user query.
    - Note any discrepancies between text and image
    - Provide any additional insights or interpretations

7. If you think any answer to user query can be inferred from the image, provide that as well in short and concise manner.


Please be thorough and precise in your analysis, noting even minor details that might be relevant for future queries. Provide Response fast and accurate."""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2-vision",
                    "prompt": detailed_prompt,
                    "images": [base64_image],
                    "options": {
                        "temperature": 0.2,  # Slightly increased for more natural language
                        "num_predict": 500   # Increased for more detailed response
                    }
                }
            )
            if response.status_code == 200:
                return response.json().get('response', None)
            return None
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def _get_image_embedding(self, base64_image: str) -> Optional[list]:
        """Get embeddings using nomic-embed-text."""
        try:
            response = requests.post(
                self.embedding_url,
                json={
                    "model": "nomic-embed-text",
                    "prompt": "",
                    "images": [base64_image]
                }
            )
            if response.status_code == 200:
                return response.json().get('embeddings', None)
            return None
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def process_image(self, image_path: str) -> Dict[str, any]:
        """Process image using vision model and embeddings."""
        try:
            image_hash = self._get_image_hash(image_path)
            cache_file = self.cache_dir / f"{image_hash}.json"

            # Check cache
            if cache_file.exists():
                cache_data = json.loads(cache_file.read_text())
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time < self.cache_duration:
                    return cache_data

            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Get both analysis and embeddings
            vision_analysis = self._get_image_analysis(base64_image)
            embeddings = self._get_image_embedding(base64_image)

            # Cache results
            cache_data = {
                'base64_image': base64_image,
                'vision_analysis': vision_analysis,
                'embeddings': embeddings,
                'timestamp': datetime.now().isoformat(),
                'path': str(image_path)
            }
            cache_file.write_text(json.dumps(cache_data))
            return cache_data

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

