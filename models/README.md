# VisiQ-GPT Models Documentation

This document provides an overview of the model architecture, data flow, and component interactions in the VisiQ-GPT system.

## Architecture Overview

The system consists of four main components:
- Model Loader (Entry Point)
- Image Embedding Processor
- PDF Embedding Processor
- MongoDB Manager

### Component Interactions
```mermaid
graph TD
    A[Model Loader] --> B[Image Processor]
    A --> C[PDF Processor]
    B --> D[MongoDB Manager]
    C --> D
    B --> E[Local Cache]
    C --> F[ChromaDB]

Data Flow
1. Image Processing Flow
Entry Point: model_loader.py::generate_response()

Accepts image path and user query
Initializes ImageEmbeddingProcessor with MongoDB enabled
Caching Strategy:

Primary: MongoDB cache
Secondary: Local file cache
Cache duration: 7 days
Processing Steps:
Image Input -> Base64 Encoding -> Vision Analysis -> Embeddings Generation -> Storage in MongoDB


4. MongoDB Storage Format:
{
    "image_id": "unique_image_id",
    "image_path": "path_to_image",
    "embeddings": "image_embeddings",
    "created_at": "timestamp",
    "updated_at": "timestamp"
}



2. PDF Processing Flow
1. Entry Point: model_loader.py::generate_response()

Accepts PDF path and user query
Initializes EmbeddingsProcessor
2. Storage Strategy:

Primary: MongoDB for caching
Secondary: ChromaDB for vector search
Document chunking for efficient processing
3. Processing Steps:

PDF Input -> Text Extraction -> Chunk Splitting -> Embeddings Generation -> Dual Storage in MongoDB and ChromaDB
4.
MongoDB Storage Format:
{
    "pdf_id": "unique_pdf_id",
    "pdf_path": "path_to_pdf",
    "embeddings": "pdf_embeddings",
    "created_at": "timestamp",
    "updated_at": "timestamp"
}

Usage Examples
Image Processing:
```python   
# Process image query
response = generate_response(
    prompt="What's in this image?",
    image_path="path/to/image.jpg",
    model_name="llama3.2-vision"
)
```

PDF Processing

```python
from models.model_loader import generate_response

# Process PDF query
response = generate_response(
    prompt="Summarize the document",
    pdf_path="path/to/document.pdf",
    detailed_response=True
)
```

### Configuration
MongoDB Setup

```bash
# MongoDB Configuration
from models.db_manager import MongoDBManager

# Custom MongoDB configuration
MongoDBManager.set_connection_string(
    username="your_username",
    password="your_password",
    cluster="your_cluster.mongodb.net"
)
```

Cache Directories
Image cache: ./image_cache/
ChromaDB storage: ./demo-rag-chroma/
Error Handling
The system implements multiple fallback mechanisms:

MongoDB cache check
Local file cache check
Fresh processing with error logging
Graceful degradation to basic functionality
Performance Considerations
Cached responses typically return in < 1 second
Fresh processing may take 2-5 seconds
MongoDB indexes optimize query performance
ChromaDB provides efficient vector similarity search
Security Notes
MongoDB credentials should be stored securely
Base64 encoding used for image transfer
Local cache implements file-based security







-> To do:

(db_manager.py) - Implement MongoDB storage for image and PDF embeddings update to do so.

Now i want to implement the mongodb for the image and pdf processing. I will use the pymongo library to connect to the mongodb and store the image and pdf embeddings in the mongodb. then i will use the mongodb to retrieve the embeddings for the image and pdf processing. Such that the response time will be reduced and the system will be more efficient. 

- Implemnet mongo db such that it consists of user last 5 query whether related to image or pdf so the model have knowledge base of the user query and can provide the better response to the user.

- implement fastapi to create the api for the model such that the user can interact with the model using the api.

- impelment the docker to containerize the model and deploy it on the cloud such that the user can access the model from anywhere.

- implement streamlit to create the web app for the model such that the user can interact with the model using the web app.
