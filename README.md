VisiQ-GPT: "Vision" and "IQ" Suggesting a system that provides intelligent answers using visual data with the integration of GPT

This project using llama3.2 vision for image processing and for general Q/A also.
Project name is VisiQ-GPT.
It should:
1. Implemented using ollama locally.
2. It is RAG based. For embeddings i will be using Nomic embed text also from ollama.
3. Document Upload and Indexing: Upload PDFs and images, which are then indexed using ColPali for retrieval.
4. Chat Interface: Engage in a conversational interface to ask questions about the uploaded documents. 
5. Session Management: Create, rename, switch between, and delete chat sessions.
6. Persistent Indexes: Indexes are saved on disk and loaded upon application restart.
7. It is integrated with MongoDB and Docker.
The model(llama3.2-vision) generate responses by understanding both the visual and textual content of the documents.




# VisiQ-GPT

VisiQ-GPT is an advanced document processing and question-answering system that implements VISION Retrieval Augmented Generation (V-RAG) architecture. The system combines document processing, intelligent indexing, and vision-language models to provide accurate responses to queries about documents and images.

## Architecture

The system is built on two main components:

### 1. Knowledge Base Creation
- *Document Processing*: Handles various document formats (PDFs, images, text files)
- *Page Extraction*: Splits documents into manageable pages (page-1, page-2, ..., page-n)
- *Indexing*: Uses ColPali for efficient document indexing and retrieval

### 2. Generation Part
- *Query Processing*: Handles user queries about the documents
- *Relevant Page Retrieval*: ColPali retrieves relevant document sections
- *Vision-Language Model (VLM)*: Processes both text and visual information to generate responses

## Key Features

1. *Multi-Modal Processing*
   - PDF document processing
   - Image processing with OCR
   - Text document handling

2. *Advanced Indexing*
   - ColPali-based document indexing
   - Efficient retrieval system
   - Persistent index storage

3. *Intelligent Response Generation*
   - Integration with Llama 3.2 Vision model
   - Context-aware responses
   - Visual and textual understanding

## Technical Stack

- *Core Model*: Llama 3.2 Vision

## System Requirements

- Python 3.8+
- Ollama

## Getting Started

1. Clone the repository
2. Install dependencies:   bash
   pip install -r requirements.txt   
3. Configure Ollama with required models
4. Run the application:   bash
   python main.py   

## Features

1. *Document Management*
   - Upload PDFs, images, and text documents
   - Automatic document processing and indexing
   - Persistent storage of processed documents

2. *Session Management*
   - Create new chat sessions
   - Rename existing sessions
   - Switch between multiple sessions
   - Delete unnecessary sessions

3. *Query Interface*
   - Natural language query processing
   - Context-aware responses
   - Support for both text and image-based queries

## Architecture Benefits

- *Efficient Retrieval*: ColPali indexing ensures fast and accurate document retrieval
- *Scalable Design*: Modular architecture allows for easy scaling and maintenance
- *Multi-Modal Support*: Seamless handling of both text and visual content
- *Persistent Storage*: Indexes and sessions are preserved across system restarts

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

[Add your license information here]

```