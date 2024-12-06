import os
import sys
import time
import base64
from typing import Optional, Dict, Any, List
import logging

# External library imports
import requests
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

class DocumentProcessor:
    """Handles processing of various document types."""
    
    @staticmethod
    def process_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        Extract text from a PDF file with optional page limit.
        
        Args:
            pdf_path (str): Path to the PDF file
            max_pages (Optional[int]): Maximum number of pages to extract
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                pages_to_process = reader.pages[:max_pages] if max_pages else reader.pages
                
                text = ''.join(page.extract_text() or '' for page in pages_to_process)
                
                logging.info(f"Successfully processed PDF: {pdf_path}")
                logging.info(f"Total pages: {total_pages}, Processed pages: {len(pages_to_process)}")
                return text
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            return ""


class VectorStoreManager:
    """Manages vector storage and retrieval using Chroma DB."""
    
    def __init__(self, collection_name: str = "visiq_collection"):
        """
        Initialize Chroma DB client and collection.
        
        Args:
            collection_name (str): Name of the vector collection
        """
        try:
            self.client = chromadb.PersistentClient(path="./chroma_storage")
            
            # Use Nomic embedding from Ollama
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name="nomic-embed-text"
            )
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logging.info(f"Initialized Chroma DB collection: {collection_name}")
        
        except Exception as e:
            logging.error(f"Vector store initialization error: {e}")
            raise

    def add_document(self, documents: List[str], metadata: Optional[List[Dict[str, str]]] = None, chunk_size: int = 500):
        """
        Add documents to the vector store with chunking to prevent timeout.
        
        Args:
            documents (List[str]): List of document texts
            metadata (Optional[List[Dict[str, str]]]): Optional metadata for documents
            chunk_size (int): Size of chunks to process
        """
        try:
            # Chunk large documents
            def chunk_text(text, chunk_size):
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            processed_docs = []
            processed_metadata = []
            ids = []

            for i, doc in enumerate(documents):
                chunks = chunk_text(doc, chunk_size)
                for j, chunk in enumerate(chunks):
                    processed_docs.append(chunk)
                    processed_metadata.append(metadata[i] if metadata else {})
                    ids.append(f"doc_{i}_chunk_{j}")

            # Add chunks with a delay to prevent potential timeout
            for i in range(0, len(processed_docs), 10):
                batch_docs = processed_docs[i:i+10]
                batch_metadata = processed_metadata[i:i+10]
                batch_ids = ids[i:i+10]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                time.sleep(0.5)  # Small delay between batches
            
            logging.info(f"Added {len(processed_docs)} document chunks to vector store")
        
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            raise

    def query_documents(self, query: str, n_results: int = 5):
        """
        Query documents from the vector store.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
        
        Returns:
            Dict[str, Any]: Query results
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            logging.info(f"Queried documents with top {n_results} results")
            return results
        
        except Exception as e:
            logging.error(f"Document query error: {e}")
            raise

def main():
    """Main execution point for document processing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('visiq_gpt.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        # Initialize vector store
        vector_store = VectorStoreManager()
        
        # Process PDF
        pdf_path = r"D:\Code\VisiQ-GPT\data\Vector Database.pdf"
        
        # Extract text with page limit to prevent excessive processing
        pdf_text = DocumentProcessor.process_pdf(pdf_path, max_pages=10)
        
        if pdf_text:
            # Add document to vector store with metadata
            vector_store.add_document(
                [pdf_text], 
                [{'source': pdf_path, 'type': 'pdf'}]
            )
            
            # Example query
            query = "What are vector databases?"
            results = vector_store.query_documents(query)
            
            # Print retrieved context
            for doc in results.get('documents', []):
                print("Retrieved Context:", doc)
        
    except Exception as e:
        logging.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()