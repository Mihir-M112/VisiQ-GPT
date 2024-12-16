import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from typing import Tuple, List
import os
from models.db_manager import MongoDBManager

class EmbeddingsProcessor:
    def __init__(self, use_mongodb: bool = True, custom_mongodb: str = None):
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )
        self.chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_app",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.use_mongodb = use_mongodb
        if use_mongodb:
            self.db_manager = MongoDBManager()
            if custom_mongodb:
                MongoDBManager.set_connection_string(custom_mongodb)

    def process_pdf(self, pdf_path: str) -> Tuple[str, None]:
        """Process PDF and store embeddings."""
        try:
            # First check MongoDB cache
            if self.use_mongodb and self.db_manager:
                cached_data = self.db_manager.get_pdf_embeddings(pdf_path)
                if cached_data and cached_data.get('text_content'):
                    # Uses existing content and embeddings from cache
                    return cached_data.get('text_content', ''), None

            # Process PDF if not in cache
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            
            if not docs:
                return "", None
                
            text_content = " ".join([doc.page_content for doc in docs])
            if not text_content.strip():
                return "", None
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            
            if not splits:
                return text_content, None
                
            documents, metadatas, ids = [], [], []
            for idx, split in enumerate(splits):
                if split.page_content.strip():
                    documents.append(split.page_content)
                    metadatas.append(split.metadata)
                    ids.append(f"{os.path.basename(pdf_path)}_{idx}")

            if documents:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                # Store in MongoDB if enabled
                if self.use_mongodb and self.db_manager:
                    self.db_manager.store_pdf_embeddings(pdf_path, {
                        'text_content': text_content,
                        'embeddings': self.collection.get(ids=ids)['embeddings'] if ids else None,
                        'chunks': documents,
                        'metadata': metadatas
                    })

                return text_content, None

            return text_content, None

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return "", None

    def query_similar_content(self, query: str, prompt: str, n_results: int = 3):
        """Query the vector store for similar content and validate relevance."""
        results = self.collection.query(
            query_texts=[prompt],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return None, False
            
        # Use the same re-ranking approach as app_doc_embed.py
        encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranks = encoder_model.rank(prompt, results['documents'][0], top_k=3)
        
        relevant_text = ""
        for rank in ranks:
            relevant_text += results['documents'][0][rank["corpus_id"]]
            
        # Check if we found any relevant content
        if not relevant_text:
            return None, False
            
        return relevant_text, True