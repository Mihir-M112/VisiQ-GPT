import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from typing import Tuple, List
import os

class EmbeddingsProcessor:
    def __init__(self):
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

    def process_pdf(self, pdf_path: str) -> Tuple[str, None]:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        documents, metadatas, ids = [], [], []
        for idx, split in enumerate(splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{os.path.basename(pdf_path)}_{idx}")

        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return " ".join(documents), None

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
