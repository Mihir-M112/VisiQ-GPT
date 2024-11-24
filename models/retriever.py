# models/retriever.py

import base64
import os
from PIL import Image
from io import BytesIO
from logger import get_logger
import hashlib

logger = get_logger(__name__)

def retrieve_documents(RAG, query, session_id, k=3):
    """
    Retrieves relevant documents (images or PDFs) based on the user query using Byaldi.

    Args:
        RAG (RAGMultiModalModel): The RAG model with the indexed documents.
        query (str): The user's query.
        session_id (str): The session ID to store documents in per-session folder.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of file paths corresponding to the retrieved documents.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        results = RAG.search(query, k=k)
        files = []
        session_folder = os.path.join('static', 'documents', session_id)
        os.makedirs(session_folder, exist_ok=True)

        for i, result in enumerate(results):
            if result.base64:
                file_data = base64.b64decode(result.base64)
                file_hash = hashlib.md5(file_data).hexdigest()

                # Handle image documents
                if result.type == 'image':
                    image_filename = f"retrieved_{file_hash}.png"
                    image_path = os.path.join(session_folder, image_filename)

                    if not os.path.exists(image_path):
                        image = Image.open(BytesIO(file_data))
                        image.save(image_path, format='PNG')
                        logger.debug(f"Retrieved and saved image: {image_path}")
                    else:
                        logger.debug(f"Image already exists: {image_path}")

                    files.append(os.path.join('documents', session_id, image_filename))
                
                # Handle PDF documents
                elif result.type == 'pdf':
                    pdf_filename = f"retrieved_{file_hash}.pdf"
                    pdf_path = os.path.join(session_folder, pdf_filename)

                    if not os.path.exists(pdf_path):
                        with open(pdf_path, 'wb') as pdf_file:
                            pdf_file.write(file_data)
                        logger.debug(f"Retrieved and saved PDF: {pdf_path}")
                    else:
                        logger.debug(f"PDF already exists: {pdf_path}")

                    files.append(os.path.join('documents', session_id, pdf_filename))
                
                else:
                    logger.warning(f"Unsupported document type for document {result.doc_id}")

            else:
                logger.warning(f"No base64 data for document {result.doc_id}, page {result.page_num}")

        logger.info(f"Total {len(files)} documents retrieved. File paths: {files}")
        return files

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []
