# models/indexer.py

import os
from byaldi import RAGMultiModalModel
from models.doc_to_pdf import convert_docs_to_pdfs
from logger import get_logger

logger = get_logger(__name__)

def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali', overwrite_index=True):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str, optional): The path where the index should be saved. Default is None.
        indexer_model (str): The name of the indexer model to use.
        overwrite_index (bool): Whether to overwrite existing index files. Default is True.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path, overwrite=False)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model '{indexer_model}'")
        logger.info(f"RAG model initialized with model '{indexer_model}'.")

        # Check index path and log if not specified
        if index_path is None:
            index_path = os.path.join(folder_path, index_name)
            logger.warning(f"No index path provided. Defaulting to '{index_path}'")

        # Index the documents in the folder
        RAG.index(
            input_path=folder_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=overwrite_index
        )

        logger.info(f"Indexing completed successfully. Index saved at '{index_path}'.")

        return RAG

    except FileNotFoundError as fnf_error:
        logger.error(f"Folder not found: {folder_path}. Ensure the path is correct. Error: {fnf_error}")
        raise
    except ValueError as val_error:
        logger.error(f"Model initialization failed: {val_error}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during indexing: {str(e)}")
        raise
