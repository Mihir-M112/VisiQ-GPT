# models/doc_to_pdf.py

import os
from docx2pdf import convert
from logger import get_logger

logger = get_logger(__name__)

def convert_docs_to_pdfs(folder_path, overwrite=False):
    """
    Converts .doc and .docx files in the folder to PDFs.
    
    Args:
        folder_path (str): The path to the folder containing documents.
        overwrite (bool): If True, overwrites existing PDFs. Default is False.
    """
    try:
        files_converted = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.doc', '.docx')):
                doc_path = os.path.join(folder_path, filename)
                pdf_path = os.path.splitext(doc_path)[0] + '.pdf'
                
                if not overwrite and os.path.exists(pdf_path):
                    logger.info(f"Skipped '{filename}' as PDF already exists.")
                    continue
                
                try:
                    convert(doc_path, pdf_path)
                    files_converted += 1
                    logger.info(f"Converted '{filename}' to PDF.")
                except Exception as file_error:
                    logger.error(f"Failed to convert '{filename}': {file_error}")
        
        if files_converted == 0:
            logger.warning("No files were converted. Check folder contents or overwrite setting.")
        else:
            logger.info(f"Successfully converted {files_converted} documents to PDF.")

    except Exception as e:
        logger.error(f"Error processing folder '{folder_path}': {e}")
        raise
