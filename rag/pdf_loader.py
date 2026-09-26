"""
pdf_loader.py

PDF loading module for annual reports.

Responsibilities
----------------
- Load PDF documents
- Preserve page numbers
- Return LangChain Document objects

Used by
--------
chunker.py
"""
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path:str):
    """
    Load an annual report.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        List of Document objects with metadata.
    """
    loader = PyPDFLoader(pdf_path)
    return loader.load()