"""Ingest a local annual-report PDF into the project's Chroma vector store."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.logging_config import setup_logging
from rag.chunker import split_documents
from rag.pdf_loader import load_pdf
from rag.vector_store import build_vector_store

logger = logging.getLogger(__name__)


def main() -> None:
    """Load, chunk, and embed the annual report PDF supplied on the CLI."""
    parser = argparse.ArgumentParser(
        description="Load an annual-report PDF into the local RAG vector store."
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the annual-report PDF, relative to this project or absolute.",
    )
    args = parser.parse_args()

    setup_logging()
    pdf_path = args.pdf_path.expanduser()
    if not pdf_path.is_file():
        parser.error(f"PDF file does not exist: {pdf_path}")

    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)
    if not chunks:
        parser.error(f"No extractable text found in PDF: {pdf_path}")

    build_vector_store(chunks)
    logger.info(
        "Ingested %s pages into %s searchable chunks from %s",
        len(pages),
        len(chunks),
        pdf_path,
    )
    print(f"Ingested {len(pages)} pages as {len(chunks)} searchable chunks.")


if __name__ == "__main__":
    main()
