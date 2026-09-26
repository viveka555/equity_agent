"""
embeddings.py

Embedding model configuration for the RAG pipeline.

Responsibilities
----------------
- Create a reusable embedding model
- Keep model configuration centralized
- Avoid creating multiple embedding instances
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the embedding model.

    Returns:
        Configured HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(
        model =EMBEDDING_MODEL,
        model_kwargs = {"device":"cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

# Singleton instance
embeddings = get_embedding_model()