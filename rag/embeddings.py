"""
embeddings.py

Embedding model configuration for the RAG pipeline.

Responsibilities
----------------
- Create the HuggingFace embedding model.
- Keep embedding configuration centralized.
- Lazily initialize the embedding model.
- Reuse the same embedding model instance.

Used by
--------
vector_store.py
retriever.py
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and cache the HuggingFace embedding model.

    The model is initialized only when this function is first called.
    Subsequent calls return the same cached instance.

    Returns:
        Configured HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

