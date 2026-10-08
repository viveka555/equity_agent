"""Tests for local persistence and deterministic IDs in Chroma."""

from pathlib import Path

from langchain_core.documents import Document

from rag.vector_store import build_vector_store


class _FakeEmbeddings:
    """Small deterministic vectors for offline Chroma tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one fixed-size vector for each document."""
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a fixed-size vector for a query."""
        return [float(len(text)), 1.0, 0.0]


def test_build_vector_store(tmp_path: Path, monkeypatch) -> None:
    """Verify documents persist locally without downloading an embedding model."""
    documents = [
        Document(
            page_content="Annual report risk disclosures.",
            metadata={"source": "BEL_2025.pdf", "page": 25},
        ),
        Document(
            page_content="Annual report revenue discussion.",
            metadata={"source": "BEL_2025.pdf", "page": 26},
        ),
    ]
    monkeypatch.setattr("rag.vector_store.CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(
        "rag.vector_store.get_embedding_model",
        _FakeEmbeddings,
    )

    vector_store = build_vector_store(documents)
    stored_documents = vector_store.get()

    assert len(stored_documents["ids"]) == len(documents)
    assert stored_documents["metadatas"] == [
        {"source": "BEL_2025.pdf", "page": 25},
        {"source": "BEL_2025.pdf", "page": 26},
    ]


def test_build_vector_store_rejects_empty_documents() -> None:
    """Verify empty document lists are rejected before opening Chroma."""
    import pytest

    with pytest.raises(ValueError, match="empty documents"):
        build_vector_store([])
