"""Unit tests for lazy, cached embedding-model creation."""

from unittest.mock import Mock

from rag.embeddings import get_embedding_model


def test_embedding_model_is_created_lazily_and_cached(monkeypatch) -> None:
    """Verify model construction is deferred and reused without downloads."""
    fake_model = Mock()
    factory = Mock(return_value=fake_model)
    get_embedding_model.cache_clear()
    monkeypatch.setattr("rag.embeddings._create_embedding_model", factory)

    try:
        first = get_embedding_model()
        second = get_embedding_model()
    finally:
        get_embedding_model.cache_clear()

    assert first is fake_model
    assert second is first
    factory.assert_called_once_with()
