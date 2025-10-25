"""
Tests for the corrective RAG system.
"""

import pytest
from corrective_rag import CorrectiveRAG


def test_corrective_rag_initialization():
    """Test that CorrectiveRAG can be initialized."""
    rag = CorrectiveRAG()
    assert rag is not None


def test_corrective_rag_query():
    """Test basic query functionality."""
    rag = CorrectiveRAG()
    response = rag.query("What is machine learning?")
    assert isinstance(response, str)
    assert "machine learning" in response.lower()


def test_corrective_rag_with_config():
    """Test initialization with configuration."""
    config = {"debug": True}
    rag = CorrectiveRAG(config=config)
    assert rag.config == config