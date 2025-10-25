"""
Corrective RAG with LangGraph

A corrective Retrieval-Augmented Generation system that implements 
self-correction mechanisms for improved response quality.
"""

__version__ = "0.1.0"
__author__ = "Robin Ochieng"

from .core import CorrectiveRAG

__all__ = ["CorrectiveRAG"]