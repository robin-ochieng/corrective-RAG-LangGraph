"""
Base definitions for LangGraph nodes used in corrective RAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NodeContext:
    """
    Carries contextual information through the LangGraph workflow.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


class BaseNode:
    """
    Base class for LangGraph nodes.

    Concrete nodes should implement the ``run`` method to perform their
    specific transformation in the corrective RAG workflow.
    """

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def run(self, context: NodeContext) -> NodeContext:
        """Execute the node logic and return the updated context."""
        raise NotImplementedError("BaseNode subclasses must implement run().")
