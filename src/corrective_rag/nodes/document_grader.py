"""Document grading node for filtering retrieved evidence."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .base import BaseNode
from .state import GraphState


class RelevanceScore(BaseModel):
    """Structured relevance signal returned by the grader."""

    binary_score: str = Field(
        description="Whether the document is relevant to the question. Must be 'yes' or 'no'."
    )


class DocumentGraderNode(BaseNode):
    """Filters retrieved documents by leveraging an LLM-based relevance grader."""

    name = "document_grader"

    def __init__(
        self,
        *,
        chain: Optional[Runnable] = None,
    llm: Optional[Any] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(config=config)
        self._parser = PydanticOutputParser(pydantic_object=RelevanceScore)
        self._chain = chain or self._build_chain(llm)
        self._last_scores: List[RelevanceScore] = []

    def _build_chain(self, llm: Optional[Any]) -> Runnable:
        llm_instance = llm or ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a retrieval grader deciding if a document contains information "
                    "that helps answer a user's question. Return 'yes' if it is relevant, "
                    "otherwise return 'no'.",
                ),
                (
                    "human",
                    "Question: {question}\n" "Document:\n{document}\n" "{format_instructions}",
                ),
            ]
        )

        return prompt | llm_instance | self._parser

    def run(self, state: GraphState) -> GraphState:  # type: ignore[override]
        if not state.retrieved_documents:
            return state

        relevant_docs: List[Document] = []
        dropped = 0
        scores: List[RelevanceScore] = []

        for doc in state.retrieved_documents:
            try:
                result: RelevanceScore = self._chain.invoke(
                    {
                        "question": state.question,
                        "document": doc.page_content,
                        "format_instructions": self._parser.get_format_instructions(),
                    }
                )
            except Exception:
                dropped += 1
                continue

            scores.append(result)
            if result.binary_score.strip().lower() == "yes":
                relevant_docs.append(doc)
            else:
                dropped += 1

        self._last_scores = scores

        if relevant_docs:
            state.retrieved_documents = relevant_docs
        else:
            state.retrieved_documents = []

        state.metadata.setdefault("retrieval", {})
        state.metadata["retrieval"].update(
            {
                "graded": True,
                "kept_documents": len(relevant_docs),
                "dropped_documents": dropped,
            }
        )

        if dropped > 0:
            state.metadata["web_search_required"] = True
        else:
            state.metadata.setdefault("web_search_required", False)

        return state

    @property
    def last_scores(self) -> List[RelevanceScore]:
        """Structured grading results from the most recent run."""

        return self._last_scores


__all__ = ["DocumentGraderNode", "RelevanceScore"]
