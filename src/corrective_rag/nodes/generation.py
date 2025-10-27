"""Generation node responsible for synthesising the final answer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .base import BaseNode
from .state import GraphState


class GenerationNode(BaseNode):
    """Produces the final answer by prompting an LLM with retrieved evidence."""

    name = "generation"

    def __init__(
        self,
        *,
        chain: Optional[Runnable] = None,
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self._chain = chain or self._build_chain(llm)
        self.context_separator = self.config.get("context_separator", "\n\n")
        self.include_sources = bool(self.config.get("include_sources", True))
        self._last_prompt: Optional[Dict[str, Any]] = None

    def _build_chain(self, llm: Optional[Any]) -> Runnable:
        llm_instance = llm or ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=float(self.config.get("temperature", 0.2)),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a meticulous AI assistant that answers questions using the provided context. "
                    "Cite sources inline using numbers in brackets (e.g., [1]) when relevant. "
                    "If the context is insufficient, clearly say so and advise what information is missing.",
                ),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}\n\n"
                    "Instructions: Provide a concise yet complete answer. Use bullet points when listing items "
                    "and end with a short takeaway sentence.",
                ),
            ]
        )

        return prompt | llm_instance | StrOutputParser()

    def run(self, state: GraphState) -> GraphState:  # type: ignore[override]
        if not state.question or not state.question.strip():
            raise ValueError("GraphState.question must be populated before running GenerationNode.")

        context_blocks = self._format_documents(state.retrieved_documents)
        context_text = self.context_separator.join(context_blocks) if context_blocks else "(No supporting context provided.)"

        prompt_inputs = {
            "question": state.question,
            "context": context_text,
        }
        self._last_prompt = prompt_inputs

        try:
            answer = self._chain.invoke(prompt_inputs)
        except Exception as exc:
            answer = (
                "An error occurred while generating the answer. "
                "Please retry once the language model service is available."
            )
            state.metadata.setdefault("generation", {})
            state.metadata["generation"].update({"error": str(exc)})

        state.final_answer = answer
        state.metadata.setdefault("generation", {})
        state.metadata["generation"].update(
            {
                "used_documents": len(state.retrieved_documents),
                "context_chars": len(context_text),
            }
        )

        return state

    def _format_documents(self, documents: Iterable[Document]) -> List[str]:
        blocks: List[str] = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            source_label = metadata.get("source") or metadata.get("title") or f"doc-{idx}"
            block = doc.page_content.strip()
            if self.include_sources:
                block += f"\n[SOURCE: {source_label}]"
            blocks.append(block)

        return blocks

    @property
    def last_prompt(self) -> Optional[Dict[str, Any]]:
        """Expose the last prompt sent to the LLM for observability."""

        return self._last_prompt


__all__ = ["GenerationNode"]
