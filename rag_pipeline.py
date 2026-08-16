"""Retrieval and optional Groq grounded-answer generation."""

import os
from typing import Any

from langchain_groq import ChatGroq # type: ignore

from prompt import build_gemini_prompt, build_prompt


TOP_K = 4
DEFAULT_MODEL = "llama-3.3-70b-versatile"
NO_DOCUMENT_ANSWER = "I could not find this information in the uploaded documents."


class RAGPipeline:
    """Retrieve document chunks and optionally turn them into a grounded answer."""

    def __init__(self, vector_store, model_name: str | None = None):
        self.vector_store = vector_store
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.llm = None

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key:
            self.llm = ChatGroq(
                model=self.model_name,
                api_key=api_key,
                temperature=0,
            )

    @property
    def has_groq(self) -> bool:
        return self.llm is not None

    def retrieve(self, question: str, k: int = TOP_K):
        """Retrieve the most relevant document chunks."""
        return self.vector_store.similarity_search(question, k=k)

    @staticmethod
    def _source_list(docs) -> list[dict[str, Any]]:
        seen = set()
        sources = []
        for document in docs:
            metadata = document.metadata or {}
            source = str(metadata.get("source", "Document"))
            page = metadata.get("page", "?")
            key = (source, str(page))
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": source, "page": page})
        return sources

    def retrieve_context(self, question: str, k: int = TOP_K):
        """Return retrieved LangChain documents, context text, and source metadata."""
        docs = self.retrieve(question, k=k)
        if not docs:
            return [], "", []

        context = "\n\n".join(
            f"[{d.metadata.get('source', 'Document')} - page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
        return docs, context, self._source_list(docs)

    def answer_with_context(self, question: str):
        """Return grounded answer, sources, and raw retrieved context."""
        _, context, sources = self.retrieve_context(question)
        if not context:
            return NO_DOCUMENT_ANSWER, [], ""

        # Retrieval remains useful when only Gemini is configured. The final
        # Gemini prompt receives the source metadata and can explain the result.
        if self.llm is None:
            return "", sources, context

        response = self.llm.invoke(build_prompt(context, question))
        answer = str(getattr(response, "content", response)).strip()
        return answer or NO_DOCUMENT_ANSWER, sources, context

    def answer_general(
        self,
        question: str,
        knowledge_base_answer: str = "",
        sources: list[dict[str, Any]] | None = None,
        retrieved_context: str = "",
        history: list[dict[str, Any]] | None = None,
        retrieval_status: str = "",
    ) -> str:
        """Generate the same structured response with Groq as a fallback."""
        if self.llm is None:
            raise ValueError("GROQ_API_KEY is not configured for fallback generation.")

        response = self.llm.invoke(
            build_gemini_prompt(
                question=question,
                knowledge_base_answer=knowledge_base_answer,
                sources=sources or [],
                retrieved_context=retrieved_context,
                history=history or [],
                retrieval_status=retrieval_status,
            )
        )
        return str(getattr(response, "content", response)).strip()

    def answer(self, question: str):
        """Preserve the original project API: return answer and source metadata."""
        answer, sources, _ = self.answer_with_context(question)
        return answer, sources
    