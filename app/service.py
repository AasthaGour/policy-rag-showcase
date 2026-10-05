from pathlib import Path
from typing import Dict, List, Optional

from .chunking import chunk_document
from .domain import RAGAnswer
from .generation import generate_grounded_answer
from .loader import load_directory
from .router import route_query
from .vector_store import InMemoryVectorStore


class PolicyRAGService:
    def __init__(self, store: InMemoryVectorStore, min_score: float = 0.12):
        self.store = store
        self.min_score = min_score

    def ingest_directory(self, directory: Path) -> int:
        chunks = [
            chunk
            for document in load_directory(directory)
            for chunk in chunk_document(document)
        ]
        self.store.add(chunks)
        return len(chunks)

    def answer(
        self, query: str, geo: Optional[str] = None, top_k: int = 3, debug: bool = False
    ) -> RAGAnswer:
        route = route_query(query)
        if route == "invalid":
            return RAGAnswer("Please enter a question.", route, [], False)
        if route == "greeting":
            return RAGAnswer("Hello! How can I help with company policy?", route, [], True)
        if route == "bot_behavior":
            return RAGAnswer(
                "I am a policy assistant. I answer using the indexed policy documents.",
                route,
                [],
                True,
            )

        metadata_filter: Optional[Dict[str, str]] = {"geo": geo} if geo else None
        retrieved = self.store.search(query, top_k=top_k, metadata_filter=metadata_filter)
        relevant = [result for result in retrieved if result.score >= self.min_score]
        generated = generate_grounded_answer(query, relevant)

        citations: List[Dict[str, object]] = [
            {
                "source": result.chunk.metadata.get("source"),
                "title": result.chunk.metadata.get("title"),
                "geo": result.chunk.metadata.get("geo"),
                "chunk_id": result.chunk.chunk_id,
                "score": round(result.score, 4),
            }
            for result in relevant
        ]
        debug_data = None
        if debug:
            debug_data = {
                "filter": metadata_filter,
                "retrieved": len(retrieved),
                "accepted": len(relevant),
                "minimum_score": self.min_score,
            }

        if not generated:
            return RAGAnswer(
                "I do not have sufficient policy evidence to answer that question.",
                route,
                citations,
                False,
                debug_data,
            )
        return RAGAnswer(generated, route, citations, True, debug_data)

