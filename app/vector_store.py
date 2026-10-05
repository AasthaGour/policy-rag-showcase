from typing import Dict, List, Optional, Sequence, Tuple

from .domain import Chunk, SearchResult
from .embeddings import EmbeddingProvider, cosine_similarity


class InMemoryVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self._items: List[Tuple[Chunk, List[float]]] = []

    @property
    def size(self) -> int:
        return len(self._items)

    def add(self, chunks: Sequence[Chunk]) -> None:
        vectors = self.embedding_provider.embed([chunk.text for chunk in chunks])
        self._items.extend(zip(chunks, vectors))

    def search(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        query_vector = self.embedding_provider.embed([query])[0]
        scored: List[SearchResult] = []

        for chunk, vector in self._items:
            if metadata_filter and any(
                chunk.metadata.get(key, "").lower() != value.lower()
                for key, value in metadata_filter.items()
            ):
                continue
            scored.append(
                SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

