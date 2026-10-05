import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Free deterministic embedding baseline for a clone-and-run demo.

    This is intentionally lightweight rather than a replacement for a production
    semantic model. The provider interface makes swapping models straightforward.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

