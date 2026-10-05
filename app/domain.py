from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Document:
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    route: str
    citations: List[Dict[str, object]]
    grounded: bool
    debug: Optional[Dict[str, object]] = None

