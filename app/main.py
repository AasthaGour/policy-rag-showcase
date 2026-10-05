import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .embeddings import LocalHashEmbeddingProvider
from .service import PolicyRAGService
from .vector_store import InMemoryVectorStore


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    geo: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)
    debug: bool = False


def build_service() -> PolicyRAGService:
    store = InMemoryVectorStore(LocalHashEmbeddingProvider())
    service = PolicyRAGService(store, min_score=float(os.getenv("RAG_MIN_SCORE", "0.12")))
    data_dir = Path(os.getenv("RAG_DATA_DIR", "data"))
    if data_dir.exists():
        service.ingest_directory(data_dir)
    return service


service = build_service()
app = FastAPI(
    title="Policy RAG Showcase",
    version="1.0.0",
    description="Grounded policy retrieval with routing, metadata filters and citations.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": service.store.size}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return asdict(
        service.answer(
            request.query,
            geo=request.geo,
            top_k=request.top_k,
            debug=request.debug,
        )
    )

