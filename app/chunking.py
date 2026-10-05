import hashlib
from typing import List

from .domain import Chunk, Document


def chunk_document(
    document: Document, chunk_size: int = 120, overlap: int = 25
) -> List[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    words = document.text.split()
    chunks: List[Chunk] = []
    step = chunk_size - overlap

    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        text = " ".join(chunk_words)
        source = document.metadata.get("source", "document")
        digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:12]
        metadata = {**document.metadata, "chunk_index": str(index)}
        chunks.append(Chunk(chunk_id=digest, text=text, metadata=metadata))
        if start + chunk_size >= len(words):
            break
    return chunks

