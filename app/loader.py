from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .domain import Document


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Parse a tiny key:value frontmatter block without an extra YAML dependency."""
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    metadata: Dict[str, str] = {}
    for line in text[4:end].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            metadata[key.strip().lower()] = value.strip()
    return metadata, text[end + 5 :].strip()


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        raw_text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to ingest PDF files") from exc
        reader = PdfReader(str(path))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    metadata, text = _parse_frontmatter(raw_text)
    metadata.setdefault("source", path.name)
    metadata.setdefault("title", path.stem.replace("_", " ").title())
    return Document(text=text, metadata=metadata)


def load_directory(directory: Path) -> List[Document]:
    supported = {".md", ".txt", ".pdf"}
    paths: Iterable[Path] = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in supported
    )
    return [load_document(path) for path in paths]

