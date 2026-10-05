import re
from typing import List, Sequence, Set

from .domain import SearchResult


STOPWORDS = {
    "a", "an", "and", "are", "for", "how", "i", "in", "is", "me", "of",
    "on", "policy", "the", "to", "what", "with",
}


def _tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


def generate_grounded_answer(query: str, results: Sequence[SearchResult]) -> str:
    """Extract the best evidence sentences; never invent unsupported content."""
    query_tokens = _tokens(query)
    candidates = []
    for result in results:
        for sentence in re.split(r"(?<=[.!?])\s+", result.chunk.text):
            overlap = len(query_tokens & _tokens(sentence))
            # A single shared generic word is weak evidence and can create a
            # plausible-looking but unsupported answer. Require two query terms.
            if overlap >= 2:
                candidates.append((overlap, result.score, sentence.strip()))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: List[str] = []
    for _, _, sentence in candidates:
        if sentence and sentence not in selected:
            selected.append(sentence)
        if len(selected) == 2:
            break
    return " ".join(selected)
