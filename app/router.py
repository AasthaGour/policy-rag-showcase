import re


GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|good morning|good evening|bye|goodbye)[!. ]*$",
    re.IGNORECASE,
)


def route_query(query: str) -> str:
    cleaned = query.strip().lower()
    if not cleaned:
        return "invalid"
    if GREETING_PATTERN.match(query):
        return "greeting"
    if any(phrase in cleaned for phrase in ("are you ai", "who are you", "what can you do")):
        return "bot_behavior"
    return "knowledge_search"

