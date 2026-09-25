"""Decision definitions: question templates and option sets for each typed decision."""

from dataclasses import dataclass

SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class Decision:
    name: str
    kind: str  # "choice" | "noul" | "score"
    question: str  # template, formatted with the example's fields
    options: tuple[str, ...]


DECISIONS: dict[str, Decision] = {
    "relevance": Decision(
        name="relevance",
        kind="score",
        question="How relevant is this passage to the query: {query}",
        options=("irrelevant", "partially relevant", "directly answers"),
    ),
    "sufficient": Decision(
        name="sufficient",
        kind="noul",
        question="Does the context contain enough information to answer: {query}",
        options=("yes", "no"),
    ),
    "grounded": Decision(
        name="grounded",
        kind="noul",
        question="Is this claim supported by the context? Claim: {claim}",
        options=("yes", "no"),
    ),
}


def format_passages(passages: list[tuple[str, str]]) -> str:
    """Render (title, text) passages as one state string."""
    return "\n".join(f"[{i + 1}] {title}: {text}" for i, (title, text) in enumerate(passages))
