"""Nano-Jev: a tiny Jev-style typed decision model for RAG.

    import nanojev
    d = nanojev.load()          # selected model (default: v0.1), downloaded on first use
    d = nanojev.load("v0.1")    # a specific version
    nanojev.list_models()       # versions on the Hub + which are downloaded
"""

__version__ = "0.1.0"

from .registry import (DEFAULT_REPO, DEFAULT_VERSION, get_selected, list_models,  # noqa: E402
                       set_selected)
from .schema import DECISIONS, SCHEMA_VERSION  # noqa: E402

__all__ = ["Decider", "load", "list_models", "get_selected", "set_selected", "__version__",
           "DEFAULT_REPO", "DEFAULT_VERSION", "DECISIONS", "SCHEMA_VERSION"]


def __getattr__(name):
    # Imported lazily so `python -m nanojev list` doesn't pay for torch/transformers.
    if name == "Decider":
        from .decider import Decider
        return Decider
    raise AttributeError(f"module 'nanojev' has no attribute {name!r}")


def load(model: str | None = None, device: str | None = None):
    """Load a Decider. model: version ("v0.1"), Hub repo ("user/repo@v0.1") or local folder."""
    from .decider import Decider
    return Decider.from_pretrained(model, device=device)
