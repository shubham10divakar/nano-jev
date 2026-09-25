"""Inference API: typed decisions with calibrated probabilities."""

import json
from pathlib import Path

import torch

from . import model as M
from .schema import DECISIONS, format_passages


class Decider:
    def __init__(self, model, tok, max_length: int, temperatures: dict[str, float], device):
        self.model, self.tok, self.max_length = model, tok, max_length
        self.temperatures, self.device = temperatures, device

    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None) -> "Decider":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, tok = M.load(path, device)
        cfg_path = Path(path) / "nanojev_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        cal_path = Path(path) / "calibration.json"
        temps = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        return cls(model, tok, cfg.get("max_length", 512), temps, device)

    def decide_many(self, items: list[dict]) -> list[dict[str, float]]:
        """items: {question, options, state, decision?} -> [{option: probability}]."""
        logits = M.score(self.model, self.tok, items, self.max_length, self.device)
        out = []
        for item, lg in zip(items, logits):
            t = self.temperatures.get(item.get("decision", ""), 1.0)
            probs = torch.softmax(lg / t, dim=0).tolist()
            out.append(dict(zip(item["options"], probs)))
        return out

    def decide(self, question: str, options: list[str], state: str,
               decision: str | None = None) -> dict[str, float]:
        return self.decide_many([{"question": question, "options": options, "state": state,
                                  "decision": decision or ""}])[0]

    # Convenience wrappers for the built-in decisions --------------------------------

    def _builtin(self, decision: str, state: str, **fields) -> dict:
        d = DECISIONS[decision]
        return {"decision": decision, "question": d.question.format(**fields),
                "options": list(d.options), "state": state}

    def relevance(self, query: str, passages: list[str]) -> list[dict[str, float]]:
        return self.decide_many([self._builtin("relevance", p, query=query) for p in passages])

    def sufficient(self, query: str, passages: list[str]) -> dict[str, float]:
        state = format_passages([("passage", p) for p in passages])
        return self.decide_many([self._builtin("sufficient", state, query=query)])[0]

    def grounded(self, claim: str, context: str) -> dict[str, float]:
        return self.decide_many([self._builtin("grounded", context, claim=claim)])[0]
