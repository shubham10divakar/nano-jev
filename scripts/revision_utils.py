"""Shared helpers for the paper-revision experiments (per-example predictions, I/O).

Every evaluation in the revision writes a *predictions file* next to its metrics JSON, so
significance tests, bootstrap CIs, cascades and tables can all be recomputed offline from
the same numbers. One JSON line per example:

    {"decision": "grounded", "idx": 17, "source": "mnli", "label": 0,
     "logits": [2.1, -0.4], "temperature": 1.38, "probs": [0.86, 0.14]}

`idx` is the example's position within its decision in the test file (the order of
`by_decision(read_jsonl(test.jsonl))`), so files from different systems on the same test
set line up by (decision, idx).
"""

import json
import random
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pred_rows(decision: str, examples: list[dict], logits: torch.Tensor,
              temperature: float) -> list[dict]:
    probs = torch.softmax(logits / temperature, dim=1)
    return [{"decision": decision, "idx": i, "source": ex.get("source", ""),
             "label": int(ex["label"]),
             "logits": [round(float(x), 5) for x in lg],
             "temperature": float(temperature),
             "probs": [round(float(x), 5) for x in p]}
            for i, (ex, lg, p) in enumerate(zip(examples, logits, probs))]


def per_class(logits, labels, options, temperature):
    pred = torch.softmax(logits / temperature, dim=1).argmax(1).numpy()
    p, r, f, n = precision_recall_fscore_support(labels.numpy(), pred,
                                                 labels=list(range(len(options))), zero_division=0)
    return {opt: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                  "support": int(n[i])} for i, opt in enumerate(options)}


def load_preds(path) -> dict[str, list[dict]]:
    """predictions file -> {decision: rows sorted by idx}."""
    out = {}
    for r in read_jsonl(path):
        out.setdefault(r["decision"], []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["idx"])
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
