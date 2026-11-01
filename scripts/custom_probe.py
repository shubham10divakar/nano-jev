"""Supervised reference for the custom-option evaluation (R1.2, R3.3).

The 1-pass classifier has fixed positional heads, so it cannot answer an unseen option set
zero-shot. To give the zero-shot K-pass numbers a ceiling, this fits a logistic-regression
probe on frozen CLS features of an encoder, using the task's calib.jsonl labels (300 by
default), and reports test accuracy. It is a *supervised* reference, not a zero-shot one.

    python scripts/custom_probe.py --encoder runs/v2/onepass/seed0 \
        --tasks data_custom/agnews data_custom/yahoo data_custom/emotion \
        --out results/v2/custom/probe_onepass.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_utils import read_jsonl  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def features(model, tok, rows, batch_size=32):
    out = []
    for i in range(0, len(rows), batch_size):
        b = rows[i : i + batch_size]
        enc = tok([r["question"] for r in b], [r["state"] for r in b], truncation="only_second",
                  max_length=512, padding=True, return_tensors="pt").to(DEVICE)
        out.append(model(**enc).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(DEVICE).eval()
    res = {}
    for task in args.tasks:
        cal, test = read_jsonl(Path(task) / "calib.jsonl"), read_jsonl(Path(task) / "test.jsonl")
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(features(model, tok, cal), [r["label"] for r in cal])
        acc = float((lr.predict(features(model, tok, test)) == np.array([r["label"] for r in test])).mean())
        res[Path(task).name] = {"accuracy": acc, "n_train_labels": len(cal), "n_test": len(test),
                                "k": len(test[0]["options"])}
        print(f"{Path(task).name}: probe acc {acc:.3f} (K={len(test[0]['options'])}, {len(cal)} labels)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
