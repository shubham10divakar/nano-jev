"""Baselines on the same calib / test split as Nano-Jev.

- reranker: a cross-encoder relevance model scores (query or claim, context); a logistic
  regression fitted on calib maps that single score to the decision's options.
- llm: a prompted instruction LLM; option probabilities are read from the answer-letter
  logits of one forward pass (no generation).

Both then get the same temperature scaling and metrics as Nano-Jev.

    python scripts/baselines.py --which all
    python scripts/baselines.py --which llm --limit 50     # quick check
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanojev.report import (by_decision, decision_report, labels_of, read_jsonl,  # noqa: E402
                            render_table, save)
from nanojev.schema import subject_of  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# What each option means, spelled out for the zero-shot LLM.
OPTION_TEXT = {
    "relevance": ["irrelevant: the passage does not help answer the query",
                  "partially relevant: the passage is related or gives part of the information "
                  "needed, but does not itself contain the answer",
                  "directly answers: the passage contains the answer to the query"],
    "sufficient": ["yes", "no"],
    "grounded": ["yes", "no"],
}


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- reranker baseline

@torch.no_grad()
def reranker_scores(model, tok, examples, max_length=512, batch_size=32) -> np.ndarray:
    scores = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        enc = tok([subject_of(ex) for ex in chunk], [ex["state"] for ex in chunk],
                  truncation="only_second", max_length=max_length, padding=True,
                  return_tensors="pt").to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            scores.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(scores)


def run_reranker(name, calib, test) -> dict:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
    report = {}
    for dec in sorted(test):
        c = reranker_scores(model, tok, calib[dec])
        t0 = time.time()
        t = reranker_scores(model, tok, test[dec])
        ms = 1000 * (time.time() - t0) / len(test[dec])
        lr = LogisticRegression(max_iter=1000).fit(c[:, None], labels_of(calib[dec]).numpy())
        to_logits = lambda s: torch.tensor(np.log(np.clip(lr.predict_proba(s[:, None]),  # noqa: E731
                                                          1e-12, None)), dtype=torch.float32)
        report[dec] = decision_report(to_logits(c), labels_of(calib[dec]),
                                      to_logits(t), labels_of(test[dec]), ms)
    del model
    free_gpu()
    return report


# ---------------------------------------------------------------- prompted LLM baseline

def llm_prompt(tok, ex, max_state_tokens: int) -> str:
    state_ids = tok(ex["state"], add_special_tokens=False)["input_ids"][:max_state_tokens]
    state = tok.decode(state_ids)
    letters = [chr(65 + i) for i in range(len(ex["options"]))]
    options = "\n".join(f"{l}. {o}" for l, o in zip(letters, OPTION_TEXT[ex["decision"]]))
    content = (
        "You are a precise judge inside a retrieval-augmented question answering system.\n\n"
        f"{ex['question']}\n\n<context>\n{state}\n</context>\n\n"
        f"Options:\n{options}\n\nAnswer with the letter of the correct option only."
    )
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def llm_logits(model, tok, examples, batch_size=8, max_state_tokens=900) -> torch.Tensor:
    k = len(examples[0]["options"])
    letter_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(k)]
    out = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        enc = tok([llm_prompt(tok, ex, max_state_tokens) for ex in chunk], return_tensors="pt",
                  padding=True).to(DEVICE)
        logits = model(**enc).logits[:, -1, :]  # left padding: last position is the answer slot
        out.append(logits[:, letter_ids].float().cpu())
    return torch.cat(out)


def run_llm(name, calib, test) -> dict:
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEVICE).eval()
    report = {}
    for dec in sorted(test):
        c = llm_logits(model, tok, calib[dec])
        t0 = time.time()
        t = llm_logits(model, tok, test[dec])
        ms = 1000 * (time.time() - t0) / len(test[dec])
        report[dec] = decision_report(c, labels_of(calib[dec]), t, labels_of(test[dec]), ms)
        print(f"  {dec}: acc {report[dec]['calibrated']['accuracy']:.3f}", flush=True)
    del model
    free_gpu()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["reranker", "llm", "all"])
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--limit", type=int, default=0, help="max examples per decision and split")
    args = ap.parse_args()

    calib = by_decision(read_jsonl(Path(args.data) / "calib.jsonl"))
    test = by_decision(read_jsonl(Path(args.data) / "test.jsonl"))
    if args.limit:
        calib = {k: v[: args.limit] for k, v in calib.items()}
        test = {k: v[: args.limit] for k, v in test.items()}

    runs = []
    if args.which in ("reranker", "all"):
        runs.append(("reranker", args.reranker, run_reranker))
    if args.which in ("llm", "all"):
        runs.append(("llm", args.llm, run_llm))

    for kind, name, fn in runs:
        print(f"== {kind}: {name}", flush=True)
        report = fn(name, calib, test)
        table = render_table(f"{kind} baseline — `{name}`", report)
        print(table, flush=True)
        if not args.limit:
            save(report, Path(args.out) / f"{kind}.json")
            (Path(args.out) / f"{kind}.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
