"""How much does the 512-token limit cut, and does it hurt? (reviewer R1.4)

For every test example, the longest (question+option, state) pair is tokenized exactly as
Nano-Jev does (truncation="only_second"). Reported per decision / source:
  * % of inputs longer than max_length, median and 95th-percentile length;
  * multi-passage states (sufficiency): % with a passage partly cut, fully dropped, and
    % where a GOLD passage (one labelled partially/directly relevant for the same query in
    the relevance rows of the same file) is cut or dropped;
  * with --preds: accuracy on truncated vs. non-truncated inputs.

--write-balanced DIR writes a copy of calib/test where each passage of a multi-passage state
is cut to an equal share of the token budget, so every passage survives (bonus variant).
Evaluate it with evaluate.py --test-data DIR (temperatures still from data_clean calib).

    python scripts/truncation_analysis.py --model runs/v2/nanojev/seed0 --data data_clean \
        --preds results/v2/nanojev/seed0/test_preds.jsonl --out results/v2/truncation/test.json \
        --write-balanced data_clean_balanced
    python scripts/truncation_analysis.py --model runs/v2/nanojev/seed0 --data data_heldout \
        --preds results/v2/nanojev/seed0/heldout_preds.jsonl --out results/v2/truncation/heldout.json \
        --write-balanced data_heldout_balanced
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanojev.report import by_decision  # noqa: E402
from nanojev.schema import subject_of  # noqa: E402
from revision_utils import load_preds, read_jsonl, write_jsonl  # noqa: E402

LINE = re.compile(r"^\[(\d+)\] ")


def budget(tok, ex, max_length):
    """Tokens available for the state with the longest option's first segment."""
    first = max(len(tok(f"question: {ex['question']} option: {o}", add_special_tokens=False)["input_ids"])
                for o in ex["options"])
    return max_length - 3 - first


def analyse(tok, exs, max_length, gold_states):
    rows = []
    for ex in exs:
        b = budget(tok, ex, max_length)
        n_state = len(tok(ex["state"], add_special_tokens=False)["input_ids"])
        r = {"truncated": n_state > b, "len": n_state + max_length - b}
        lines = ex["state"].split("\n")
        if len(lines) > 1 and all(LINE.match(x) for x in lines):
            used, partly, dropped, gold_cut = 0, 0, 0, 0
            gold = gold_states.get(subject_of(ex), set())
            for x in lines:
                n = len(tok(x + "\n", add_special_tokens=False)["input_ids"])
                cut = used + n > b
                if used >= b:
                    dropped += 1
                elif cut:
                    partly += 1
                if cut and LINE.sub("", x) in gold:
                    gold_cut += 1
                used += n
            r.update(multi=True, partly=partly > 0, dropped=dropped > 0, gold_cut=gold_cut > 0)
        rows.append(r)
    return rows


def balanced(tok, ex, max_length):
    lines = ex["state"].split("\n")
    if len(lines) < 2 or not all(LINE.match(x) for x in lines):
        return ex
    share = budget(tok, ex, max_length) // len(lines) - 1
    cut = []
    for x in lines:
        ids = tok(x, add_special_tokens=False)["input_ids"]
        cut.append(x if len(ids) <= share else tok.decode(ids[:share]))
    return {**ex, "state": "\n".join(cut)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="any Nano-Jev folder (for its tokenizer)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", help="test predictions from evaluate.py on the same data")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-balanced")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    test = read_jsonl(Path(args.data) / "test.jsonl")
    gold_states = defaultdict(set)
    for ex in test:
        if ex["decision"] == "relevance" and ex["label"] in (1, 2):
            gold_states[subject_of(ex)].add(ex["state"])
    preds = load_preds(args.preds) if args.preds else {}

    report = {}
    for dec, exs in sorted(by_decision(test).items()):
        rows = analyse(tok, exs, args.max_length, gold_states)
        correct = None
        if dec in preds:
            correct = [int(np.argmax(p["probs"]) == p["label"]) for p in preds[dec]]
        by_src = defaultdict(list)
        for i, (ex, r) in enumerate(zip(exs, rows)):
            by_src[ex["source"]].append((r, None if correct is None else correct[i]))
        for src, items in sorted(by_src.items()):
            rs = [r for r, _ in items]
            lens = np.array([r["len"] for r in rs])
            entry = {"n": len(rs), "pct_truncated": float(np.mean([r["truncated"] for r in rs])),
                     "len_median": float(np.median(lens)), "len_p95": float(np.percentile(lens, 95))}
            if any(r.get("multi") for r in rs):
                m = [r for r in rs if r.get("multi")]
                entry.update(pct_passage_partly_cut=float(np.mean([r["partly"] for r in m])),
                             pct_passage_dropped=float(np.mean([r["dropped"] for r in m])),
                             pct_gold_cut=float(np.mean([r["gold_cut"] for r in m])))
            if correct is not None:
                t = [c for r, c in items if r["truncated"]]
                nt = [c for r, c in items if not r["truncated"]]
                entry.update(acc_truncated=float(np.mean(t)) if t else None, n_truncated=len(t),
                             acc_not_truncated=float(np.mean(nt)) if nt else None)
            report[f"{dec}/{src}"] = entry
            print(f"{dec}/{src}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in entry.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    if args.write_balanced:
        out = Path(args.write_balanced)
        for name in ("calib.jsonl", "test.jsonl"):
            src = Path(args.data) / name
            if src.exists():
                write_jsonl(out / name, [balanced(tok, ex, args.max_length) for ex in read_jsonl(src)])
        print(f"wrote balanced-truncation copy to {out}")


if __name__ == "__main__":
    main()
