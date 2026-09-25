"""Fit per-decision temperatures on calib, report metrics on test.

    python scripts/evaluate.py --model runs/nano-jev-v0.1
    python scripts/evaluate.py --model cross-encoder/ms-marco-MiniLM-L6-v2 --no-save   # zero-shot init baseline
    python scripts/evaluate.py --model runs/nano-jev-v0.1 --no-save --test-data data_heldout \
        --results-dir results/v0.1 --name heldout

Temperatures are always fitted on <data>/calib.jsonl. --test-data evaluates them unchanged
on another test set (out-of-distribution protocol).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanojev import model as M  # noqa: E402
from nanojev.report import (by_decision, decision_report, labels_of, read_jsonl,  # noqa: E402
                            render_table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/nano-jev-v0.1")
    ap.add_argument("--data", default="data", help="folder with calib.jsonl (and test.jsonl)")
    ap.add_argument("--test-data", help="folder with the test.jsonl to report on (default: --data)")
    ap.add_argument("--name", default="results", help="results file name (without extension)")
    ap.add_argument("--results-dir", help="where to write results (default: the model folder)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--no-save", action="store_true",
                    help="don't write anything into the model folder (use for released weights)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = M.load(args.model, device)
    calib = by_decision(read_jsonl(Path(args.data) / "calib.jsonl"))
    test = by_decision(read_jsonl(Path(args.test_data or args.data) / "test.jsonl"))

    report = {}
    for dec in sorted(test):
        c_logits = torch.stack(M.score(model, tok, calib[dec], args.max_length, device))
        t0 = time.time()
        t_logits = torch.stack(M.score(model, tok, test[dec], args.max_length, device))
        ms = 1000 * (time.time() - t0) / len(test[dec])
        report[dec] = decision_report(c_logits, labels_of(calib[dec]),
                                      t_logits, labels_of(test[dec]), ms)

    table = render_table(f"Nano-Jev — `{args.model}` on `{args.test_data or args.data}`", report)
    print(table)

    if not args.no_save:
        temps = {dec: r["temperature"] for dec, r in report.items()}
        (Path(args.model) / "calibration.json").write_text(json.dumps(temps, indent=2))
        print(f"\nsaved calibration.json to {args.model}")
    results_dir = Path(args.results_dir) if args.results_dir else None
    if results_dir is None and not args.no_save:
        results_dir = Path(args.model)
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{args.name}.json").write_text(json.dumps(report, indent=2))
        (results_dir / f"{args.name}.md").write_text(table + "\n", encoding="utf-8")
        print(f"saved {args.name}.json/.md to {results_dir}")


if __name__ == "__main__":
    main()
