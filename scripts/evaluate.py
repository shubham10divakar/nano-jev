"""Fit per-decision temperatures on calib, report metrics on test.

    python scripts/evaluate.py --model runs/nano-jev-v0.1
    python scripts/evaluate.py --model cross-encoder/ms-marco-MiniLM-L6-v2 --no-save   # zero-shot init baseline
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
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--no-save", action="store_true", help="don't write calibration.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = M.load(args.model, device)
    calib = by_decision(read_jsonl(Path(args.data) / "calib.jsonl"))
    test = by_decision(read_jsonl(Path(args.data) / "test.jsonl"))

    report = {}
    for dec in sorted(test):
        c_logits = torch.stack(M.score(model, tok, calib[dec], args.max_length, device))
        t0 = time.time()
        t_logits = torch.stack(M.score(model, tok, test[dec], args.max_length, device))
        ms = 1000 * (time.time() - t0) / len(test[dec])
        report[dec] = decision_report(c_logits, labels_of(calib[dec]),
                                      t_logits, labels_of(test[dec]), ms)

    table = render_table(f"Nano-Jev — `{args.model}`", report)
    print(table)

    if not args.no_save:
        out = Path(args.model)
        temps = {dec: r["temperature"] for dec, r in report.items()}
        (out / "calibration.json").write_text(json.dumps(temps, indent=2))
        (out / "results.json").write_text(json.dumps(report, indent=2))
        (out / "results.md").write_text(table + "\n", encoding="utf-8")
        print(f"\nsaved calibration.json and results to {out}")


if __name__ == "__main__":
    main()
