"""Download datasets and write train / calib / test JSONL files.

    python scripts/prepare_data.py --preset default
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanojev.data import build_all  # noqa: E402

PRESETS = {
    # hotpot_* counts questions (each gives ~4 relevance + 2 sufficient examples)
    "smoke": dict(hotpot_train=200, squad_train=200, mnli_train=200,
                  hotpot_eval=50, squad_eval=50, mnli_eval=50),
    "default": dict(hotpot_train=6000, squad_train=6000, mnli_train=8000,
                    hotpot_eval=500, squad_eval=500, mnli_eval=500),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default", choices=PRESETS)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = build_all(PRESETS[args.preset], seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts = Counter((ex["decision"], ex["options"][ex["label"]]) for ex in examples)
        print(f"{name}: {len(examples)} examples")
        for (dec, lab), c in sorted(counts.items()):
            print(f"    {dec:<11} {lab:<20} {c}")


if __name__ == "__main__":
    main()
