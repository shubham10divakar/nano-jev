"""Try the Decider on a hand-written RAG example.

    python scripts/demo.py                       # selected model (default v0.1, downloaded on first use)
    python scripts/demo.py --model runs/nano-jev-dev
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanojev import Decider  # noqa: E402


def show(d):
    return "  ".join(f"{k}={v:.2f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="version, Hub repo or local folder (default: selected)")
    args = ap.parse_args()
    d = Decider.from_pretrained(args.model)
    print(f"model version {d.version}\n")

    query = "In which year was the university attended by Ada Lovelace's tutor founded?"
    passages = [
        "Augustus De Morgan tutored Ada Lovelace in mathematics. He was the first professor "
        "of mathematics at University College London.",
        "University College London was founded in 1826 as London University.",
        "The Analytical Engine was a proposed mechanical general-purpose computer.",
    ]

    print("query:", query)
    for p, r in zip(passages, d.relevance(query, passages)):
        print(f"  relevance  {show(r)}  | {p[:60]}...")
    print("sufficient (all)        ", show(d.sufficient(query, passages)))
    print("sufficient (first only) ", show(d.sufficient(query, passages[:1])))
    print("grounded 'founded 1826' ", show(d.grounded("UCL was founded in 1826.", passages[1])))
    print("grounded 'founded 1900' ", show(d.grounded("UCL was founded in 1900.", passages[1])))

    # Dynamic options: a label set the model never saw in training.
    print("custom choice           ", show(d.decide(
        "Which topic is this passage about?",
        ["computing history", "cooking", "football"],
        passages[2])))


if __name__ == "__main__":
    main()
