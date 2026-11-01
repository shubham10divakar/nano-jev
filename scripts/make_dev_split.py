"""Carve a development set out of the TRAINING pool (reviewer R3.2).

The v1.0 protocol selected the best epoch on a slice of calib.jsonl and then fitted the
temperatures on the same calib.jsonl. The revised protocol keeps calib untouched until
temperature fitting:

    train.jsonl  -> train (95%) + dev (5%)     dev: checkpoint selection only
    calib.jsonl  -> unchanged                  temperatures / thresholds / mappings only
    test.jsonl   -> unchanged                  reporting only

Examples are grouped by (source, query/claim) before splitting, so all HotpotQA rows of
one question (4 relevance + 2 sufficiency) land on the same side. The split is stratified
by source.

    python scripts/make_dev_split.py --data data --out data_clean
"""

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanojev.schema import subject_of  # noqa: E402
from revision_utils import read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_clean")
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = read_jsonl(Path(args.data) / "train.jsonl")
    groups = defaultdict(list)  # (source, subject) -> example indices
    for i, ex in enumerate(train):
        groups[(ex["source"], subject_of(ex))].append(i)

    rng = random.Random(args.seed)
    by_source = defaultdict(list)
    for key in groups:
        by_source[key[0]].append(key)
    dev_idx = set()
    for src, keys in sorted(by_source.items()):
        keys.sort()
        rng.shuffle(keys)
        for key in keys[: round(args.dev_frac * len(keys))]:
            dev_idx.update(groups[key])

    new_train = [ex for i, ex in enumerate(train) if i not in dev_idx]
    dev = [ex for i, ex in enumerate(train) if i in dev_idx]

    out = Path(args.out)
    write_jsonl(out / "train.jsonl", new_train)
    write_jsonl(out / "dev.jsonl", dev)
    for name in ("calib.jsonl", "test.jsonl"):
        shutil.copyfile(Path(args.data) / name, out / name)

    for name, rows in (("train", new_train), ("dev", dev),
                       ("calib", read_jsonl(out / "calib.jsonl")),
                       ("test", read_jsonl(out / "test.jsonl"))):
        c = Counter((r["decision"], r["source"]) for r in rows)
        print(f"{name:<6} {len(rows):>6}  " + "  ".join(f"{d}/{s}={n}" for (d, s), n in sorted(c.items())))

    # Leakage check: no query/claim may appear in both train and dev.
    tr = {(ex["source"], subject_of(ex)) for ex in new_train}
    dv = {(ex["source"], subject_of(ex)) for ex in dev}
    assert not tr & dv, "train/dev overlap"
    print(f"\nwrote {out}/{{train,dev,calib,test}}.jsonl  (no train/dev query overlap)")


if __name__ == "__main__":
    main()
