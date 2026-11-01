"""Seed statistics, bootstrap CIs and paired significance tests (reviewers R2.3, R3.5).

Works only on predictions files (`*_preds.jsonl`, `cascade_preds.jsonl`), so it never
touches a GPU.

1) Across seeds: mean +- std of accuracy / ECE, plus a bootstrap 95% CI per seed.
    python scripts/stats.py seeds --name nanojev_test \
        --preds results/v2/nanojev/seed*/test_preds.jsonl --out results/v2/stats

2) Two systems on the same test set: accuracy difference, paired bootstrap CI of the
   difference, exact McNemar test.
    python scripts/stats.py compare --name nanojev_vs_llm_test \
        --a results/v2/nanojev/seed0/test_preds.jsonl \
        --b results/v2/baselines/llm_preds.jsonl --out results/v2/stats

3) Cascade at a threshold vs each model alone (paired, same examples).
    python scripts/stats.py cascade --name cascade_seed0 \
        --cascade results/v2/cascade/seed0/cascade_preds.jsonl \
        --cascade-json results/v2/cascade/seed0/cascade.json --out results/v2/stats

Bootstrap: 10,000 resamples of test examples (paired: the same resample for both systems).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanojev.calibration import ece  # noqa: E402
from revision_utils import load_preds, read_jsonl  # noqa: E402

B = 10_000


def correct_of(rows) -> np.ndarray:
    return np.array([int(np.argmax(r["probs"]) == r["label"]) for r in rows])


def bootstrap_ci(correct: np.ndarray, rng, b=B) -> tuple[float, float]:
    idx = rng.integers(0, len(correct), size=(b, len(correct)))
    accs = correct[idx].mean(1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def paired(ca: np.ndarray, cb: np.ndarray, rng, b=B) -> dict:
    """ca, cb: 0/1 correctness of systems A and B on the same examples."""
    assert len(ca) == len(cb), "systems were scored on different example sets"
    idx = rng.integers(0, len(ca), size=(b, len(ca)))
    diffs = ca[idx].mean(1) - cb[idx].mean(1)
    n01 = int(((ca == 1) & (cb == 0)).sum())  # A right, B wrong
    n10 = int(((ca == 0) & (cb == 1)).sum())  # A wrong, B right
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"n": int(len(ca)), "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
            "diff": float(ca.mean() - cb.mean()),
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "a_only_right": n01, "b_only_right": n10, "mcnemar_p": float(p)}


def cmd_seeds(args, rng):
    files = sorted({f for pat in args.preds for f in glob.glob(pat)})
    if not files:
        sys.exit(f"no files match {args.preds}")
    per_seed = {}
    for f in files:
        for dec, rows in load_preds(f).items():
            c = correct_of(rows)
            probs = np.array([r["probs"] for r in rows])
            labels = np.array([r["label"] for r in rows])
            per_seed.setdefault(dec, []).append(
                {"file": f, "n": len(c), "accuracy": float(c.mean()),
                 "ci95": bootstrap_ci(c, rng), "ece": ece(probs, labels)})
    out = {}
    for dec, runs in sorted(per_seed.items()):
        acc = np.array([r["accuracy"] for r in runs])
        e = np.array([r["ece"] for r in runs])
        out[dec] = {"n_seeds": len(runs), "n": runs[0]["n"],
                    "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "ece_mean": float(e.mean()), "ece_std": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
                    "ci95_halfwidth_mean": float(np.mean([(r["ci95"][1] - r["ci95"][0]) / 2 for r in runs])),
                    "runs": runs}
    lines = [f"## {args.name}: {len(files)} run(s)", "",
             "| decision | n | acc mean ± std | 95% CI half-width (bootstrap) | ECE mean ± std |",
             "|---|---|---|---|---|"]
    for dec, r in out.items():
        lines.append(f"| {dec} | {r['n']} | {r['acc_mean']:.3f} ± {r['acc_std']:.3f} "
                     f"| ±{r['ci95_halfwidth_mean']:.3f} | {r['ece_mean']:.3f} ± {r['ece_std']:.3f} |")
    return out, lines


def cmd_compare(args, rng):
    a, b = load_preds(args.a), load_preds(args.b)
    out = {}
    for dec in sorted(set(a) & set(b)):
        out[dec] = paired(correct_of(a[dec]), correct_of(b[dec]), rng)
    lines = [f"## {args.name}", f"A = `{args.a}`", f"B = `{args.b}`", "",
             "| decision | n | acc A | acc B | A − B | 95% CI of diff | A-only right | B-only right | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for dec, r in out.items():
        lines.append(f"| {dec} | {r['n']} | {r['acc_a']:.3f} | {r['acc_b']:.3f} | {r['diff']:+.3f} "
                     f"| [{r['diff_ci95'][0]:+.3f}, {r['diff_ci95'][1]:+.3f}] | {r['a_only_right']} "
                     f"| {r['b_only_right']} | {r['mcnemar_p']:.3g} |")
    return out, lines


def cmd_cascade(args, rng):
    rows = read_jsonl(args.cascade)
    report = json.loads(Path(args.cascade_json).read_text()) if args.cascade_json else {}
    groups = {}
    for r in rows:
        groups.setdefault((r["set"], r["decision"]), []).append(r)
    out, lines = {}, [f"## {args.name}", "",
                      "| set | decision | t | n | % escalated | cascade | Nano-Jev | LLM "
                      "| p (casc vs nano) | p (casc vs LLM) |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
    for (name, dec), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["idx"])
        t = args.threshold if args.threshold is not None else \
            report.get(f"{dec}@calib", {}).get("chosen_threshold", 0.8)
        y = np.array([r["label"] for r in rs])
        pn = np.array([r["nano_probs"] for r in rs])
        pl = np.array([r["llm_probs"] for r in rs])
        esc = pn.max(1) < t
        pred = np.where(esc, pl.argmax(1), pn.argmax(1))
        cc, cn, cl = (pred == y).astype(int), (pn.argmax(1) == y).astype(int), (pl.argmax(1) == y).astype(int)
        vs_n, vs_l = paired(cc, cn, rng), paired(cc, cl, rng)
        out[f"{dec}@{name}"] = {"threshold": t, "escalated": float(esc.mean()),
                                "vs_nano": vs_n, "vs_llm": vs_l}
        lines.append(f"| {name} | {dec} | {t:.2f} | {len(y)} | {esc.mean():.1%} | {cc.mean():.3f} "
                     f"| {cn.mean():.3f} | {cl.mean():.3f} | {vs_n['mcnemar_p']:.3g} | {vs_l['mcnemar_p']:.3g} |")
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seeds", "compare", "cascade"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True, help="output file stem")
        p.add_argument("--out", default="results/v2/stats")
        p.add_argument("--seed", type=int, default=0)
    sub.choices["seeds"].add_argument("--preds", nargs="+", required=True, help="files or globs")
    sub.choices["compare"].add_argument("--a", required=True)
    sub.choices["compare"].add_argument("--b", required=True)
    sub.choices["cascade"].add_argument("--cascade", required=True)
    sub.choices["cascade"].add_argument("--cascade-json", help="cascade.json (for calib-chosen thresholds)")
    sub.choices["cascade"].add_argument("--threshold", type=float, help="override the calib-chosen threshold")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out, lines = {"seeds": cmd_seeds, "compare": cmd_compare, "cascade": cmd_cascade}[args.cmd](args, rng)
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    (od / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
