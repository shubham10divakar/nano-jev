"""Confidence cascade: accept Nano-Jev when it is confident, escalate the rest to an LLM.

For each decision, both models are temperature-calibrated on <data>/calib.jsonl. Then for
thresholds t, examples where Nano-Jev's top probability >= t keep its answer and the rest
use the LLM's answer. Reports accuracy vs the fraction escalated on each test set.

    python scripts/cascade.py --model runs/nano-jev-v1.0 --test-data data data_heldout

Revision (all three decisions, clean-protocol model):
    python scripts/cascade.py --model runs/v2/nanojev/seed0 --data data_clean \
        --test-data data_clean data_heldout --decisions relevance sufficient grounded \
        --out results/v2/cascade/seed0

Also written: for every test set, the same sweep with uncalibrated Nano-Jev (tau = 1,
threshold also re-chosen on calib) as "rows_raw" (temperature ablation), and
cascade_preds.jsonl with both models' calibrated probabilities per example, so stats.py
can run paired tests on any threshold.

Latency: ms/decision = Nano-Jev batched ms/decision (all examples)
                       + escalated fraction x LLM batched ms/decision, per decision type.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import llm_logits  # noqa: E402
from revision_utils import write_jsonl  # noqa: E402
from nanojev import model as M  # noqa: E402
from nanojev.calibration import fit_temperature  # noqa: E402
from nanojev.report import by_decision, labels_of, read_jsonl  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]


def timed(fn, *args):
    t0 = time.time()
    out = fn(*args)
    return out, 1000 * (time.time() - t0)


def sweep(nano_p, llm_p, labels):
    conf, nano_pred, llm_pred = nano_p.max(1).values, nano_p.argmax(1), llm_p.argmax(1)
    rows = []
    for t in THRESHOLDS:
        escalate = conf < t
        pred = torch.where(escalate, llm_pred, nano_pred)
        rows.append({"threshold": t, "escalated": float(escalate.float().mean()),
                     "accuracy": float((pred == labels).float().mean())})
    return rows


def pick_threshold(rows, tolerance=0.002):
    """Threshold with the best calib accuracy; near-ties go to the one escalating less."""
    best = max(r["accuracy"] for r in rows)
    ok = [r for r in rows if r["accuracy"] >= best - tolerance]
    return min(ok, key=lambda r: r["escalated"])["threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/nano-jev-dev")
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="data", help="folder with calib.jsonl")
    ap.add_argument("--test-data", nargs="+", default=["data"])
    ap.add_argument("--decisions", nargs="+", default=["grounded"])
    ap.add_argument("--out", default="results/cascade")
    args = ap.parse_args()

    calib = by_decision(read_jsonl(Path(args.data) / "calib.jsonl"))
    tests = {d: by_decision(read_jsonl(Path(d) / "test.jsonl")) for d in args.test_data}

    # Score everything with Nano-Jev first, then load the LLM (both don't fit in VRAM at once).
    nano, tok = M.load(args.model, DEVICE)
    nano_logits, nano_ms = {}, {}
    for dec in args.decisions:
        nano_logits[("calib", dec)] = torch.stack(M.score(nano, tok, calib[dec], 512, DEVICE))
        for name, t in tests.items():
            lg, ms = timed(M.score, nano, tok, t[dec], 512, DEVICE)
            nano_logits[(name, dec)], nano_ms[(name, dec)] = torch.stack(lg), ms / len(t[dec])
    del nano
    gc.collect()
    torch.cuda.empty_cache()

    ltok = AutoTokenizer.from_pretrained(args.llm, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True).eval()  # straight to GPU: avoids a ~8 GB RAM peak
    llm_lg, llm_ms = {}, {}
    for dec in args.decisions:
        llm_lg[("calib", dec)] = llm_logits(llm, ltok, calib[dec])
        for name, t in tests.items():
            lg, ms = timed(llm_logits, llm, ltok, t[dec])
            llm_lg[(name, dec)], llm_ms[(name, dec)] = lg, ms / len(t[dec])

    report, lines, preds = {}, [f"# Cascade: `{args.model}` → `{args.llm}`", ""], []
    for dec in args.decisions:
        c_labels = labels_of(calib[dec])
        t_nano = fit_temperature(nano_logits[("calib", dec)], c_labels)
        t_llm = fit_temperature(llm_lg[("calib", dec)], c_labels)
        prob = lambda lg, t: torch.softmax(lg / t, dim=1)  # noqa: E731

        calib_rows = sweep(prob(nano_logits[("calib", dec)], t_nano),
                           prob(llm_lg[("calib", dec)], t_llm), c_labels)
        chosen = pick_threshold(calib_rows)
        calib_rows_raw = sweep(prob(nano_logits[("calib", dec)], 1.0),
                               prob(llm_lg[("calib", dec)], t_llm), c_labels)
        chosen_raw = pick_threshold(calib_rows_raw)
        report[f"{dec}@calib"] = {"chosen_threshold": chosen, "rows": calib_rows,
                                  "chosen_threshold_raw": chosen_raw, "rows_raw": calib_rows_raw,
                                  "t_nano": t_nano, "t_llm": t_llm}

        for name, t in tests.items():
            labels = labels_of(t[dec])
            p_nano = prob(nano_logits[(name, dec)], t_nano)
            p_llm = prob(llm_lg[(name, dec)], t_llm)
            rows = sweep(p_nano, p_llm, labels)
            rows_raw = sweep(prob(nano_logits[(name, dec)], 1.0), p_llm, labels)
            for r in rows + rows_raw:
                r["ms_per_decision"] = nano_ms[(name, dec)] + r["escalated"] * llm_ms[(name, dec)]
            report[f"{dec}@{name}"] = {"chosen_threshold": chosen, "rows": rows,
                                       "chosen_threshold_raw": chosen_raw, "rows_raw": rows_raw,
                                       "nano_ms": nano_ms[(name, dec)], "llm_ms": llm_ms[(name, dec)],
                                       "n": len(labels)}
            for i, ex in enumerate(t[dec]):
                preds.append({"set": name, "decision": dec, "idx": i, "label": int(ex["label"]),
                              "nano_probs": [round(float(x), 5) for x in p_nano[i]],
                              "llm_probs": [round(float(x), 5) for x in p_llm[i]]})
            lines += [f"## {dec} on `{name}` (threshold chosen on calib: {chosen})", "",
                      "| accept Nano-Jev if confidence ≥ | escalated to LLM | accuracy | ms/decision |",
                      "|---|---|---|---|"]
            for r in rows:
                label = ("never (LLM only)" if r["threshold"] > 1 else
                         "always (Nano-Jev only)" if r["threshold"] == 0 else f"{r['threshold']:.2f}")
                mark = " ← chosen" if r["threshold"] == chosen else ""
                lines.append(f"| {label}{mark} | {r['escalated']:.1%} | {r['accuracy']:.3f} "
                             f"| {r['ms_per_decision']:.1f} |")
            lines.append("")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cascade.json").write_text(json.dumps(report, indent=2))
    (out / "cascade.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_jsonl(out / "cascade_preds.jsonl", preds)
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, r in report.items():
            if key.endswith("@calib"):
                continue
            ax.plot([x["escalated"] * 100 for x in r["rows"]], [x["accuracy"] for x in r["rows"]],
                    marker="o", label=key)
        ax.set_xlabel("% of decisions escalated to the LLM")
        ax.set_ylabel("accuracy")
        ax.set_title("Nano-Jev → LLM confidence cascade")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "cascade.png", dpi=150)
        print(f"\nplot: {out / 'cascade.png'}")
    except ImportError:
        print("\n(matplotlib not installed: skipped plot)")


if __name__ == "__main__":
    main()
