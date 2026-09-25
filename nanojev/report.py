"""Shared evaluation flow: fit temperature on calib, report test metrics, render tables."""

import json
from pathlib import Path

import torch

from .calibration import fit_temperature, metrics


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def by_decision(examples: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for ex in examples:
        groups.setdefault(ex["decision"], []).append(ex)
    return groups


def labels_of(examples: list[dict]) -> torch.Tensor:
    return torch.tensor([ex["label"] for ex in examples])


def decision_report(calib_logits, calib_labels, test_logits, test_labels, ms: float) -> dict:
    t = fit_temperature(calib_logits, calib_labels)
    return {
        "temperature": t,
        "raw": metrics(test_logits, test_labels),
        "calibrated": metrics(test_logits, test_labels, t),
        "ms_per_decision": ms,
    }


def render_table(title: str, report: dict) -> str:
    lines = [f"## {title}", "",
             "| decision | n | T | acc | macro-F1 | NLL raw → cal | Brier raw → cal "
             "| ECE raw → cal | ms/decision |",
             "|---|---|---|---|---|---|---|---|---|"]
    for dec, r in report.items():
        raw, cal = r["raw"], r["calibrated"]
        lines.append(
            f"| {dec} | {raw['n']} | {r['temperature']:.2f} | {cal['accuracy']:.3f} "
            f"| {cal['macro_f1']:.3f} | {raw['nll']:.3f} → {cal['nll']:.3f} "
            f"| {raw['brier']:.3f} → {cal['brier']:.3f} | {raw['ece']:.3f} → {cal['ece']:.3f} "
            f"| {r['ms_per_decision']:.1f} |")
    return "\n".join(lines)


def save(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
