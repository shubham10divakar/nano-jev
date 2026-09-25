# Changelog

Package versions (PyPI `nano-jev`) and weight versions (Hugging Face `sdmlai/nano-jev`
tags) are listed together. Every released weight version stays available:
`nanojev.load("v0.1")` keeps working after newer releases.

## 1.0.0 — weights v1.0 (2026-09-25) — stable

- **New default weights `v1.0`**: 33.4M MiniLM-L12 cross-encoder from
  `microsoft/MiniLM-L12-H384-uncased` (MIT, no MS MARCO lineage), chosen from three
  clean-licence bases. Weights licence: **MIT**.
- Test accuracy vs v0.1: relevance 0.815 (+1.7), sufficient 0.845 (+8.6), grounded 0.844 (+4.0).
- **Held-out evaluation** on MuSiQue and VitaminC (`scripts/prepare_data.py --heldout`):
  v1.0 0.635 / 0.670 / 0.675, on par with v0.1; calibration is overconfident out of domain.
- **Cascade** (`scripts/cascade.py`): escalating unsure `grounded` decisions to Qwen3-4B
  beats both models on the test split (0.868) with 26% of LLM calls.
- `v0.1` stays available: `nanojev.load("v0.1")`, `nano-jev use v0.1`.
- Known regression: custom option sets (`decide`) are weaker than in v0.1.
- Upgrade note: probabilities and good thresholds differ between v0.1 and v1.0; re-check
  your thresholds, or pin `v0.1`.
- Tooling: held-out options for `evaluate.py` / `baselines.py`, released weight folders are
  never written to (`--no-save --results-dir`), LLM baseline loads straight to GPU and keeps
  only last-position logits (much lower memory).

## 0.1.0 — weights v0.1 (2026-09-25) — research preview

- First release. Decisions: `relevance` (3 levels), `sufficient`, `grounded`, plus
  `decide()` for custom option sets.
- Weights: 22.7M MiniLM-L6 cross-encoder fine-tuned from
  `cross-encoder/ms-marco-MiniLM-L6-v2` on HotpotQA / SQuAD 2.0 / MultiNLI (49,760 examples),
  per-decision temperature calibration.
- Weights registry: `nanojev.load()`, `nanojev.list_models()`; `nano-jev` CLI with
  `list`, `download`, `use`, `current` and decision commands.
- Licence note: the base model was trained on MS MARCO (non-commercial terms).
