# nano-jev (CaliDec) paper — reviewer comments and revision plan

Paper: `nanojev_icck_paper.tex` (revised in place; submitted version kept as
`paper_revision/submitted_v1.tex`). Plan written 2026-11-01.
Status legend: `[ ]` todo · `[~]` prepared, waiting on a run · `[x]` done

**What to run next → [`RUNBOOK.md`](RUNBOOK.md).** Response letter draft →
[`response_letter.md`](response_letter.md).

---

## 0. Facts checked against the repo

| Item | Submitted paper | Repo actually has |
|---|---|---|
| Train examples | 49,760 | 49,760 ✔ = 5,960 HotpotQA q × (4 rel + 2 suf) + 6,000 SQuAD2 + 8,000 MNLI |
| Calib examples | 4,800 (`\ph`) | **3,982** (rel 1,988 · suf 1,494 · grd 500) |
| Test examples | 4,800 = 2,400 + 1,600 + 800 (`\ph`) | **3,976** (rel 1,984 · suf 1,492 · grd **500**) |
| Held-out | — | 6,695 (MuSiQue rel 4,655 · suf 1,040 · VitaminC grd 1,000) |
| Placeholders | — | 41 `\ph{}` values in the submitted .tex |
| Dev / checkpoint selection | — | `scripts/train.py` sampled dev **from calib.jsonl** → R3.2 leak (now fixed) |

## 0b. Findings so far (no training needed, measured 2026-11-01)

- **Custom option sets fail for v1.0**: zero-shot accuracy is *below chance* on all three tasks
  (AG News 0.186 vs 0.250 · Yahoo 0.056 vs 0.100 · Emotion 0.138 vs 0.167). v0.1 is above chance
  but weak (0.396 / 0.239 / 0.334). The claim has been removed from the paper (R1.2, R3.3).
- **Truncation is negligible in-distribution** (≤0.5% of inputs > 512 tokens) but **MuSiQue
  sufficiency is 14.2% truncated, with 7.9% losing part of a gold passage**; accuracy is 0.628
  truncated vs. 0.677 not truncated (v1.0 model; final numbers come from seed 0).
- v1.0 re-evaluated with the new scripts reproduces 0.815 / 0.845 / 0.844 exactly. Relevance
  macro-F1 is **0.789** (paper Table 8 dummy said 0.805). Real raw ECE: test 0.041 / 0.050 / 0.065,
  held-out 0.220 / 0.221 / 0.211.
- v0.1 real temperatures are 1.45 / 1.38 / 1.23 (paper dummies: 1.18 / 1.09 / 1.22).
- The cascade code picks t = 0 (escalate nothing) where the LLM is weaker (smoke test with a
  small LLM). This is the expected "protective" behaviour.

---

## 1. Reviewer comments

| ID | Reviewer(s) | Comment | Type | Status |
|---|---|---|---|---|
| C1 | R1.1, R1.2, R3.1 | No fine-tuned 1-pass baseline; K-pass not ablated | experiment | [~] `onepass.py` ready → train T2 |
| C2 | R3.1, R1 m4 | Weak baselines; "fair comparison" overstated | exp + text | [~] `baselines_grounded.py` ready; text done |
| C3 | R1.3, R2.2, R3.4 | Cascade only on grounded; no held-out % escalated | exp + text | [~] `cascade.py` all decisions; text done |
| C4 | R1.2, R3.3 | Custom options = anecdote | exp + text | [x] measured (v0.1/v1.0); claim removed · [~] Qwen + revised model |
| C5 | R3.2 | Calib used for checkpoint selection | retrain | [~] `data_clean/` + `train.py` fixed → train T1 |
| C6 | R2.3, R3.5 | Single run, no CIs/tests | experiment | [~] `stats.py` ready → needs seeds |
| C7 | R1.4 | Truncation, short claims, no downstream eval | experiment | [~] truncation/AggreFact/e2e scripts ready; AggreFact gated |
| C8 | R1 | Table 7 ≠ Table 8 | table | [x] both generated from one run |
| C9 | R1 | "Block apps" caption leftover | text | [x] |
| C10 | R1 | Table 2 counts, 20k→49,760 | table | [x] generated table + explanation |
| C11 | R1 | Missing citations | text | [x] §2.3 + 13 new references |
| C12 | R2.1 | Repetition | text | [x] calibration merged in §5.5, gap stated once per place |

| ID | Self-found | Status |
|---|---|---|
| S1 | 41 placeholders | [x] every number is generated; missing ones show red `--` |
| S2 | Latency inconsistent | [x] one protocol (`latency.py`), per-decision columns · [~] run E9 |
| S3 | "400+ downloads" | [x] not present in this version |
| S4 | Conclusion overclaim | [x] qualified as in-distribution |
| S5 | Plant-disease future work | [x] removed |
| S6 | CaliDec vs nano-jev naming | [x] one sentence in §4.5 |

---

## 2. What was prepared (2026-11-01)

New / changed code (all in `scripts/`; the `nanojev` package is untouched):

| File | Purpose |
|---|---|
| `make_dev_split.py` | dev from train pool, grouped by question → `data_clean/` |
| `train.py` (patched) | uses `dev.jsonl` if present; `--train-fraction`; logs seed/dev source |
| `evaluate.py`, `baselines.py`, `cascade.py` (patched) | save per-example predictions; per-class metrics; cascade on all decisions + τ=1 ablation |
| `onepass.py` | 1-pass classifier: train + eval |
| `baselines_grounded.py` | NLI / MiniCheck / AlignScore / CaliDec chunk-max |
| `prepare_extra_data.py` | AggreFact, custom option sets, e2e question sets |
| `custom_probe.py` | supervised probe reference for custom options |
| `truncation_analysis.py` | truncation stats, gold-cut, balanced-truncation copy |
| `e2e_rag.py` | BM25 → filter → Qwen generation, EM/F1 + latency |
| `latency.py` | single latency protocol for every system |
| `stats.py` | seeds mean±std, bootstrap CI, McNemar, cascade tests |
| `make_tables.py` | all LaTeX tables + `numbers.tex` macros from results |
| `revision_utils.py` | shared helpers |

Smoke-tested: evaluate → preds → stats; onepass + train.py on a tiny model; cascade (all decisions)
and e2e with small stand-in models; truncation; make_tables; paper compiles (stub class).

Paper: all tables `\input` from `paper_revision/tables/`, prose numbers are macros, new sections
5.3 (1-pass), 5.4 (groundedness checkers), 5.9 (truncation), 5.10 (end-to-end), 5.11 (custom
options, systematic). Blue `\revnote{}` marks wording to settle once results are in.

---

## 3. Resume here
Next action (user): run RUNBOOK §0 (AggreFact access) and §2 training T1–T3; the "can run NOW"
baselines in §3 can go in parallel or before. Then §3–§5, resolve blue notes, fill the letter.

Progress 2026-11-01 (commit 6a801ba): done = BGE reranker on test + held-out, Qwen3-4B on test
(acc grounded/relevance/sufficient: Qwen 0.844/0.660/0.706; BGE test 0.798/0.694/0.692;
BGE held-out 0.737/0.478/0.597). Still missing: Qwen held-out (`llm_heldout.json`; re-run the
held-out baselines command), AggreFact, grounded checkers, custom Qwen, and ALL training (T1–T3)
plus everything after it. Ordered command list = RUNBOOK, or `paper_revision/run_all.ps1`
(skips finished steps).
