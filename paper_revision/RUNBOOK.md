# Revision runbook — what to run, in order

All commands run from `code_repo/nano_jev` in **PowerShell**. Results land in `results/v2/`,
models in `runs/v2/`. After any step you can run `python scripts/make_tables.py` to refresh the
paper tables; red `--` in the PDF = not run yet.

Status legend: ✅ done by Claude · 🟡 you run (no training) · 🔴 you run (TRAINING)

---

## 0. One-time setup

```powershell
$env:PYTHONUTF8 = "1"          # every new terminal: scripts print → and ± characters
pip install rank-bm25 scipy     # already installed on this machine
```

- 🟡 **LLM-AggreFact is gated.** Open https://huggingface.co/datasets/lytang/LLM-AggreFact, click
  "request access", then `huggingface-cli login`, then:
  ```powershell
  python scripts/prepare_extra_data.py aggrefact      # -> data_aggrefact/{calib,test}.jsonl
  ```
- 🟡 Optional groundedness baselines (official packages):
  ```powershell
  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
  # AlignScore pins old libraries: install it in a separate venv if it conflicts
  pip install git+https://github.com/yuh-zha/AlignScore.git ; python -m spacy download en_core_web_sm
  # download AlignScore-base.ckpt from https://huggingface.co/yzha/AlignScore into ckpts/
  ```
- Close other GPU programs: on 2026-11-01 the 3060 already had 8.6 GB in use, and Qwen3-4B needs ~8 GB.

## 1. Data (✅ done)

| Folder | What | Made by |
|---|---|---|
| `data_clean/` | train 47,272 · **dev 2,488 (from train pool)** · calib 3,982 · test 3,976 | `make_dev_split.py` ✅ |
| `data_custom/{agnews,yahoo,emotion}/` | zero-shot option-set tasks | `prepare_extra_data.py custom` ✅ |
| `data_e2e/{hotpot,musique}.jsonl` | 300 + 300 unused questions with paragraph pools | `prepare_extra_data.py e2e` ✅ |
| `data_aggrefact/` | LLM-AggreFact | 🟡 needs access (step 0) |

---

## 2. 🔴 TRAINING (your part)

v1.0 recipe, clean protocol. `train.py` now picks the epoch on `data_clean/dev.jsonl` and never
reads calib. ~40 min per K-pass run on the 3060.

```powershell
# T1  CaliDec K-pass, 5 seeds  (~3.5 h)
foreach ($s in 0..4) { python scripts/train.py --data data_clean --base microsoft/MiniLM-L12-H384-uncased --epochs 3 --lr 5e-5 --seed $s --out runs/v2/nanojev/seed$s }

# T2  1-pass classifier, 5 seeds  (same backbone/data/recipe; faster, 1 pass per example)
foreach ($s in 0..4) { python scripts/onepass.py train --data data_clean --base microsoft/MiniLM-L12-H384-uncased --epochs 3 --lr 5e-5 --seed $s --out runs/v2/onepass/seed$s }

# T3  data-size ablation, seed 0  (~5 + 10 + 20 min)
foreach ($f in 10, 25, 50) { python scripts/train.py --data data_clean --base microsoft/MiniLM-L12-H384-uncased --epochs 3 --lr 5e-5 --seed 0 --train-fraction ($f / 100) --out runs/v2/ablation_data/frac$f }
```

Check after T1: `runs/v2/nanojev/seed0/nanojev_config.json` must say
`"dev_source": "dev.jsonl (from train pool)"`.

Optional T4 (the "stronger route" for custom options: option-label augmentation retrain) is NOT
scripted yet. Ask Claude to write it if you want to try it; it is only worth it if time allows.

---

## 3. Evaluation (🟡 no training, run after section 2)

Seed-0 runs below without `--no-save` write `calibration.json` into the run folder; later calls
use `--no-save`.

```powershell
# E1  CaliDec: test / held-out / AggreFact for every seed
foreach ($s in 0..4) {
  python scripts/evaluate.py --model runs/v2/nanojev/seed$s --data data_clean --results-dir results/v2/nanojev/seed$s --name test
  python scripts/evaluate.py --model runs/v2/nanojev/seed$s --data data_clean --test-data data_heldout --results-dir results/v2/nanojev/seed$s --name heldout --no-save
  python scripts/evaluate.py --model runs/v2/nanojev/seed$s --data data_clean --test-data data_aggrefact --results-dir results/v2/nanojev/seed$s --name aggrefact --no-save
}
# domain re-fit demo (temperature fitted on AggreFact dev instead of MultiNLI calib)
python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_aggrefact --test-data data_aggrefact --results-dir results/v2/nanojev/seed0 --name aggrefact_refit --no-save

# E2  1-pass classifier
foreach ($s in 0..4) {
  python scripts/onepass.py eval --model runs/v2/onepass/seed$s --data data_clean --results-dir results/v2/onepass/seed$s --name test
  python scripts/onepass.py eval --model runs/v2/onepass/seed$s --data data_clean --test-data data_heldout --results-dir results/v2/onepass/seed$s --name heldout
  python scripts/onepass.py eval --model runs/v2/onepass/seed$s --data data_clean --test-data data_aggrefact --results-dir results/v2/onepass/seed$s --name aggrefact
}

# E3  data-size ablation
foreach ($f in 10, 25, 50) { python scripts/evaluate.py --model runs/v2/ablation_data/frac$f --data data_clean --results-dir results/v2/ablation_data/frac$f --name test --no-save }
```

### Can run NOW (do not depend on training)

```powershell
# E4  reference baselines (BGE reranker + prompted Qwen3-4B), now with per-example predictions
#     calib is unchanged, so accuracies should reproduce the submitted ones
python scripts/baselines.py --which all --data data_clean --out results/v2/baselines
python scripts/baselines.py --which all --data data_clean --test-data data_heldout --suffix _heldout --out results/v2/baselines
python scripts/baselines.py --which all --data data_clean --test-data data_aggrefact --suffix _aggrefact --out results/v2/baselines

# E5a compact groundedness checkers (MiniCheck lines need the package from step 0)
python scripts/baselines_grounded.py --systems nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli nli:MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli minicheck:roberta-large minicheck:deberta-v3-large --data data_clean --test-data data_clean data_heldout data_aggrefact --out results/v2/grounded
python scripts/baselines_grounded.py --systems alignscore:ckpts/AlignScore-base.ckpt --data data_clean --test-data data_clean data_heldout data_aggrefact --out results/v2/grounded

# E7a custom option sets: Qwen reference (v0.1 / v1.0 zero-shot already done ✅)
foreach ($t in "agnews", "yahoo", "emotion") { python scripts/baselines.py --which llm --data data_custom/$t --out results/v2/custom --suffix _$t }
```

### After training

```powershell
# E5b CaliDec chunk-and-max on long documents
python scripts/baselines_grounded.py --systems nanojev:runs/v2/nanojev/seed0 --data data_clean --test-data data_clean data_heldout data_aggrefact --out results/v2/grounded

# E6  cascade on ALL three decisions (test-data names must stay exactly data_clean / data_heldout)
python scripts/cascade.py --model runs/v2/nanojev/seed0 --data data_clean --test-data data_clean data_heldout --decisions relevance sufficient grounded --out results/v2/cascade/seed0

# E7b custom option sets: revised model zero-shot + supervised probe reference
foreach ($t in "agnews", "yahoo", "emotion") { python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_custom/$t --results-dir results/v2/custom/v2 --name $t --no-save }
python scripts/custom_probe.py --encoder runs/v2/onepass/seed0 --tasks data_custom/agnews data_custom/yahoo data_custom/emotion --out results/v2/custom/probe_onepass.json

# E8  truncation analysis + balanced-truncation variant
python scripts/truncation_analysis.py --model runs/v2/nanojev/seed0 --data data_clean --preds results/v2/nanojev/seed0/test_preds.jsonl --out results/v2/truncation/test.json --write-balanced data_clean_balanced
python scripts/truncation_analysis.py --model runs/v2/nanojev/seed0 --data data_heldout --preds results/v2/nanojev/seed0/heldout_preds.jsonl --out results/v2/truncation/heldout.json --write-balanced data_heldout_balanced
python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_clean --test-data data_clean_balanced --results-dir results/v2/nanojev/seed0 --name test_balanced --no-save
python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_clean --test-data data_heldout_balanced --results-dir results/v2/nanojev/seed0 --name heldout_balanced --no-save

# E9  latency, one protocol for every table (CPU run is slow for Qwen: n=64)
python scripts/latency.py --systems nanojev:runs/v2/nanojev/seed0 onepass:runs/v2/onepass/seed0 reranker:BAAI/bge-reranker-v2-m3 llm:Qwen/Qwen3-4B-Instruct-2507 --device cuda --out results/v2/latency/gpu.json
python scripts/latency.py --systems nanojev:runs/v2/nanojev/seed0 onepass:runs/v2/onepass/seed0 reranker:BAAI/bge-reranker-v2-m3 llm:Qwen/Qwen3-4B-Instruct-2507 --device cpu --n 64 --out results/v2/latency/cpu.json

# E10 end-to-end RAG (300 HotpotQA + 300 MuSiQue; 4 filters x generation)
python scripts/e2e_rag.py --nanojev runs/v2/nanojev/seed0 --sets data_e2e/hotpot.jsonl data_e2e/musique.jsonl --out results/v2/e2e
```

## 4. Statistics (CPU only, seconds)

```powershell
python scripts/stats.py seeds --name nanojev_test    --preds "results/v2/nanojev/seed*/test_preds.jsonl"
python scripts/stats.py seeds --name nanojev_heldout --preds "results/v2/nanojev/seed*/heldout_preds.jsonl"
python scripts/stats.py seeds --name onepass_test    --preds "results/v2/onepass/seed*/test_preds.jsonl"
python scripts/stats.py compare --name nanojev_vs_onepass_test --a results/v2/nanojev/seed0/test_preds.jsonl --b results/v2/onepass/seed0/test_preds.jsonl
python scripts/stats.py compare --name nanojev_vs_llm_test     --a results/v2/nanojev/seed0/test_preds.jsonl --b results/v2/baselines/llm_preds.jsonl
python scripts/stats.py compare --name nanojev_vs_llm_heldout  --a results/v2/nanojev/seed0/heldout_preds.jsonl --b results/v2/baselines/llm_heldout_preds.jsonl
python scripts/stats.py compare --name nanojev_vs_reranker_test --a results/v2/nanojev/seed0/test_preds.jsonl --b results/v2/baselines/reranker_preds.jsonl
python scripts/stats.py compare --name nanojev_vs_minicheck_aggrefact --a results/v2/nanojev/seed0/aggrefact_preds.jsonl --b results/v2/grounded/minicheck-roberta-large_data_aggrefact_preds.jsonl
python scripts/stats.py cascade --name cascade_seed0 --cascade results/v2/cascade/seed0/cascade_preds.jsonl --cascade-json results/v2/cascade/seed0/cascade.json
```
Each writes `results/v2/stats/<name>.{json,md}`.

## 5. Tables, paper, check

```powershell
python scripts/make_tables.py                       # -> paper_revision/tables/*.tex + numbers.tex
latexmk -pdf nanojev_icck_paper.tex                 # needs ICCKjournal.cls next to the .tex
Select-String -Path paper_revision/tables/*.tex -Pattern "ph\{" | Measure-Object   # must be 0
Select-String -Path nanojev_icck_paper.tex -Pattern "revnote\{" | Measure-Object  # resolve each blue note
```

Then read every blue `[...]` note in the PDF, rewrite that sentence to match the numbers and
delete the note. Finally fill the `[RESULT]` slots in `paper_revision/response_letter.md`.

## Rough GPU time (RTX 3060)

| Block | Time |
|---|---|
| T1 K-pass × 5 | ~3.5 h |
| T2 1-pass × 5 | ~1.5–2 h |
| T3 ablation | ~35 min |
| E4 baselines (Qwen on 3 sets) | ~1–1.5 h |
| E5 groundedness checkers | ~30–45 min |
| E6 cascade | ~40 min |
| E10 end-to-end | ~30 min |
| everything else | < 30 min |
