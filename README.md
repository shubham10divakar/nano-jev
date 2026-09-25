# Nano-Jev

> **Status: v0.1 research preview.** Jev-style decision model. Independent; not affiliated
> with TypeSafe AI or the NanoJev GitHub project.

A tiny (~22M parameter) Jev-style **typed decision model** for RAG pipelines. It takes a
question, a set of options and some state text, and returns a calibrated probability
for each option.

| Decision | Type | Options |
|---|---|---|
| `relevance` | score | irrelevant / partially relevant / directly answers |
| `sufficient` | noul | yes / no |
| `grounded` | noul | yes / no |

## How it works

Each (question, option, state) triple goes through a small cross-encoder that outputs one
logit:

```
[CLS] question: <q> option: <opt> [SEP] <state text> [SEP]  → encoder → linear → logit
```

A decision's option logits are softmaxed together, then divided by a per-decision
temperature fitted on held-out data, so the probabilities are calibrated. Because options
are scored independently, the model also accepts **option sets it never saw in training**
(`Decider.decide`). Training uses cross-entropy over each group of options.

## Setup (Windows)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
$env:HF_HOME = "D:\hf_cache"   # optional: where datasets and models are cached
```

On Linux/macOS use `.venv/bin/python`.

## Run

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py --preset default   # HotpotQA + SQuAD 2.0 + MultiNLI -> data/
.venv\Scripts\python.exe scripts\train.py                           # -> runs/nano-jev-dev  (~15 min on an RTX 3060)
.venv\Scripts\python.exe scripts\evaluate.py --model runs\nano-jev-dev   # fits temperatures, writes results
.venv\Scripts\python.exe scripts\demo.py --model runs\nano-jev-dev
.venv\Scripts\python.exe scripts\baselines.py --which all           # bge-reranker + prompted Qwen3-4B (~35 min)
```

Use `--preset smoke` for a 1-minute end-to-end check. The exact v0.1 data splits are
included in `data/` (seed 0), so `prepare_data.py` is only needed to rebuild them.

Zero-shot baseline (the untrained init):
`scripts\evaluate.py --model cross-encoder/ms-marco-MiniLM-L6-v2 --no-save`

## Use

```python
from nanojev import Decider

d = Decider.from_pretrained("sdmlai/nano-jev", revision="v0.1")   # or a local folder
d.relevance(query, passages)        # [{"irrelevant": .., "partially relevant": .., "directly answers": ..}, ...]
d.sufficient(query, passages)       # {"yes": .., "no": ..}
d.grounded(claim, context)          # {"yes": .., "no": ..}
d.decide("Which topic?", ["sports", "finance"], text)   # any option set
```

Pretrained v0.1 weights: [huggingface.co/sdmlai/nano-jev](https://huggingface.co/sdmlai/nano-jev) (tag `v0.1`). The first call downloads and caches them.

## Layout

```
nanojev/
  schema.py        decision definitions (question templates, options)
  data.py          dataset -> typed examples (HotpotQA, SQuAD 2.0, MultiNLI)
  model.py         cross-encoder scoring, grouped softmax loss
  calibration.py   temperature scaling, ECE / Brier / NLL
  report.py        shared calibrate-and-report flow
  decider.py       inference API
scripts/           prepare_data, train, evaluate, baselines, demo
data/              v0.1 train / calib / test splits (JSONL)
results/           evaluation tables and baseline comparison
```

## Results

v0.1 on held-out test halves of the validation sets. Every model is temperature-calibrated
on the same calib split. Details: [`results/v0.1.md`](results/v0.1.md).

| decision | Nano-Jev 22M | bge-reranker-v2-m3 568M | Qwen3-4B prompted | untrained init |
|---|---|---|---|---|
| relevance (3-way) | **0.798** | 0.694 | 0.660 | 0.273 |
| sufficient | **0.759** | 0.692 | 0.706 | 0.503 |
| grounded | 0.804 | 0.798 | **0.844** | 0.540 |

Nano-Jev is ~90–165× faster than the prompted LLM. The baselines are zero-shot on these
datasets and Nano-Jev is not, so read the caveats in
[`results/v0.1_comparison.md`](results/v0.1_comparison.md).

## Preview limitations

- Evaluated only on the test splits of its own training datasets; no out-of-distribution results yet.
- The base model (`cross-encoder/ms-marco-MiniLM-L6-v2`) was trained on MS MARCO, whose
  terms are non-commercial. Commercial users should check them. v1.0 will retrain on an MIT-licensed base.
- Groundedness is below a prompted 4B LLM.
- English, Wikipedia-style text only.

## Data credits

Training and evaluation examples are derived from
[HotpotQA](https://hotpotqa.github.io/) (CC BY-SA 4.0),
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0) and
[MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) (see its licence terms).
The files in `data/` carry those datasets' licences.
