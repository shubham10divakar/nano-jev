# Nano-Jev

A tiny (~22M parameter) Jev-style **typed decision model** for RAG pipelines. It takes a
question, a set of options and some state text, and returns a calibrated probability
for each option. Design notes: [`../plan/10_tiny_jev.md`](../plan/10_tiny_jev.md).

| Decision | Type | Options |
|---|---|---|
| `relevance` | score | irrelevant / partially relevant / directly answers |
| `sufficient` | noul | yes / no |
| `grounded` | noul | yes / no |

Options are scored one at a time by a cross-encoder and softmaxed together, so the model
also accepts **option sets it never saw in training** (`Decider.decide`).

## Setup (Windows, from `code_repo/`)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r nano_jev\requirements.txt
$env:HF_HOME = "D:\hf_cache"
```

## Run (from `nano_jev/`)

```powershell
..\.venv\Scripts\python.exe scripts\prepare_data.py --preset default   # HotpotQA + SQuAD 2.0 + MultiNLI -> data/
..\.venv\Scripts\python.exe scripts\train.py                           # -> runs/nano-jev
..\.venv\Scripts\python.exe scripts\evaluate.py                        # fits temperatures, writes results
..\.venv\Scripts\python.exe scripts\demo.py
```

Use `--preset smoke` for a 1-minute end-to-end check.

Zero-shot baseline (the untrained init):
`scripts\evaluate.py --model cross-encoder/ms-marco-MiniLM-L6-v2 --no-save`

## Use

```python
from nanojev import Decider

d = Decider.from_pretrained("runs/nano-jev")
d.relevance(query, passages)        # [{"irrelevant": .., "partially relevant": .., "directly answers": ..}, ...]
d.sufficient(query, passages)       # {"yes": .., "no": ..}
d.grounded(claim, context)          # {"yes": .., "no": ..}
d.decide("Which topic?", ["sports", "finance"], text)   # any option set
```

## Layout

```
nanojev/
  schema.py        decision definitions (question templates, options)
  data.py          dataset -> typed examples (HotpotQA, SQuAD 2.0, MultiNLI)
  model.py         cross-encoder scoring, grouped softmax loss
  calibration.py   temperature scaling, ECE / Brier / NLL
  decider.py       inference API
scripts/           prepare_data, train, evaluate, demo
results/           tracked copies of evaluation tables
```

## Results

See [`results/`](results/).
