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

Until `pip install nano-jev` is published, run these from the repo root with the venv
activated (`.venv\Scripts\activate`), so `nanojev` is importable.

Pretrained weights live on Hugging Face at
[huggingface.co/sdmlai/nano-jev](https://huggingface.co/sdmlai/nano-jev), one git tag per
version. They download automatically the first time you use a version, then load from the
local cache (and work offline).

### Python

```python
import nanojev

d = nanojev.load()                  # selected model (default: v0.1), downloaded on first use
d = nanojev.load("v0.1")            # a specific version
d = nanojev.load("runs/nano-jev-dev")   # your own trained folder

d.relevance(query, passages)        # [{"irrelevant": .., "partially relevant": .., "directly answers": ..}, ...]
d.sufficient(query, passages)       # {"yes": .., "no": ..}
d.grounded(claim, context)          # {"yes": .., "no": ..}
d.decide("Which topic?", ["sports", "finance"], text)   # any option set
d.version                           # "0.1"

models, online = nanojev.list_models()   # versions on the Hub, with .downloaded flags
```

`nanojev.Decider.from_pretrained("sdmlai/nano-jev", revision="v0.1")` also works.

### Command line

```powershell
python -m nanojev list                  # available weights; * marks the selected one
python -m nanojev list --local runs     # also show your local training runs
python -m nanojev download v0.1         # fetch ahead of time (e.g. before going offline)
python -m nanojev use v0.1              # choose the default weights (a version, Hub repo[@tag] or folder)
python -m nanojev use --reset           # back to the package default
python -m nanojev current               # what's selected and where it is on disk

python -m nanojev relevance -q "When was UCL founded?" -p "University College London was founded in 1826."
python -m nanojev sufficient -q QUERY -p PASSAGE -p PASSAGE
python -m nanojev grounded --claim "UCL was founded in 1900." --context "University College London was founded in 1826."
python -m nanojev decide --question "Which topic is this passage about?" -o computing -o cooking --state TEXT
```

Decision commands take `--model` to override the selection for one call, and `--json`.

Example `list` output:

```
   MODEL  SOURCE  STATUS            BASE                                 PARAMS  RELEASED    DOWNLOADED
*  v0.1   hub     research preview  cross-encoder/ms-marco-MiniLM-L6-v2  22.7M   2026-09-25  yes

selected: v0.1  (from default)
```

### Which weights get used

1. `--model` / the argument to `nanojev.load(...)`
2. the `NANOJEV_MODEL` environment variable
3. the choice saved by `python -m nanojev use` (in `~/.nanojev/config.json`)
4. the package default: `v0.1`

## Layout

```
nanojev/
  schema.py        decision definitions (question templates, options)
  data.py          dataset -> typed examples (HotpotQA, SQuAD 2.0, MultiNLI)
  model.py         cross-encoder scoring, grouped softmax loss
  calibration.py   temperature scaling, ECE / Brier / NLL
  report.py        shared calibrate-and-report flow
  decider.py       inference API
  registry.py      list / download / select weights
  __main__.py      command-line tool (python -m nanojev)
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
