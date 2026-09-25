# Nano-Jev

[![PyPI version](https://img.shields.io/pypi/v/nano-jev?label=PyPI&color=blue)](https://pypi.org/project/nano-jev/)
[![Python](https://img.shields.io/pypi/pyversions/nano-jev)](https://pypi.org/project/nano-jev/)
[![Downloads](https://static.pepy.tech/badge/nano-jev)](https://pepy.tech/project/nano-jev)
[![Monthly downloads](https://img.shields.io/pypi/dm/nano-jev?label=downloads%2Fmonth)](https://pypistats.org/packages/nano-jev)
[![Model version](https://img.shields.io/badge/model-v0.1-yellow?logo=huggingface)](https://huggingface.co/sdmlai/nano-jev/tree/v0.1)
[![HF downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fsdmlai%2Fnano-jev&query=%24.downloads&label=HF%20downloads&logo=huggingface&color=yellow)](https://huggingface.co/sdmlai/nano-jev)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/shubham10divakar/nano-jev/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-research%20preview-orange)](#limitations)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/shubham10divakar/nano-jev/blob/main/examples/kaggle_smoke_test.ipynb)

> Jev-style decision model. Independent; not affiliated with TypeSafe AI or the NanoJev
> GitHub project.

**A tiny (22.7M parameter) calibrated decision model for RAG pipelines.** Give it a
question, some options and some context; it returns a **probability for each option** in
under a millisecond on a GPU (a few ms on CPU). It never generates text, so it can't
return a malformed answer.

Use it for the small decisions an agentic RAG loop makes over and over:

| Decision | Options | Asks |
|---|---|---|
| `relevance` | irrelevant / partially relevant / directly answers | Is this passage useful for the query? |
| `sufficient` | yes / no | Do these passages contain enough to answer? |
| `grounded` | yes / no | Is this claim supported by the context? |
| `decide` | **your own options** | Anything else, e.g. topic or route |

## At a glance

| | |
|---|---|
| Package | `nano-jev` 0.1.0 on [PyPI](https://pypi.org/project/nano-jev/) (import `nanojev`, command `nano-jev`) |
| Weights | [`sdmlai/nano-jev`](https://huggingface.co/sdmlai/nano-jev), version `v0.1` (default), downloaded automatically on first use |
| Model | MiniLM cross-encoder, 6 layers, hidden size 384, **22.7M parameters**, max 512 tokens |
| Accuracy (test split) | relevance **0.798**, sufficient **0.759**, grounded **0.804** ([results](#results)) |
| Calibration error (ECE) | 0.020 / 0.019 / 0.045 after temperature scaling |
| Speed | ~0.4–2.3 ms per decision batched on an RTX 3060; runs on CPU |
| Python | 3.10+ · PyTorch 2.1+ · Transformers 4.45+ |
| Licence | Apache-2.0 (see the [MS MARCO note](#licence)) |
| Status | **v0.1 research preview**; v1.0 planned |

## Install

```bash
pip install nano-jev
```

PyTorch is installed from PyPI if you don't have it. For a CUDA build, install PyTorch
first following [pytorch.org](https://pytorch.org/get-started/locally/).

## Quick start

```python
import nanojev

d = nanojev.load()   # default weights (v0.1); downloads ~90 MB the first time, then cached

d.relevance("When was UCL founded?", [
    "University College London was founded in 1826.",
    "The Analytical Engine was a proposed mechanical computer.",
])
# [{'irrelevant': 0.021, 'partially relevant': 0.041, 'directly answers': 0.938},
#  {'irrelevant': 0.822, 'partially relevant': 0.143, 'directly answers': 0.035}]

d.sufficient("In which year was the university attended by Ada Lovelace's tutor founded?", [
    "Augustus De Morgan tutored Ada Lovelace. He was a professor at University College London.",
    "University College London was founded in 1826.",
])
# {'yes': 0.899, 'no': 0.101}      (with only the first passage: {'yes': 0.055, 'no': 0.945})

d.grounded("UCL was founded in 1900.", "University College London was founded in 1826.")
# {'yes': 0.051, 'no': 0.949}

d.decide("Which topic is this passage about?", ["computing", "cooking", "football"],
         "The Analytical Engine was a proposed mechanical computer.")
# {'computing': 0.783, 'cooking': 0.143, 'football': 0.074}
```

### In a RAG loop

```python
d = nanojev.load()

def answer(query, retrieve, llm):
    passages = retrieve(query, k=10)
    scores = d.relevance(query, passages)
    kept = [p for p, s in zip(passages, scores) if s["irrelevant"] < 0.5]

    if d.sufficient(query, kept)["yes"] < 0.85:          # not enough context
        kept += retrieve(query, k=10, page=2)            # retrieve more / rewrite the query / abstain

    reply = llm(query, kept)
    if d.grounded(reply, "\n".join(kept))["yes"] < 0.5:  # unsupported answer
        reply = llm(query, kept, strict=True)            # regenerate, flag, or escalate
    return reply
```

The thresholds are examples. Calibrated probabilities make thresholds meaningful, but
**check them on your own data**.

## Choosing weights

Weights live on Hugging Face at [`sdmlai/nano-jev`](https://huggingface.co/sdmlai/nano-jev),
one git tag per version. A version downloads the first time you use it and then loads
from the local cache (works offline).

```python
import nanojev

models, online = nanojev.list_models()           # versions on the Hub (+ which are downloaded)
for m in models:
    print(m.name, m.status, m.params, "downloaded" if m.downloaded else "")

d = nanojev.load("v0.1")                          # a specific version
d = nanojev.load("sdmlai/nano-jev@v0.1")          # any Hub repo, optionally @tag
d = nanojev.load("path/to/my-trained-model")      # a local folder
print(d.version, d.config["base"])
```

Which weights `nanojev.load()` uses when you don't say:

1. the `NANOJEV_MODEL` environment variable
2. the choice saved with `nano-jev use ...` (stored in `~/.nanojev/config.json`)
3. the package default, `v0.1` (pinned per package release, so results stay reproducible)

## Command line

Installing the package adds a `nano-jev` command (`python -m nanojev` does the same).

```bash
nano-jev --version
nano-jev list                         # available weights; * marks the selected one
nano-jev list --local runs            # also list your own trained folders under runs/
nano-jev download v0.1                # fetch ahead of time, e.g. before going offline
nano-jev use v0.1                     # choose default weights: a version, Hub repo[@tag] or folder
nano-jev use --reset                  # back to the package default
nano-jev current                      # what's selected and where it is on disk

nano-jev relevance -q "When was UCL founded?" -p "University College London was founded in 1826." -p "Another passage."
nano-jev sufficient -q "QUERY" -p "PASSAGE 1" -p "PASSAGE 2"
nano-jev grounded --claim "UCL was founded in 1900." --context "University College London was founded in 1826."
nano-jev decide --question "Which topic is this passage about?" -o computing -o cooking --state "TEXT"
```

Decision commands accept `--model` (override the selection for one call) and `--json`.

```
$ nano-jev list
   MODEL  SOURCE  STATUS            BASE                                 PARAMS  RELEASED    DOWNLOADED
*  v0.1   hub     research preview  cross-encoder/ms-marco-MiniLM-L6-v2  22.7M   2026-09-25  yes

selected: v0.1  (from default)

$ nano-jev grounded --claim "UCL was founded in 1900." --context "University College London was founded in 1826."
yes=0.051  no=0.949
```

## How it works

Each (question, option, context) triple goes through a small cross-encoder that outputs
one logit:

```
[CLS] question: <q> option: <opt> [SEP] <context> [SEP]  → encoder → linear → logit
```

A decision's option logits are divided by that decision's temperature (fitted on
held-out data) and softmaxed, so the probabilities are calibrated. Options are scored
independently, which is why you can pass option sets the model never saw in training.

## Results

Test split = held-out halves of the HotpotQA / SQuAD 2.0 / MultiNLI validation sets.
Every model is temperature-calibrated on the same calibration split. Accuracy:

| Decision | **Nano-Jev v0.1** (22.7M) | bge-reranker-v2-m3 (568M) | Qwen3-4B-Instruct, prompted (4B) | untrained base |
|---|---|---|---|---|
| relevance (3-way) | **0.798** | 0.694 | 0.660 | 0.273 |
| sufficient | **0.759** | 0.692 | 0.706 | 0.503 |
| grounded | 0.804 | 0.798 | **0.844** | 0.540 |

Nano-Jev is ~90–165× faster than the prompted LLM. The baselines are zero-shot on these
datasets and Nano-Jev is not, so read the caveats in
[results/v0.1_comparison.md](https://github.com/shubham10divakar/nano-jev/blob/main/results/v0.1_comparison.md).

## Tests

```bash
git clone https://github.com/shubham10divakar/nano-jev
cd nano-jev
pip install -e ".[test,train]"

pytest                                   # offline: builds a tiny random model, ~10 s
NANOJEV_NETWORK_TESTS=1 pytest           # also downloads the real v0.1 weights and checks them
```

What the tests cover (see [`tests/`](https://github.com/shubham10divakar/nano-jev/tree/main/tests)):

| File | Examples of what's checked |
|---|---|
| `test_decider.py` | every decision returns a probability distribution over its options; option order doesn't change probabilities; temperatures and config are loaded |
| `test_registry.py` | version names, `repo@tag` parsing, selection order (env > saved > default), local folders |
| `test_cli.py` | `--version`, `use` / `current` / `use --reset`, `relevance` and `decide` with `--json`, offline `list` |
| `test_calibration.py` | temperature fitting recovers a known temperature; ECE of an overconfident model |
| `test_network.py` | (opt-in) v0.1 is listed on the Hub, downloads, and makes sensible decisions |

### Smoke test on Kaggle / Colab / a fresh machine

- **Kaggle:** click the *Open in Kaggle* badge above (or upload
  [`examples/kaggle_smoke_test.ipynb`](https://github.com/shubham10divakar/nano-jev/blob/main/examples/kaggle_smoke_test.ipynb)),
  turn on *Settings → Internet*, then *Run All*.
- **Anywhere:**
  ```bash
  pip install nano-jev
  wget https://raw.githubusercontent.com/shubham10divakar/nano-jev/main/examples/smoke_test.py
  python smoke_test.py        # 9 checks, ends with ALL SMOKE TESTS PASSED
  ```

## Train your own

The repo also contains the full training pipeline (≈15 min on an RTX 3060):

```bash
pip install -e ".[train]"
python scripts/prepare_data.py --preset default   # HotpotQA + SQuAD 2.0 + MultiNLI -> data/ (v0.1 splits are included)
python scripts/train.py                           # -> runs/nano-jev-dev
python scripts/evaluate.py --model runs/nano-jev-dev
python scripts/baselines.py --which all           # bge-reranker + prompted Qwen3-4B (~35 min)
nano-jev use runs/nano-jev-dev                     # use your model everywhere
```

`--preset smoke` gives a 1-minute end-to-end run. Layout:

```
nanojev/        the package: decider (API), registry (weights), model, schema,
                calibration, report, data, __main__ (CLI)
scripts/        prepare_data, train, evaluate, baselines, demo
tests/          pytest suite
examples/       smoke_test.py, kaggle_smoke_test.ipynb
data/           v0.1 train / calib / test splits (JSONL)
results/        evaluation tables and baseline comparison
model_cards/    Hugging Face model card per released version
```

## Limitations

- **In-distribution evaluation only (v0.1).** Trained on the train splits of the same three
  datasets it is tested on; the baselines are zero-shot. No out-of-distribution results yet.
- **Groundedness** is below a prompted 4B LLM. Consider escalating low-confidence
  `grounded` calls to a larger model.
- **English, Wikipedia-style text.** Expect worse behaviour on other domains and languages.
- **Custom option sets** work but aren't trained for; treat those probabilities as rough.
- "partially relevant" follows HotpotQA rules: a supporting paragraph without the answer string.

## Licence

Code and weights: **Apache-2.0**.

The v0.1 weights start from `cross-encoder/ms-marco-MiniLM-L6-v2`, which was trained on
MS MARCO, whose terms are **non-commercial**. Commercial users should check those terms, or
wait for **v1.0**, which will be retrained on an MIT-licensed base with no MS MARCO lineage.

Training and evaluation data derive from [HotpotQA](https://hotpotqa.github.io/) (CC BY-SA 4.0),
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0) and
[MultiNLI](https://cims.nyu.edu/~sbowman/multinli/); the files in `data/` carry those licences.

## Roadmap

- **v1.0:** MIT-licensed base, held-out evaluation (MuSiQue, FEVER / RAGTruth), a
  grounded-decision cascade that escalates unsure cases to an LLM, stable API.
- Later: a larger model that reads all chunks in one pass, and a `next_action` decision
  for agentic RAG loops.
