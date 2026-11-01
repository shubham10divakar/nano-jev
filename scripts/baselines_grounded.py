"""Compact task-specific groundedness baselines (reviewers R3.1, R1 minor 4).

Systems (each gives one P(supported) per (document, claim)):
  nli:<hf-id>        off-the-shelf NLI cross-encoder, P(entailment); long documents are split
                     into ~400-token chunks and the max over chunks is taken (SummaC/MiniCheck
                     style). Default: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (184M).
                     Also try MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (435M).
  minicheck:<name>   MiniCheck (Tang et al., 2024) via its official package; name is
                     roberta-large | deberta-v3-large | flan-t5-large.
                     pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
  alignscore:<ckpt>  AlignScore-base (Zha et al., 2023) via its official package; <ckpt> is the
                     path to AlignScore-base.ckpt (https://huggingface.co/yzha/AlignScore).
                     pip install git+https://github.com/yuh-zha/AlignScore.git  (old pins:
                     use a separate venv if it conflicts) + python -m spacy download en_core_web_sm
  nanojev:<path>     Nano-Jev with the same chunk-and-max over the document (truncation fix
                     for long documents; Nano-Jev's normal path truncates at 512 tokens).

Protocol: identical to scripts/baselines.py. A 1-feature logistic regression (log-odds of
P(supported) -> 2 options) and a temperature are fitted on <data>/calib.jsonl (in-domain
MultiNLI grounded) and applied unchanged to every --test-data set. Also reported:
"acc@0.5", the system's own zero-shot decision rule P(supported) >= 0.5, no fitting at all.

    python scripts/baselines_grounded.py --systems nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
        minicheck:roberta-large minicheck:deberta-v3-large alignscore:ckpts/AlignScore-base.ckpt \
        --data data_clean --test-data data_clean data_heldout data_aggrefact --out results/v2/grounded
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanojev import model as M  # noqa: E402
from nanojev.report import by_decision, decision_report, labels_of, read_jsonl, render_table  # noqa: E402
from nanojev.schema import subject_of  # noqa: E402
from revision_utils import pred_rows, write_jsonl  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(tok, text: str, size: int = 400, stride: int = 300) -> list[str]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= size:
        return [text]
    return [tok.decode(ids[i : i + size]) for i in range(0, max(1, len(ids) - size + stride), stride)]


# ------------------------------------------------------------------ scorers: list[(doc, claim)] -> P(supported)

class NLIScorer:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = labels.get("entailment", 0)

    @torch.no_grad()
    def __call__(self, pairs, batch_size=32):
        flat, owner = [], []
        for i, (doc, claim) in enumerate(pairs):
            for c in chunks(self.tok, doc):
                flat.append((c, claim))
                owner.append(i)
        probs = []
        for i in range(0, len(flat), batch_size):
            b = flat[i : i + batch_size]
            enc = self.tok([d for d, _ in b], [c for _, c in b], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                lg = self.model(**enc).logits.float()
            probs.append(torch.softmax(lg, -1)[:, self.ent].cpu().numpy())
        probs, out = np.concatenate(probs), np.zeros(len(pairs))
        np.maximum.at(out, np.array(owner), probs)
        return out


class NanoJevChunked:
    """Nano-Jev grounded decision, chunk-and-max over long documents. Uses raw (tau=1)
    P(yes); the calib-fitted LR + temperature on top handle the scale."""

    def __init__(self, path):
        self.model, self.tok = M.load(path, DEVICE)

    def __call__(self, pairs):
        items, owner = [], []
        for i, (doc, claim) in enumerate(pairs):
            for c in chunks(self.tok, doc):
                items.append({"question": f"Is this claim supported by the context? Claim: {claim}",
                              "options": ["yes", "no"], "state": c})
                owner.append(i)
        lg = torch.stack(M.score(self.model, self.tok, items, 512, DEVICE))
        p = torch.softmax(lg, 1)[:, 0].numpy()
        out = np.zeros(len(pairs))
        np.maximum.at(out, np.array(owner), p)
        return out


class MiniCheckScorer:
    def __init__(self, name):
        from minicheck.minicheck import MiniCheck  # official package
        self.scorer = MiniCheck(model_name=name, cache_dir="./ckpts")

    def __call__(self, pairs):
        _, raw_prob, _, _ = self.scorer.score(docs=[d for d, _ in pairs], claims=[c for _, c in pairs])
        return np.asarray(raw_prob, dtype=float)


class AlignScoreScorer:
    def __init__(self, ckpt):
        from alignscore import AlignScore  # official package
        self.scorer = AlignScore(model="roberta-base", batch_size=32,
                                 device="cuda:0" if DEVICE == "cuda" else "cpu",
                                 ckpt_path=ckpt, evaluation_mode="nli_sp")

    def __call__(self, pairs):
        return np.asarray(self.scorer.score(contexts=[d for d, _ in pairs],
                                            claims=[c for _, c in pairs]), dtype=float)


def make_scorer(spec):
    kind, _, arg = spec.partition(":")
    if kind == "nli":
        return NLIScorer(arg or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    if kind == "minicheck":
        return MiniCheckScorer(arg or "roberta-large")
    if kind == "alignscore":
        return AlignScoreScorer(arg)
    if kind == "nanojev":
        return NanoJevChunked(arg)
    raise ValueError(spec)


def slug(spec):
    return spec.replace("/", "_").replace(":", "-").replace("\\", "_").replace(".ckpt", "")


def logodds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def to_logits(p, lr):
    # LR classes are labels {0: yes, 1: no}; log-probs in that order = option logits.
    return torch.tensor(np.log(np.clip(lr.predict_proba(logodds(p)), 1e-12, None)), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--data", default="data_clean", help="folder with calib.jsonl (in-domain fit)")
    ap.add_argument("--test-data", nargs="+", default=["data_clean", "data_heldout", "data_aggrefact"])
    ap.add_argument("--out", default="results/v2/grounded")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    calib = by_decision(read_jsonl(Path(args.data) / "calib.jsonl"))["grounded"]
    tests = {Path(d).name: by_decision(read_jsonl(Path(d) / "test.jsonl"))["grounded"] for d in args.test_data}
    if args.limit:
        calib = calib[: args.limit]
        tests = {k: v[: args.limit] for k, v in tests.items()}
    pairs = lambda exs: [(ex["state"], subject_of(ex)) for ex in exs]  # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for spec in args.systems:
        print(f"== {spec}", flush=True)
        scorer = make_scorer(spec)
        p_cal = scorer(pairs(calib))
        lr = LogisticRegression(max_iter=1000).fit(logodds(p_cal), labels_of(calib).numpy())
        c_logits = to_logits(p_cal, lr)
        for name, exs in tests.items():
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            p = scorer(pairs(exs))
            ms = 1000 * (time.time() - t0) / len(exs)
            t_logits = to_logits(p, lr)
            y = labels_of(exs)
            rep = {"grounded": decision_report(c_logits, labels_of(calib), t_logits, y, ms)}
            rep["grounded"]["acc_at_0.5"] = float(((p < 0.5).astype(int) == y.numpy()).mean())
            by_src = {}
            pred = torch.softmax(t_logits / rep["grounded"]["temperature"], 1).argmax(1).numpy()
            for ex, pr in zip(exs, pred):
                by_src.setdefault(ex["source"], []).append(int(pr == ex["label"]))
            rep["grounded"]["acc_by_source"] = {k: float(np.mean(v)) for k, v in by_src.items()}
            table = render_table(f"{spec} on `{name}` (fitted on `{args.data}` calib)", rep)
            print(table + f"\nacc@0.5 (no fitting): {rep['grounded']['acc_at_0.5']:.3f}\n", flush=True)
            if not args.limit:
                stem = f"{slug(spec)}_{name}"
                (out / f"{stem}.json").write_text(json.dumps(rep, indent=2))
                (out / f"{stem}.md").write_text(table + "\n", encoding="utf-8")
                write_jsonl(out / f"{stem}_preds.jsonl",
                            pred_rows("grounded", exs, t_logits, rep["grounded"]["temperature"]))
        del scorer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
