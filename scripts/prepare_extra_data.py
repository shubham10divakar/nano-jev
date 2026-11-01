"""Extra evaluation data for the revision (no training data here).

    python scripts/prepare_extra_data.py aggrefact   # -> data_aggrefact/{calib,test}.jsonl   (R1.4, R3.1)
    python scripts/prepare_extra_data.py custom      # -> data_custom/<task>/{calib,test}.jsonl (R1.2, R3.3)
    python scripts/prepare_extra_data.py e2e         # -> data_e2e/{hotpot,musique}.jsonl      (R1.4)
    python scripts/prepare_extra_data.py all

aggrefact: LLM-AggreFact (multi-sentence, LLM-generated claims over real documents). Test:
  per sub-dataset, up to --per-dataset balanced supported/unsupported claims from the test
  split. Calib: same from the dev split; used ONLY for the "re-fit temperature on the target
  domain" row. The main OOD protocol fits temperatures on data_clean/calib.jsonl.
custom: zero-shot topic / emotion classification with option sets never seen in training.
  calib (from train split) fits a temperature; test is from the test split.
e2e: HotpotQA validation questions and MuSiQue dev questions that are NOT in any calib /
  test / held-out file (the shuffles below match nanojev/data.py), with their full paragraph
  pools and gold flags, for the end-to-end RAG evaluation (scripts/e2e_rag.py).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanojev.data import HOTPOT, MUSIQUE, _example, _hotpot_paragraphs  # noqa: E402
from revision_utils import write_jsonl  # noqa: E402

AGGREFACT = "lytang/LLM-AggreFact"

# task -> (dataset, config, split_calib, split_test, text_fn, label_names, question)
CUSTOM = {
    "agnews": ("fancyzhx/ag_news", None, "train", "test", lambda r: r["text"],
               ["world news", "sports", "business", "science and technology"],
               "Which topic is this text about?"),
    "yahoo": ("community-datasets/yahoo_answers_topics", None, "train", "test",
              lambda r: f"{r['question_title']} {r['question_content']} {r['best_answer']}".strip(),
              ["society and culture", "science and mathematics", "health", "education and reference",
               "computers and internet", "sports", "business and finance",
               "entertainment and music", "family and relationships", "politics and government"],
              "Which topic is this text about?"),
    "emotion": ("dair-ai/emotion", None, "train", "test", lambda r: r["text"],
                ["sadness", "joy", "love", "anger", "fear", "surprise"],
                "Which emotion does the writer express?"),
}
LABEL_COL = {"agnews": "label", "yahoo": "topic", "emotion": "label"}


def summarize(name, rows):
    c = Counter((r["decision"], r.get("source", ""), r["label"]) for r in rows)
    print(f"{name}: {len(rows)}  " + "  ".join(f"{d}/{s}/{l}={n}" for (d, s, l), n in sorted(c.items())))


# ------------------------------------------------------------------ LLM-AggreFact

def aggrefact_split(split, per_dataset, rng):
    ds = load_dataset(AGGREFACT, split=split)
    buckets = defaultdict(lambda: ([], []))
    for row in ds:
        pos, neg = buckets[row["dataset"]]
        (pos if int(row["label"]) == 1 else neg).append(row)
    out = []
    for name, (pos, neg) in sorted(buckets.items()):
        k = min(per_dataset // 2, len(pos), len(neg))
        for row in rng.sample(pos, k) + rng.sample(neg, k):
            ex = _example("grounded", 0 if int(row["label"]) == 1 else 1, row["doc"],
                          f"aggrefact/{name}", claim=row["claim"])
            out.append(ex)
    rng.shuffle(out)
    return out


def build_aggrefact(args):
    rng = random.Random(args.seed)
    out = Path("data_aggrefact")
    test = aggrefact_split("test", args.per_dataset, rng)
    calib = aggrefact_split("dev", args.per_dataset_calib, rng)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "calib.jsonl", calib)
    summarize("aggrefact test", test)
    summarize("aggrefact calib", calib)


# ------------------------------------------------------------------ custom option sets

def build_custom(args):
    rng = random.Random(args.seed)
    for task, (name, config, s_cal, s_test, text_fn, labels, question) in CUSTOM.items():
        out = Path("data_custom") / task
        for split, n, fname in ((s_cal, args.custom_calib, "calib.jsonl"),
                                (s_test, args.custom_test, "test.jsonl")):
            ds = load_dataset(name, config, split=split).shuffle(seed=args.seed)
            per_class = n // len(labels)
            got, rows = Counter(), []
            for r in ds:
                y = int(r[LABEL_COL[task]])
                if got[y] >= per_class:
                    continue
                got[y] += 1
                rows.append({"decision": f"custom:{task}", "question": question,
                             "options": labels, "state": text_fn(r), "label": y, "source": task})
                if len(rows) >= per_class * len(labels):
                    break
            rng.shuffle(rows)
            write_jsonl(out / fname, rows)
            summarize(f"{task} {fname}", rows)


# ------------------------------------------------------------------ end-to-end RAG questions

def build_e2e(args):
    out = Path("data_e2e")
    # HotpotQA validation: questions [0, 1000) of shuffle(seed) are calib + test.
    ds = load_dataset(*HOTPOT, split="validation").shuffle(seed=args.seed)
    ds = ds.select(range(1000, 1000 + args.e2e_n))
    rows = []
    for r in ds:
        paras = _hotpot_paragraphs(r, max_sents=10**6)
        gold = set(r["supporting_facts"]["title"])
        rows.append({"id": r["id"], "question": r["question"], "answers": [r["answer"]],
                     "paragraphs": [{"title": t, "text": x, "gold": t in gold} for t, x in paras.items()]})
    write_jsonl(out / "hotpot.jsonl", rows)
    print(f"hotpot e2e: {len(rows)} questions")

    # MuSiQue dev: questions [0, 1000) of shuffle(seed) are the held-out set.
    ds = load_dataset(MUSIQUE[0], data_files={"dev": MUSIQUE[1]}, split="dev").shuffle(seed=args.seed)
    ds = ds.select(range(1000, min(len(ds), 1000 + args.e2e_n)))
    rows = []
    for r in ds:
        rows.append({"id": r["id"], "question": r["question"],
                     "answers": [a for a in [r["answer"], *r["answer_aliases"]] if a.strip()],
                     "paragraphs": [{"title": p["title"], "text": p["paragraph_text"].strip(),
                                     "gold": bool(p["is_supporting"])} for p in r["paragraphs"]]})
    write_jsonl(out / "musique.jsonl", rows)
    print(f"musique e2e: {len(rows)} questions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["aggrefact", "custom", "e2e", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-dataset", type=int, default=100, help="AggreFact test claims per sub-dataset")
    ap.add_argument("--per-dataset-calib", type=int, default=40, help="AggreFact dev claims per sub-dataset")
    ap.add_argument("--custom-calib", type=int, default=300)
    ap.add_argument("--custom-test", type=int, default=1000)
    ap.add_argument("--e2e-n", type=int, default=300)
    args = ap.parse_args()
    for what in (["aggrefact", "custom", "e2e"] if args.what == "all" else [args.what]):
        {"aggrefact": build_aggrefact, "custom": build_custom, "e2e": build_e2e}[what](args)
        print(json.dumps({"done": what}))


if __name__ == "__main__":
    main()
