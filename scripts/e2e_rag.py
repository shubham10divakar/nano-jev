"""Small end-to-end RAG evaluation (reviewer R1.4: "no downstream evidence").

Per question (data_e2e/*.jsonl from prepare_extra_data.py e2e):
  1. retrieve: BM25 over the question's own paragraph pool (HotpotQA distractor: 10,
     MuSiQue: 20 paragraphs), keep top-k (default 5);
  2. filter the top-k with one of
       none     keep all k
       bge      BGE-reranker-v2-m3 score -> 3-way LR fitted on data_clean calib relevance
                (same mapping as the paper's reranker baseline); keep if not "irrelevant"
       nanojev  Nano-Jev relevance decision; keep if not "irrelevant"
       qwen     Qwen3-4B prompted relevance judge (same prompt as the baseline); keep if
                not "irrelevant"
     (if a filter keeps nothing, the BM25 top-1 is kept);
  3. generate a short answer with Qwen3-4B-Instruct from the kept passages (greedy).
Reports EM / F1 (SQuAD normalisation, max over aliases), gold-passage recall and precision of
the kept set, passages kept, and filter / generation / total ms per question.

    python scripts/e2e_rag.py --nanojev runs/v2/nanojev/seed0 --sets data_e2e/hotpot.jsonl \
        data_e2e/musique.jsonl --out results/v2/e2e
"""

import argparse
import gc
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import llm_logits, reranker_scores  # noqa: E402
from nanojev import model as M  # noqa: E402
from nanojev.report import by_decision, read_jsonl  # noqa: E402
from nanojev.schema import DECISIONS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REL = DECISIONS["relevance"]


def free():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ metrics (SQuAD style)

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def em(pred, gold):
    return float(norm(pred) == norm(gold))


# ------------------------------------------------------------------ pipeline pieces

def tokenize(s):
    return re.findall(r"\w+", s.lower())


def retrieve(q, k):
    paras = q["paragraphs"]
    bm = BM25Okapi([tokenize(f"{p['title']} {p['text']}") for p in paras])
    scores = bm.get_scores(tokenize(q["question"]))
    return [int(i) for i in np.argsort(-scores)[:k]]


def rel_items(q, idxs):
    return [{"decision": "relevance", "question": REL.question.format(query=q["question"]),
             "options": list(REL.options), "label": 0,
             "state": f"{q['paragraphs'][i]['title']}: {q['paragraphs'][i]['text']}"} for i in idxs]


def gen_prompt(tok, q, idxs):
    ctx = "\n\n".join(f"[{n + 1}] {q['paragraphs'][i]['title']}: {q['paragraphs'][i]['text']}"
                      for n, i in enumerate(idxs))
    content = (f"Answer the question using the passages. Reply with a short answer phrase only "
               f"(a few words, or yes/no).\n\n{ctx}\n\nQuestion: {q['question']}\nAnswer:")
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(model, tok, prompts, batch_size=8, max_new_tokens=24):
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += [s.strip().split("\n")[0] for s in
                 tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["data_e2e/hotpot.jsonl", "data_e2e/musique.jsonl"])
    ap.add_argument("--nanojev", required=True)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--calib", default="data_clean", help="folder with calib.jsonl (BGE mapping)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/v2/e2e")
    args = ap.parse_args()

    sets = {Path(s).stem: read_jsonl(s)[: args.limit or None] for s in args.sets}
    t0 = time.time()
    topk = {name: [retrieve(q, args.k) for q in qs] for name, qs in sets.items()}
    bm25_ms = 1000 * (time.time() - t0) / sum(len(v) for v in sets.values())
    keep = {name: {} for name in sets}           # name -> filter -> [kept idx lists]
    fms = {name: {"none": 0.0} for name in sets}  # filter ms / question
    for name in sets:
        keep[name]["none"] = topk[name]

    def apply(name, filt, labels_flat):
        out, j = [], 0
        for q, idxs in zip(sets[name], topk[name]):
            lab = labels_flat[j : j + len(idxs)]
            j += len(idxs)
            kept = [i for i, l in zip(idxs, lab) if l != 0]
            out.append(kept or idxs[:1])
        keep[name][filt] = out

    # Nano-Jev
    nano, ntok = M.load(args.nanojev, DEVICE)
    for name, qs in sets.items():
        items = [it for q, idxs in zip(qs, topk[name]) for it in rel_items(q, idxs)]
        sync(); t = time.time()
        lg = M.score(nano, ntok, items, 512, DEVICE)
        sync(); fms[name]["nanojev"] = 1000 * (time.time() - t) / len(qs)
        apply(name, "nanojev", [int(x.argmax()) for x in lg])
    del nano
    free()

    # BGE reranker + 3-way LR fitted on calib relevance
    rtok = AutoTokenizer.from_pretrained(args.reranker)
    rr = AutoModelForSequenceClassification.from_pretrained(args.reranker).to(DEVICE).eval()
    cal = by_decision(read_jsonl(Path(args.calib) / "calib.jsonl"))["relevance"]
    lr = LogisticRegression(max_iter=1000).fit(reranker_scores(rr, rtok, cal)[:, None],
                                               [ex["label"] for ex in cal])
    for name, qs in sets.items():
        items = [it for q, idxs in zip(qs, topk[name]) for it in rel_items(q, idxs)]
        sync(); t = time.time()
        s = reranker_scores(rr, rtok, items)
        sync(); fms[name]["bge"] = 1000 * (time.time() - t) / len(qs)
        apply(name, "bge", lr.predict(s[:, None]).tolist())
    del rr
    free()

    # Qwen: relevance judge, then generation for every filter
    ltok = AutoTokenizer.from_pretrained(args.llm, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16, device_map=DEVICE,
                                               low_cpu_mem_usage=True).eval()
    report, lines = {}, ["| set | filter | EM | F1 | gold recall | gold precision | kept "
                         "| filter ms/q | gen ms/q | total ms/q |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, qs in sets.items():
        items = [it for q, idxs in zip(qs, topk[name]) for it in rel_items(q, idxs)]
        sync(); t = time.time()
        lg = llm_logits(llm, ltok, items)
        sync(); fms[name]["qwen"] = 1000 * (time.time() - t) / len(qs)
        apply(name, "qwen", lg.argmax(1).tolist())
        for filt in ("none", "bge", "nanojev", "qwen"):
            kept = keep[name][filt]
            sync(); t = time.time()
            answers = generate(llm, ltok, [gen_prompt(ltok, q, k) for q, k in zip(qs, kept)])
            sync(); gms = 1000 * (time.time() - t) / len(qs)
            ems = [max(em(a, g) for g in q["answers"]) for a, q in zip(answers, qs)]
            f1s = [max(f1(a, g) for g in q["answers"]) for a, q in zip(answers, qs)]
            rec, prec = [], []
            for q, k in zip(qs, kept):
                gold = {i for i, p in enumerate(q["paragraphs"]) if p["gold"]}
                rec.append(len(gold & set(k)) / max(1, len(gold)))
                prec.append(len(gold & set(k)) / len(k))
            r = {"n": len(qs), "em": float(np.mean(ems)), "f1": float(np.mean(f1s)),
                 "gold_recall": float(np.mean(rec)), "gold_precision": float(np.mean(prec)),
                 "kept": float(np.mean([len(k) for k in kept])), "bm25_ms": bm25_ms,
                 "filter_ms": fms[name][filt], "gen_ms": gms,
                 "total_ms": bm25_ms + fms[name][filt] + gms,
                 "examples": [{"q": q["question"], "pred": a, "gold": q["answers"]}
                              for q, a in list(zip(qs, answers))[:20]]}
            report[f"{name}/{filt}"] = r
            lines.append(f"| {name} | {filt} | {r['em']:.3f} | {r['f1']:.3f} | {r['gold_recall']:.3f} "
                         f"| {r['gold_precision']:.3f} | {r['kept']:.2f} | {r['filter_ms']:.1f} "
                         f"| {r['gen_ms']:.1f} | {r['total_ms']:.1f} |")
            print(lines[-1], flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e.json").write_text(json.dumps(report, indent=2))
    (out / "e2e.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
