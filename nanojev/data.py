"""Build typed-decision examples from public datasets.

Each example: {decision, question, options, state, label, source}.
"""

import random

from datasets import load_dataset

from .schema import DECISIONS, format_passages

HOTPOT = ("hotpotqa/hotpot_qa", "distractor")
SQUAD2 = ("rajpurkar/squad_v2", None)
MNLI = ("nyu-mll/multi_nli", None)


def _example(decision: str, label: int, state: str, source: str, **fields) -> dict:
    d = DECISIONS[decision]
    return {
        "decision": decision,
        "question": d.question.format(**fields),
        "options": list(d.options),
        "state": state,
        "label": label,
        "source": source,
    }


def _hotpot_paragraphs(row, max_sents: int = 3) -> dict[str, str]:
    """Title -> paragraph text, keeping the first sentences plus every supporting one."""
    support: dict[str, set[int]] = {}
    for t, s in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        support.setdefault(t, set()).add(s)
    paras = {}
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        keep = [s for i, s in enumerate(sents) if i < max_sents or i in support.get(title, ())]
        paras[title] = " ".join(x.strip() for x in keep)
    return paras


def hotpot_examples(split: str, n: int, rng: random.Random, start: int = 0,
                    seed: int = 0) -> list[dict]:
    """Questions [start, start + n) of the split after a fixed shuffle."""
    ds = load_dataset(*HOTPOT, split=split).shuffle(seed=seed).select(range(start, start + n))
    out = []
    for row in ds:
        q, ans = row["question"], row["answer"].strip().lower()
        paras = _hotpot_paragraphs(row)
        gold = [t for t in dict.fromkeys(row["supporting_facts"]["title"]) if t in paras]
        distract = [t for t in paras if t not in gold]
        if len(gold) != 2 or len(distract) < 3:
            continue

        # relevance: gold paragraph with the answer string -> directly answers (2),
        # other gold -> partially relevant (1), distractor -> irrelevant (0)
        for t in gold:
            has_answer = ans not in ("yes", "no") and ans in paras[t].lower()
            out.append(_example("relevance", 2 if has_answer else 1, f"{t}: {paras[t]}",
                                "hotpot", query=q))
        for t in rng.sample(distract, 2):
            out.append(_example("relevance", 0, f"{t}: {paras[t]}", "hotpot", query=q))

        # sufficient: both gold + 1 distractor -> yes; one gold swapped for a second
        # distractor -> no. Same passage count, so length is not a shortcut.
        d1, d2 = rng.sample(distract, 2)
        pos = [gold[0], gold[1], d1]
        neg = [rng.choice(gold), d1, d2]
        for titles, label in ((pos, 0), (neg, 1)):
            rng.shuffle(titles)
            state = format_passages([(t, paras[t]) for t in titles])
            out.append(_example("sufficient", label, state, "hotpot", query=q))
    return out


def squad2_examples(split: str, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(SQUAD2[0], split=split).shuffle(seed=rng.randint(0, 10**6))
    pos, neg = [], []
    for row in ds:
        answerable = len(row["answers"]["text"]) > 0
        bucket = pos if answerable else neg
        if len(bucket) >= n // 2:
            if len(pos) >= n // 2 and len(neg) >= n // 2:
                break
            continue
        state = format_passages([(row["title"].replace("_", " "), row["context"])])
        bucket.append(_example("sufficient", 0 if answerable else 1, state, "squad2",
                               query=row["question"]))
    return pos + neg


def mnli_examples(split: str, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(MNLI[0], split=split).shuffle(seed=rng.randint(0, 10**6))
    pos, neg = [], []
    for row in ds:
        if row["label"] not in (0, 1, 2):
            continue
        entailed = row["label"] == 0
        bucket = pos if entailed else neg
        if len(bucket) >= n // 2:
            if len(pos) >= n // 2 and len(neg) >= n // 2:
                break
            continue
        bucket.append(_example("grounded", 0 if entailed else 1, row["premise"], "mnli",
                               claim=row["hypothesis"]))
    return pos + neg


# ------------------------------------------------------------------ held-out test sets
# Datasets never used for training or calibration, to measure generalisation.

MUSIQUE = ("bdsaglam/musique", "musique_ans_v1.0_dev.jsonl")  # CC BY 4.0
VITAMINC = ("tals/vitaminc", "test.jsonl")  # CC BY-SA 3.0


def musique_examples(n: int, rng: random.Random, seed: int = 0) -> list[dict]:
    """Relevance (all hop counts) and sufficiency (2-hop only) from MuSiQue dev.

    Same labelling rules as HotpotQA. Sufficiency uses 2-hop questions so the state has
    three passages, like the training data, and fits in 512 tokens.
    """
    ds = load_dataset(MUSIQUE[0], data_files={"dev": MUSIQUE[1]}, split="dev")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        q = row["question"]
        answers = [a.strip().lower() for a in [row["answer"], *row["answer_aliases"]] if a.strip()]
        paras = {p["idx"]: (p["title"], p["paragraph_text"].strip()) for p in row["paragraphs"]}
        gold = [p["idx"] for p in row["paragraphs"] if p["is_supporting"]]
        distract = [p["idx"] for p in row["paragraphs"] if not p["is_supporting"]]
        if len(gold) < 2 or len(distract) < 3:
            continue

        for i in gold:
            has_answer = any(a in paras[i][1].lower() for a in answers)
            out.append(_example("relevance", 2 if has_answer else 1,
                                f"{paras[i][0]}: {paras[i][1]}", "musique", query=q))
        for i in rng.sample(distract, 2):
            out.append(_example("relevance", 0, f"{paras[i][0]}: {paras[i][1]}", "musique",
                                query=q))

        if len(gold) == 2:
            d1, d2 = rng.sample(distract, 2)
            for idxs, label in (([gold[0], gold[1], d1], 0), ([rng.choice(gold), d1, d2], 1)):
                rng.shuffle(idxs)
                state = format_passages([paras[i] for i in idxs])
                out.append(_example("sufficient", label, state, "musique", query=q))
    return out


def vitaminc_examples(n: int, rng: random.Random, seed: int = 0) -> list[dict]:
    """Groundedness from VitaminC test: SUPPORTS -> yes; REFUTES / NOT ENOUGH INFO -> no."""
    ds = load_dataset(VITAMINC[0], data_files={"test": VITAMINC[1]}, split="test").shuffle(seed=seed)
    pos, neg = [], []
    for row in ds:
        supported = row["label"] == "SUPPORTS"
        bucket = pos if supported else neg
        if len(bucket) < n // 2:
            bucket.append(_example("grounded", 0 if supported else 1, row["evidence"],
                                   "vitaminc", claim=row["claim"]))
        if len(pos) >= n // 2 and len(neg) >= n // 2:
            break
    return pos + neg


def build_heldout(sizes: dict, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return (musique_examples(sizes["musique"], rng, seed)
            + vitaminc_examples(sizes["vitaminc"], rng, seed))


def build_all(sizes: dict, seed: int = 0) -> dict[str, list[dict]]:
    """Return {"train", "calib", "test"} example lists.

    Validation splits are halved into calib (temperature fitting) and test (reporting).
    """
    rng = random.Random(seed)
    train = (
        hotpot_examples("train", sizes["hotpot_train"], rng, seed=seed)
        + squad2_examples("train", sizes["squad_train"], rng)
        + mnli_examples("train", sizes["mnli_train"], rng)
    )
    # HotpotQA yields several examples per question: split by question, not by example.
    n_hp = sizes["hotpot_eval"]
    calib = hotpot_examples("validation", n_hp, rng, start=0, seed=seed)
    test = hotpot_examples("validation", n_hp, rng, start=n_hp, seed=seed)
    for part in (
        squad2_examples("validation", 2 * sizes["squad_eval"], rng),
        mnli_examples("validation_matched", 2 * sizes["mnli_eval"], rng),
    ):
        rng.shuffle(part)
        calib += part[: len(part) // 2]
        test += part[len(part) // 2 :]
    rng.shuffle(train)
    return {"train": train, "calib": calib, "test": test}
