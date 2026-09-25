"""Nano-Jev scorer: a cross-encoder that gives one logit per (question, option, state).

A decision's options are scored independently and softmaxed together, so option order
does not matter and new option sets work without retraining.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load(name_or_path: str, device: str | torch.device):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path, num_labels=1, ignore_mismatched_sizes=True
    )
    return model.to(device), tok


def encode(tok, items: list[dict], max_length: int):
    """Flatten items into (question+option, state) pairs.

    Returns the tokenized batch and, per pair, its item index and option position.
    """
    firsts, seconds, group_idx, pos_idx = [], [], [], []
    for g, item in enumerate(items):
        for p, opt in enumerate(item["options"]):
            firsts.append(f"question: {item['question']} option: {opt}")
            seconds.append(item["state"])
            group_idx.append(g)
            pos_idx.append(p)
    enc = tok(
        firsts,
        seconds,
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return enc, torch.tensor(group_idx), torch.tensor(pos_idx)


def group_logits(model, enc, group_idx, pos_idx, n_groups: int) -> torch.Tensor:
    """Run the encoder and return [n_groups, max_options] logits, padded with -inf."""
    flat = model(**enc).logits.squeeze(-1).float()
    out = torch.full((n_groups, int(pos_idx.max()) + 1), float("-inf"), device=flat.device)
    out[group_idx.to(flat.device), pos_idx.to(flat.device)] = flat
    return out


def grouped_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.to(logits.device))


@torch.no_grad()
def score(model, tok, items: list[dict], max_length: int, device, batch_size: int = 32):
    """Return a list of 1-D logit tensors (CPU), one per item."""
    model.eval()
    results = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        enc, g, p = encode(tok, chunk, max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            logits = group_logits(model, enc, g, p, len(chunk))
        for row, item in zip(logits.cpu(), chunk):
            results.append(row[: len(item["options"])])
    return results
