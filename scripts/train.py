"""Fine-tune the Nano-Jev scorer on typed-decision examples.

    python scripts/train.py --data data --out runs/nano-jev-dev

If <data>/dev.jsonl exists (see scripts/make_dev_split.py) it selects the best epoch, and
calib.jsonl is not read at all. Otherwise the legacy v1.0 protocol is used: a slice of
calib.jsonl selects the epoch (calib is then also used for temperatures: a leak, R3.2).

Revision runs (clean protocol, v1.0 recipe):
    python scripts/train.py --data data_clean --base microsoft/MiniLM-L12-H384-uncased         --epochs 3 --lr 5e-5 --seed 0 --out runs/v2/nanojev/seed0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanojev import model as M  # noqa: E402
from nanojev.schema import DECISIONS, SCHEMA_VERSION  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def dev_nll(model, tok, examples, max_length, device):
    logits = M.score(model, tok, examples, max_length, device)
    labels = torch.tensor([ex["label"] for ex in examples])
    return float(torch.nn.functional.cross_entropy(torch.stack(logits), labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/nano-jev-dev")
    ap.add_argument("--base", default=M.DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16, help="examples (not option pairs)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dev-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="use a random fraction of train.jsonl (data-size ablation)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train = read_jsonl(Path(args.data) / "train.jsonl")
    if args.train_fraction < 1.0:
        train = random.Random(args.seed).sample(train, round(args.train_fraction * len(train)))
    dev_path = Path(args.data) / "dev.jsonl"
    if dev_path.exists():
        # Clean protocol: dev comes from the training pool; calib is never touched here.
        dev, dev_source = read_jsonl(dev_path), "dev.jsonl (from train pool)"
    else:
        # Legacy v1.0 protocol: dev loss on a slice of calib (only selects the best epoch).
        print("WARNING: no dev.jsonl - selecting the epoch on a slice of calib.jsonl (legacy)")
        calib = read_jsonl(Path(args.data) / "calib.jsonl")
        dev = random.Random(args.seed).sample(calib, min(args.dev_size, len(calib)))
        dev_source = "calib slice (legacy)"
    # Group by option count so each group's logits stack into one tensor.
    dev_by_k = {}
    for ex in dev:
        dev_by_k.setdefault(len(ex["options"]), []).append(ex)

    model, tok = M.load(args.base, device)

    def collate(batch):
        enc, g, p = M.encode(tok, batch, args.max_length)
        return enc, g, p, torch.tensor([ex["label"] for ex in batch])

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    def eval_dev():
        return sum(dev_nll(model, tok, exs, args.max_length, device) * len(exs)
                   for exs in dev_by_k.values()) / len(dev)

    out = Path(args.out)
    print(f"init dev NLL {eval_dev():.4f}  ({len(train)} train examples, {total} steps, {device})")
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, (enc, g, p, labels) in enumerate(loader, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = M.group_logits(model, enc, g, p, len(labels))
            loss = M.grouped_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} "
                      f"loss {running / 200:.4f}  {time.time() - t0:.0f}s")
                running = 0.0
        nll = eval_dev()
        print(f"epoch {epoch + 1} done in {time.time() - t0:.0f}s  dev NLL {nll:.4f}")
        if nll < best:
            best = nll
            model.save_pretrained(out)
            tok.save_pretrained(out)
            (out / "nanojev_config.json").write_text(json.dumps({
                "base": args.base, "max_length": args.max_length,
                "schema_version": SCHEMA_VERSION,
                "decisions": {k: list(v.options) for k, v in DECISIONS.items()},
                "epochs_trained": epoch + 1, "dev_nll": nll,
                "seed": args.seed, "dev_source": dev_source, "n_train": len(train),
                "train_fraction": args.train_fraction, "lr": args.lr,
                "epochs_max": args.epochs, "batch_size": args.batch_size,
            }, indent=2))
            print(f"  saved to {out}")


if __name__ == "__main__":
    main()
