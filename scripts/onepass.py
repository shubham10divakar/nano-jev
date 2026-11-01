"""1-pass fine-tuned baseline (reviewers R1.1, R1.2, R3.1): K-pass vs 1-pass ablation.

Same backbone, same data, same epochs / lr / batch / dev selection as Nano-Jev, but a
standard classifier: ONE forward pass per example,

    [CLS] question-template [SEP] context [SEP]  ->  CLS vector  ->  head_d  ->  K_d logits

with a shared encoder and one linear head per decision type (relevance 3-way, sufficient
2-way, grounded 2-way). Options are positional, so it cannot take unseen option sets and
is not option-order invariant (the two properties the K-pass design buys).

Train (GPU, ~1/2.5 of Nano-Jev's time since K passes become 1):
    python scripts/onepass.py train --data data_clean --base microsoft/MiniLM-L12-H384-uncased \
        --epochs 3 --lr 5e-5 --seed 0 --out runs/v2/onepass/seed0

Evaluate (temperatures fitted on calib, same metrics / preds files as evaluate.py):
    python scripts/onepass.py eval --model runs/v2/onepass/seed0 --data data_clean \
        --results-dir results/v2/onepass/seed0 --name test
    python scripts/onepass.py eval --model runs/v2/onepass/seed0 --data data_clean \
        --test-data data_heldout --results-dir results/v2/onepass/seed0 --name heldout
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanojev.report import by_decision, decision_report, labels_of, render_table  # noqa: E402
from nanojev.schema import DECISIONS  # noqa: E402
from revision_utils import per_class, pred_rows, read_jsonl, set_seed, write_jsonl  # noqa: E402

HEADS = {name: len(d.options) for name, d in DECISIONS.items()}


class OnePass(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({d: nn.Linear(h, k) for d, k in HEADS.items()})

    def forward(self, enc, decisions: list[str]) -> list[torch.Tensor]:
        cls = self.dropout(self.encoder(**enc).last_hidden_state[:, 0])
        return [self.heads[d](cls[i]) for i, d in enumerate(decisions)]

    def save(self, out: Path, tok, meta: dict):
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out)
        tok.save_pretrained(out)
        torch.save(self.heads.state_dict(), out / "heads.pt")
        (out / "onepass_config.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path):
        m = cls(str(path))
        m.heads.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"))
        return m


def encode(tok, items, max_length):
    return tok([ex["question"] for ex in items], [ex["state"] for ex in items],
               truncation="only_second", max_length=max_length, padding=True, return_tensors="pt")


def loss_of(logits: list[torch.Tensor], labels: list[int]) -> torch.Tensor:
    return torch.stack([nn.functional.cross_entropy(lg.float()[None], torch.tensor([y], device=lg.device))
                        for lg, y in zip(logits, labels)]).mean()


@torch.no_grad()
def score(model, tok, items, max_length, device, batch_size=32) -> list[torch.Tensor]:
    model.eval()
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        enc = {k: v.to(device) for k, v in encode(tok, chunk, max_length).items()}
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            lgs = model(enc, [ex["decision"] for ex in chunk])
        out += [lg.float().cpu() for lg in lgs]
    return out


def cmd_train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = read_jsonl(Path(args.data) / "train.jsonl")
    dev_path = Path(args.data) / "dev.jsonl"
    if not dev_path.exists():
        sys.exit("needs <data>/dev.jsonl - run scripts/make_dev_split.py first (clean protocol)")
    dev = read_jsonl(dev_path)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = OnePass(args.base).to(device)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: (encode(tok, b, args.max_length),
                                              [ex["decision"] for ex in b], [ex["label"] for ex in b]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    def dev_nll():
        lgs = score(model, tok, dev, args.max_length, device)
        return float(torch.stack([nn.functional.cross_entropy(lg[None], torch.tensor([ex["label"]]))
                                  for lg, ex in zip(lgs, dev)]).mean())

    print(f"init dev NLL {dev_nll():.4f}  ({len(train)} train, {total} steps, {device})")
    best, t_start = float("inf"), time.time()
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, (enc, decs, labels) in enumerate(loader, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(enc, decs)
            loss = loss_of(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {running / 200:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
                running = 0.0
        nll = dev_nll()
        print(f"epoch {epoch + 1} done in {time.time() - t0:.0f}s  dev NLL {nll:.4f}", flush=True)
        if nll < best:
            best = nll
            model.save(Path(args.out), tok, {
                "kind": "onepass", "base": args.base, "max_length": args.max_length,
                "heads": HEADS, "epochs_trained": epoch + 1, "dev_nll": nll, "seed": args.seed,
                "lr": args.lr, "epochs_max": args.epochs, "batch_size": args.batch_size,
                "n_train": len(train), "train_minutes_so_far": (time.time() - t_start) / 60})
            print(f"  saved to {args.out}")


def cmd_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(args.model)
    cfg = json.loads((path / "onepass_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path)
    model = OnePass.load(path).to(device)
    calib = by_decision(read_jsonl(Path(args.data) / "calib.jsonl"))
    test = by_decision(read_jsonl(Path(args.test_data or args.data) / "test.jsonl"))
    report, preds = {}, []
    for dec in sorted(test):
        c = torch.stack(score(model, tok, calib[dec], cfg["max_length"], device))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        t = torch.stack(score(model, tok, test[dec], cfg["max_length"], device))
        ms = 1000 * (time.time() - t0) / len(test[dec])
        report[dec] = decision_report(c, labels_of(calib[dec]), t, labels_of(test[dec]), ms)
        report[dec]["per_class"] = per_class(t, labels_of(test[dec]), test[dec][0]["options"],
                                             report[dec]["temperature"])
        preds += pred_rows(dec, test[dec], t, report[dec]["temperature"])
    table = render_table(f"1-pass baseline — `{args.model}` on `{args.test_data or args.data}`", report)
    print(table)
    od = Path(args.results_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{args.name}.json").write_text(json.dumps(report, indent=2))
    (od / f"{args.name}.md").write_text(table + "\n", encoding="utf-8")
    write_jsonl(od / f"{args.name}_preds.jsonl", preds)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data", default="data_clean")
    tr.add_argument("--out", required=True)
    tr.add_argument("--base", default="microsoft/MiniLM-L12-H384-uncased")
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--max-length", type=int, default=512)
    tr.add_argument("--seed", type=int, default=0)
    ev = sub.add_parser("eval")
    ev.add_argument("--model", required=True)
    ev.add_argument("--data", default="data_clean")
    ev.add_argument("--test-data")
    ev.add_argument("--results-dir", required=True)
    ev.add_argument("--name", default="test")
    args = ap.parse_args()
    random.seed(0)
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
