"""One latency protocol for every system (fixes the inconsistent ms numbers across tables).

Definition used in the paper: milliseconds per decision = wall-clock time to score N test
examples of one decision type / N, measured
  * after 2 warm-up batches, with torch.cuda.synchronize() around the timed region;
  * including tokenization, excluding model loading;
  * GPU: batched, batch = 32 examples (Nano-Jev: all K option pairs of those examples in one
    forward; 1-pass: 32 sequences; BGE: 32 pairs; Qwen: 8 prompts, one forward, no generation);
  * CPU (--device cpu): batch = 1 example, torch threads = --threads (latency, not throughput).
Same N examples (first --n of each decision in --data/test.jsonl) for every system.

    python scripts/latency.py --systems nanojev:runs/v2/nanojev/seed0 onepass:runs/v2/onepass/seed0 \
        reranker:BAAI/bge-reranker-v2-m3 llm:Qwen/Qwen3-4B-Instruct-2507 --device cuda \
        --out results/v2/latency/gpu.json
    python scripts/latency.py --systems nanojev:runs/v2/nanojev/seed0 onepass:runs/v2/onepass/seed0 \
        reranker:BAAI/bge-reranker-v2-m3 llm:Qwen/Qwen3-4B-Instruct-2507 --device cpu --n 64 \
        --out results/v2/latency/cpu.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402
import onepass  # noqa: E402
from nanojev import model as M  # noqa: E402
from nanojev.report import by_decision, read_jsonl  # noqa: E402


def timeit(fn, items, bs, device, warmup=2):
    for i in range(warmup):
        fn(items[i * bs : (i + 1) * bs] or items[:bs])
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(items), bs):
        fn(items[i : i + bs])
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--data", default="data_clean")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    torch.set_num_threads(args.threads)
    baselines.DEVICE = dev
    test = {d: v[: args.n] for d, v in by_decision(read_jsonl(Path(args.data) / "test.jsonl")).items()}
    gpu = dev == "cuda"
    res = {"device": dev, "n_per_decision": args.n, "threads": args.threads if not gpu else None,
           "gpu_name": torch.cuda.get_device_name(0) if gpu else None, "systems": {}}

    for spec in args.systems:
        kind, _, name = spec.partition(":")
        print(f"== {spec} on {dev}", flush=True)
        if kind == "nanojev":
            model, tok = M.load(name, dev)
            bs = 32 if gpu else 1
            fn = lambda b: M.score(model, tok, b, 512, dev, batch_size=len(b))  # noqa: E731
        elif kind == "onepass":
            model, tok = onepass.OnePass.load(Path(name)).to(dev), AutoTokenizer.from_pretrained(name)
            bs = 32 if gpu else 1
            fn = lambda b: onepass.score(model, tok, b, 512, dev, batch_size=len(b))  # noqa: E731
        elif kind == "reranker":
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
            bs = 32 if gpu else 1
            fn = lambda b: baselines.reranker_scores(model, tok, b, batch_size=len(b))  # noqa: E731
        elif kind == "llm":
            tok = AutoTokenizer.from_pretrained(name, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.bfloat16 if gpu else torch.float32, device_map=dev,
                low_cpu_mem_usage=True).eval()
            bs = 8 if gpu else 1
            fn = lambda b: baselines.llm_logits(model, tok, b, batch_size=len(b))  # noqa: E731
        else:
            raise ValueError(spec)
        params = sum(p.numel() for p in model.parameters())
        per = {dec: timeit(fn, items, bs, dev) for dec, items in sorted(test.items())}
        res["systems"][spec] = {"kind": kind, "batch_size": bs, "params": params, "ms_per_decision": per}
        print("  " + "  ".join(f"{d}: {v:.2f} ms" for d, v in per.items()), flush=True)
        del model
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
