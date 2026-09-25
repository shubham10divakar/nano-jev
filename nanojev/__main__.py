"""Command-line tool: list, download and choose weights; run decisions.

Installed as `nano-jev`; `python -m nanojev` is the same thing.

    python -m nanojev list                   # versions on the Hub (* = selected)
    python -m nanojev list --local runs      # also show local training runs
    python -m nanojev download v0.1
    python -m nanojev use v0.1               # or a Hub repo, or a local folder
    python -m nanojev use --reset            # back to the default version
    python -m nanojev current
    python -m nanojev relevance -q "When was UCL founded?" -p "UCL was founded in 1826."
    python -m nanojev sufficient -q QUERY -p PASSAGE [-p PASSAGE ...]
    python -m nanojev grounded --claim CLAIM --context TEXT
    python -m nanojev decide --question Q -o option1 -o option2 --state TEXT

Every decision command accepts --model to override the selected weights, and --json.
"""

import argparse
import json
import sys
from pathlib import Path

from . import registry


def _matches_selected(info: registry.ModelInfo, selected: str) -> bool:
    if info.source == "local":
        return Path(selected).is_dir() and Path(selected).resolve() == Path(info.name).resolve()
    repo, rev = registry.split_model(selected)
    return repo == registry.DEFAULT_REPO and rev == info.name


def cmd_list(args):
    models, online = registry.list_models(local_dirs=args.local, online=not args.offline)
    selected, origin = registry.get_selected()
    if not online and not args.offline:
        print("(Hub unreachable: showing downloaded versions only)\n")
    rows = [("", "MODEL", "SOURCE", "STATUS", "BASE", "PARAMS", "RELEASED", "DOWNLOADED")]
    for m in models:
        rows.append(("*" if _matches_selected(m, selected) else "", m.name, m.source,
                     m.status or "-", m.base or "-", m.params or "-", m.released or "-",
                     "yes" if m.downloaded else "no"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
    if len(rows) == 1:
        print("(no models found)")
    print(f"\nselected: {selected}  (from {origin})")


def cmd_download(args):
    model = args.model or registry.get_selected()[0]
    path = registry.resolve(model)
    print(f"{model} -> {path}")


def cmd_use(args):
    if args.reset:
        registry.set_selected(None)
        print(f"selection cleared; using {registry.get_selected()[0]}")
        return
    if not args.model:
        sys.exit("give a model (e.g. v0.1) or --reset")
    model = args.model
    if not Path(model).is_dir():
        # Check it exists before saving: download it (cached afterwards anyway).
        try:
            registry.resolve(model)
        except Exception as e:
            known = ", ".join(m.name for m in registry.list_models()[0]) or "none reachable"
            sys.exit(f"cannot find {model!r} ({type(e).__name__}). Available: {known}")
    path = registry.set_selected(model)
    print(f"selected {registry.get_selected()[0]}  (saved in {path})")


def cmd_current(args):
    selected, origin = registry.get_selected()
    print(f"selected: {selected}  (from {origin})")
    try:
        print(f"local path: {registry.resolve(selected, download=False)}")
    except Exception:
        print("not downloaded yet (it will download on first use)")


def _decider(args):
    from .decider import Decider
    d = Decider.from_pretrained(args.model)
    if not args.json:
        print(f"[model {args.model or registry.get_selected()[0]}, version {d.version}]",
              file=sys.stderr)
    return d


def _show(result, args):
    if args.json:
        print(json.dumps(result, indent=2))
        return
    for r in result if isinstance(result, list) else [result]:
        print("  ".join(f"{k}={v:.3f}" for k, v in r.items()))


def cmd_relevance(args):
    _show(_decider(args).relevance(args.query, args.passage), args)


def cmd_sufficient(args):
    _show(_decider(args).sufficient(args.query, args.passage), args)


def cmd_grounded(args):
    _show(_decider(args).grounded(args.claim, args.context), args)


def cmd_decide(args):
    _show(_decider(args).decide(args.question, args.option, args.state), args)


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="nano-jev",
                                 description="Nano-Jev: tiny calibrated decision model for RAG.")
    ap.add_argument("--version", action="version",
                    version=f"nano-jev {__version__} (default weights {registry.DEFAULT_VERSION})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list available weights")
    p.add_argument("--local", action="append", default=[], metavar="DIR",
                   help="also list model folders under DIR (repeatable)")
    p.add_argument("--offline", action="store_true", help="don't contact the Hub")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("download", help="download weights (default: selected)")
    p.add_argument("model", nargs="?")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("use", help="choose the weights to use by default")
    p.add_argument("model", nargs="?", help="version (v0.1), Hub repo[@tag], or local folder")
    p.add_argument("--reset", action="store_true", help="go back to the package default")
    p.set_defaults(fn=cmd_use)

    p = sub.add_parser("current", help="show the selected weights")
    p.set_defaults(fn=cmd_current)

    def decision(name, help_, fn):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--model", help="override the selected weights")
        p.add_argument("--json", action="store_true")
        p.set_defaults(fn=fn)
        return p

    p = decision("relevance", "how relevant is each passage to the query", cmd_relevance)
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-p", "--passage", action="append", required=True)

    p = decision("sufficient", "do the passages contain enough to answer", cmd_sufficient)
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-p", "--passage", action="append", required=True)

    p = decision("grounded", "is the claim supported by the context", cmd_grounded)
    p.add_argument("--claim", required=True)
    p.add_argument("--context", required=True)

    p = decision("decide", "any question with your own options", cmd_decide)
    p.add_argument("--question", required=True)
    p.add_argument("-o", "--option", action="append", required=True)
    p.add_argument("--state", required=True)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
