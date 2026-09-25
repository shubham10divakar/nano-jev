"""Smoke test for an installed nano-jev package (Kaggle, Colab, or any fresh machine).

    pip install nano-jev
    python smoke_test.py

Needs internet the first time (downloads ~90 MB of weights from Hugging Face).
Exits non-zero if any check fails.
"""

import json
import subprocess
import sys
import time

import torch

import nanojev

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


@check("package imports and reports versions")
def _():
    print(f"  nano-jev {nanojev.__version__}, default weights {nanojev.DEFAULT_VERSION}, "
          f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")
    assert nanojev.__version__


@check("published weights are listed on Hugging Face (all versions)")
def _():
    models, online = nanojev.list_models()
    names = [m.name for m in models]
    print(f"  online={online}, versions={names}")
    assert online and {"v0.1", "v1.0"} <= set(names)


@check("default weights download and load")
def _():
    global d
    t0 = time.time()
    d = nanojev.load()
    print(f"  loaded version {d.version} ({d.config.get('params')}, base {d.config.get('base')}) "
          f"on {d.device} in {time.time() - t0:.1f}s")
    assert f"v{d.version}" == nanojev.DEFAULT_VERSION
    assert set(d.temperatures) == {"relevance", "sufficient", "grounded"}


@check("older weights (v0.1) still load and work")
def _():
    old = nanojev.load("v0.1")
    r = old.relevance("When was UCL founded?", ["University College London was founded in 1826."])[0]
    print(f"  v{old.version}: directly answers={r['directly answers']:.3f}")
    assert old.version == "0.1" and max(r, key=r.get) == "directly answers"


@check("relevance: answer passage beats an unrelated one")
def _():
    rel = d.relevance("When was UCL founded?", [
        "University College London was founded in 1826.",
        "The Analytical Engine was a proposed mechanical computer.",
    ])
    for r in rel:
        print("  ", {k: round(v, 3) for k, v in r.items()})
    assert max(rel[0], key=rel[0].get) == "directly answers"
    assert max(rel[1], key=rel[1].get) == "irrelevant"


@check("sufficient: two-hop question needs both passages")
def _():
    q = "In which year was the university attended by Ada Lovelace's tutor founded?"
    both = ["Augustus De Morgan tutored Ada Lovelace. He was a professor at University College London.",
            "University College London was founded in 1826."]
    full, partial = d.sufficient(q, both), d.sufficient(q, both[:1])
    print(f"   both passages: P(yes)={full['yes']:.3f}   first only: P(yes)={partial['yes']:.3f}")
    assert full["yes"] > partial["yes"]


@check("grounded: true claim supported, false claim not")
def _():
    ctx = "University College London was founded in 1826."
    true_, false_ = d.grounded("UCL was founded in 1826.", ctx), d.grounded("UCL was founded in 1900.", ctx)
    print(f"   true claim P(yes)={true_['yes']:.3f}   false claim P(yes)={false_['yes']:.3f}")
    assert true_["yes"] > 0.5 > false_["yes"]


@check("custom options return a valid distribution")
def _():
    out = d.decide("Which topic is this passage about?", ["computing", "cooking", "football"],
                   "The Analytical Engine was a proposed mechanical computer.")
    print("  ", {k: round(v, 3) for k, v in out.items()})
    assert abs(sum(out.values()) - 1) < 1e-5


@check("throughput: 256 relevance decisions")
def _():
    passages = ["University College London was founded in 1826."] * 256
    t0 = time.time()
    d.relevance("When was UCL founded?", passages)
    ms = 1000 * (time.time() - t0) / len(passages)
    print(f"  {ms:.2f} ms per decision on {d.device}")


@check("command-line tool works")
def _():
    out = subprocess.run([sys.executable, "-m", "nanojev", "relevance", "--json",
                          "-q", "When was UCL founded?", "-p", "UCL was founded in 1826."],
                         capture_output=True, text=True, check=True).stdout
    print("  ", json.loads(out)[0])


def main():
    failed = 0
    for name, fn in CHECKS:
        print(f"[....] {name}")
        try:
            fn()
            print(f"[ OK ] {name}")
        except Exception as e:  # report every check, then fail at the end
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    if failed:
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
