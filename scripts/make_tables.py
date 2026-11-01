"""Generate every LaTeX table body and every number quoted in the paper text from the result
files of ONE protocol (fixes the Table 7/8 mismatch, the latency inconsistency and the
placeholder values by construction).

    python scripts/make_tables.py --results results/v2 --out paper_revision/tables

Writes paper_revision/tables/<name>.tex (tabular only; captions stay in the paper, which
\\input's these files) and numbers.tex (\\newcommand macros used in the prose). Anything not
yet measured is typeset as a red \\ph{--}, so `grep -c 'ph{' tables/*.tex` counts what is
still missing. Run it again after every experiment.

Expected layout under --results (see paper_revision/RUNBOOK.md):
  nanojev/seed*/{test,heldout,aggrefact,test_balanced,heldout_balanced}.json
  onepass/seed*/{test,heldout}.json         v0.1/{test,heldout}.json
  baselines/{reranker,llm}{,_heldout,_aggrefact}.json
  grounded/<system>_<data_clean|data_heldout|data_aggrefact>.json
  cascade/seed0/cascade.json                stats/*.json
  latency/{gpu,cpu}.json                    truncation/{test,heldout}.json
  custom/<model>/<task>.json  custom/llm_<task>.json  custom/probe_onepass.json
  ablation_data/frac{10,25,50}/test.json    e2e/e2e.json
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

PH = r"\ph{--}"
DECS = ["relevance", "sufficient", "grounded"]
SHORT = {"relevance": "rel.", "sufficient": "suf.", "grounded": "grd."}

R: Path = Path("results/v2")
OUT: Path = Path("paper_revision/tables")
NUM: dict[str, str] = {}


# ------------------------------------------------------------------ loading helpers

def J(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def seeds(kind, name):
    """All seed result dicts for e.g. ('nanojev', 'test')."""
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(R / kind / "seed*" / f"{name}.json")))]


def get(d, *path):
    for k in path:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def ms(runs, *path):
    """mean, std over a list of result dicts at path; (None, None) if missing."""
    vals = [get(r, *path) for r in runs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def f3(x, d=3, lead=True):
    if x is None:
        return PH
    s = f"{x:.{d}f}"
    return s if lead else s.lstrip("0") if x >= 0 else "-" + s[1:].lstrip("0")


def pm(m, s, d=3):
    if m is None:
        return PH
    return f3(m, d) + (rf"{{\scriptsize$\pm${f3(s, d, lead=False)}}}" if s is not None else "")


def pct(x, d=0):
    return PH if x is None else f"{100 * x:.{d}f}\\%"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body.strip() + "\n", encoding="utf-8")


def num(name, value):
    NUM[name] = value if value is not None else PH


def acc(r, dec):
    return get(r, dec, "calibrated", "accuracy")


# ------------------------------------------------------------------ tables

def t_datasplit(data_dirs):
    files = {"train": "data_clean/train.jsonl", "dev": "data_clean/dev.jsonl",
             "calib": "data_clean/calib.jsonl", "test": "data_clean/test.jsonl"}
    counts = {}
    for split, f in files.items():
        if Path(f).exists():
            with open(f, encoding="utf-8") as fh:
                counts[split] = Counter((json.loads(l)["decision"], json.loads(l)["source"]) for l in fh)
    held = Counter()
    for f in data_dirs:
        if Path(f).exists():
            with open(f, encoding="utf-8") as fh:
                held.update((json.loads(l)["decision"], json.loads(l)["source"].split("/")[0]) for l in fh)
    name = {"hotpot": "HotpotQA", "squad2": "SQuAD~2.0", "mnli": "MultiNLI", "musique": "MuSiQue",
            "vitaminc": "VitaminC", "aggrefact": "LLM-AggreFact"}
    rows = [r"\begin{tabular}{llrrrr}", r"\toprule",
            r"\textbf{Source} & \textbf{Decision} & \textbf{Train} & \textbf{Dev} & \textbf{Calib} & \textbf{Test} \\",
            r"\midrule"]
    keys = [("relevance", "hotpot"), ("sufficient", "hotpot"), ("sufficient", "squad2"), ("grounded", "mnli")]
    tot = Counter()
    for k in keys:
        cells = []
        for split in ("train", "dev", "calib", "test"):
            n = counts.get(split, {}).get(k)
            tot[split] += n or 0
            cells.append(f"{n:,}" if n is not None else PH)
        rows.append(f"{name[k[1]]} & {k[0]} & " + " & ".join(cells) + r" \\")
    rows += [r"\midrule", r"\textbf{Total} & & " + " & ".join(f"{tot[s]:,}" if tot[s] else PH
                                                            for s in ("train", "dev", "calib", "test")) + r" \\",
             r"\midrule"]
    for k in [("relevance", "musique"), ("sufficient", "musique"), ("grounded", "vitaminc"), ("grounded", "aggrefact")]:
        n = held.get(k)
        rows.append(f"{name[k[1]]} (held-out) & {k[0]} & --- & --- & --- & {f'{n:,}' if n else PH}" + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("datasplit", "\n".join(rows))
    for split in ("train", "dev", "calib", "test"):
        num(f"N{split.capitalize()}", f"{tot[split]:,}" if tot[split] else None)
    for k, macro in ((("relevance", "hotpot"), "NTestRel"), (("grounded", "mnli"), "NTestGrd")):
        n = counts.get("test", {}).get(k)
        num(macro, f"{n:,}" if n else None)
    suf = sum(counts.get("test", {}).get(k, 0) for k in [("sufficient", "hotpot"), ("sufficient", "squad2")])
    num("NTestSuf", f"{suf:,}" if suf else None)


def t_main():
    nj_t, nj_h = seeds("nanojev", "test"), seeds("nanojev", "heldout")
    op_t, op_h = seeds("onepass", "test"), seeds("onepass", "heldout")
    systems = [
        (r"BGE-reranker-v2-m3$^\dagger$", "568\\,M", [J("baselines/reranker.json")], [J("baselines/reranker_heldout.json")]),
        (r"Qwen3-4B-Instruct$^\dagger$", "4\\,B", [J("baselines/llm.json")], [J("baselines/llm_heldout.json")]),
        (r"CaliDec v0.1", "22.7\\,M", [J("v0.1/test.json")], [J("v0.1/heldout.json")]),
        (r"1-pass classifier", "33.4\\,M", op_t, op_h),
        (r"\textbf{CaliDec (K-pass)}", "33.4\\,M", nj_t, nj_h),
    ]
    rows = [r"\begin{tabular}{lcccccccc}", r"\toprule",
            r"& & \multicolumn{3}{c}{\textbf{Test accuracy}} & \multicolumn{3}{c}{\textbf{Held-out accuracy}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"\textbf{System} & \textbf{Params} & rel. & suf. & grd. & rel. & suf. & grd. \\", r"\midrule"]
    for label, params, t, h in systems:
        t, h = [x for x in t if x], [x for x in h if x]
        cells = [pm(*ms(t, d, "calibrated", "accuracy")) for d in DECS] + \
                [pm(*ms(h, d, "calibrated", "accuracy")) for d in DECS]
        rows.append(f"{label} & {params} & " + " & ".join(cells) + r" \\")
        if label.startswith(r"Qwen"):
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("main", "\n".join(rows))
    for d in DECS:
        D = d[:3].capitalize()
        num(f"NJ{D}Test", f3(ms(nj_t, d, "calibrated", "accuracy")[0]))
        num(f"NJ{D}Held", f3(ms(nj_h, d, "calibrated", "accuracy")[0]))
        num(f"OP{D}Test", f3(ms(op_t, d, "calibrated", "accuracy")[0]))
        num(f"OP{D}Held", f3(ms(op_h, d, "calibrated", "accuracy")[0]))
        num(f"LLM{D}Test", f3(acc(J("baselines/llm.json"), d)))
        num(f"LLM{D}Held", f3(acc(J("baselines/llm_heldout.json"), d)))
        num(f"BGE{D}Test", f3(acc(J("baselines/reranker.json"), d)))
        num(f"BGE{D}Held", f3(acc(J("baselines/reranker_heldout.json"), d)))
    num("NSeeds", str(len(nj_t)) if nj_t else None)


def t_calibration():
    systems = [("CaliDec", seeds("nanojev", "test"), seeds("nanojev", "heldout")),
               ("1-pass", seeds("onepass", "test"), seeds("onepass", "heldout")),
               ("CaliDec v0.1", [J("v0.1/test.json")], [J("v0.1/heldout.json")]),
               ("BGE-reranker", [J("baselines/reranker.json")], [J("baselines/reranker_heldout.json")]),
               ("Qwen3-4B", [J("baselines/llm.json")], [J("baselines/llm_heldout.json")])]
    rows = [r"\begin{tabular}{llcccccc}", r"\toprule",
            r"& & \multicolumn{3}{c}{\textbf{Test}} & \multicolumn{3}{c}{\textbf{Held-out}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"\textbf{System} & \textbf{Dec.} & ECE$_{\mathrm{raw}}$ & ECE$_{\mathrm{cal}}$ & $\tau$ & ECE$_{\mathrm{raw}}$ & ECE$_{\mathrm{cal}}$ & $\Delta$ECE$_{\mathrm{cal}}$ \\",
            r"\midrule"]
    for si, (label, t, h) in enumerate(systems):
        t, h = [x for x in t if x], [x for x in h if x]
        for i, d in enumerate(DECS):
            er_t, ec_t = ms(t, d, "raw", "ece")[0], ms(t, d, "calibrated", "ece")[0]
            tau = ms(t, d, "temperature")[0]
            er_h, ec_h = ms(h, d, "raw", "ece")[0], ms(h, d, "calibrated", "ece")[0]
            delta = None if ec_h is None or ec_t is None else ec_h - ec_t
            first = rf"\multirow{{3}}{{*}}{{{label}}}" if i == 0 else ""
            rows.append(f"{first} & {SHORT[d]} & {f3(er_t)} & {f3(ec_t)} & {f3(tau, 2)} & {f3(er_h)} & {f3(ec_h)} & "
                        + (PH if delta is None else f"{delta:+.3f}".replace("-", "$-$")) + r" \\")
        if si < len(systems) - 1:
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("calibration", "\n".join(rows))
    nj_t, nj_h = seeds("nanojev", "test"), seeds("nanojev", "heldout")
    e_t = [ms(nj_t, d, "calibrated", "ece")[0] for d in DECS]
    e_h = [ms(nj_h, d, "calibrated", "ece")[0] for d in DECS]
    num("NJEceTestMin", f3(min(e_t)) if None not in e_t else None)
    num("NJEceTestMax", f3(max(e_t)) if None not in e_t else None)
    num("NJEceHeldMin", f3(min(e_h)) if None not in e_h else None)
    num("NJEceHeldMax", f3(max(e_h)) if None not in e_h else None)
    for d in DECS:
        num(f"NJTau{d[:3].capitalize()}", f3(ms(nj_t, d, "temperature")[0], 2))


def cascade_rows(dec, name, raw=False):
    c = J("cascade/seed0/cascade.json")
    r = get(c, f"{dec}@{name}")
    return (r or {}).get("rows_raw" if raw else "rows", []), r


def at(rows, t):
    for r in rows:
        if abs(r["threshold"] - t) < 1e-9:
            return r
    return None


def t_cascade_grounded():
    test, rt = cascade_rows("grounded", "data_clean")
    held, rh = cascade_rows("grounded", "data_heldout")
    chosen = (rt or {}).get("chosen_threshold")
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r"& \multicolumn{3}{c}{\textbf{Test (MultiNLI)}} & \multicolumn{2}{c}{\textbf{Held-out (VitaminC)}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
            r"\textbf{Strategy} & \textbf{\% esc.} & \textbf{Acc} & \textbf{ms/dec} & \textbf{\% esc.} & \textbf{Acc} \\",
            r"\midrule"]
    for t in [0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]:
        a, b = at(test, t), at(held, t)
        label = ("CaliDec only" if t == 0 else "LLM only" if t > 1 else f"Cascade, $t={t:.2f}$")
        if chosen is not None and abs(t - chosen) < 1e-9:
            label = r"\textbf{" + label + r"} (chosen)"
        rows.append(f"{label} & {pct(get(a, 'escalated'))} & {f3(get(a, 'accuracy'))} & "
                    f"{f3(get(a, 'ms_per_decision'), 1)} & {pct(get(b, 'escalated'))} & {f3(get(b, 'accuracy'))}" + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("cascade_grounded", "\n".join(rows))
    a = at(test, chosen) if chosen is not None else None
    b = at(held, chosen) if chosen is not None else None
    num("CascT", None if chosen is None else f"{chosen:.2f}")
    num("CascGrdTestAcc", f3(get(a, "accuracy")))
    num("CascGrdTestEsc", pct(get(a, "escalated")))
    num("CascGrdHeldAcc", f3(get(b, "accuracy")))
    num("CascGrdHeldEsc", pct(get(b, "escalated")))
    num("CascGrdMs", f3(get(a, "ms_per_decision"), 1))
    llm_t, llm_h, nano_h = at(test, 1.01), at(held, 1.01), at(held, 0.0)
    num("CascGrdLLMHeldAcc", f3(get(llm_h, "accuracy")))
    num("CascGrdNanoHeldAcc", f3(get(nano_h, "accuracy")))
    num("CascGrdLLMMs", f3(get(llm_t, "ms_per_decision"), 1))
    sp = (llm_t["ms_per_decision"] / a["ms_per_decision"]) if llm_t and get(a, "ms_per_decision") else None
    num("CascGrdSpeedup", None if sp is None else f"{sp:.1f}")
    # smallest threshold at which the cascade reaches the LLM's held-out accuracy
    parity = [r["threshold"] for r in held if llm_h and r["threshold"] <= 1 and r["accuracy"] >= llm_h["accuracy"]]
    num("CascGrdHeldParityT", f"{min(parity):.2f}" if parity else None)


def t_cascade_all():
    st = J("stats/cascade_seed0.json") or {}
    rows = [r"\begin{tabular}{lccccccc}", r"\toprule",
            r"\textbf{Decision / set} & $t$ & \textbf{\% esc.} & \textbf{Cascade} & \textbf{CaliDec} & \textbf{LLM} & $p_{\text{vs CaliDec}}$ & $p_{\text{vs LLM}}$ \\",
            r"\midrule"]
    for d in DECS:
        for name, lab in (("data_clean", "test"), ("data_heldout", "held-out")):
            rs, r = cascade_rows(d, name)
            t = (r or {}).get("chosen_threshold")
            a = at(rs, t) if t is not None else None
            nano, llm = at(rs, 0.0), at(rs, 1.01)
            s = st.get(f"{d}@{name}", {})
            p1, p2 = get(s, "vs_nano", "mcnemar_p"), get(s, "vs_llm", "mcnemar_p")
            rows.append(f"{d} / {lab} & {PH if t is None else f'{t:.2f}'} & {pct(get(a, 'escalated'))} & "
                        f"{f3(get(a, 'accuracy'))} & {f3(get(nano, 'accuracy'))} & {f3(get(llm, 'accuracy'))} & "
                        f"{PH if p1 is None else f'{p1:.2g}'} & {PH if p2 is None else f'{p2:.2g}'}" + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("cascade_all", "\n".join(rows))


def t_ablation_temp():
    rs, r = cascade_rows("grounded", "data_clean")
    rr, _ = cascade_rows("grounded", "data_clean", raw=True)
    tc, tr = (r or {}).get("chosen_threshold"), (r or {}).get("chosen_threshold_raw")
    tau = get(J("cascade/seed0/cascade.json"), "grounded@calib", "t_nano")
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"\textbf{Strategy} & $\tau$ & $t$ & \textbf{Acc} & \textbf{\% escalated} \\", r"\midrule"]
    for label, src, t, tv in (("Cascade, raw, $t=0.80$", rr, 0.8, "1.00"),
                              ("Cascade, cal., $t=0.80$", rs, 0.8, f3(tau, 2)),
                              ("Cascade, raw, $t$ chosen on calib", rr, tr, "1.00"),
                              ("Cascade, cal., $t$ chosen on calib", rs, tc, f3(tau, 2))):
        a = at(src, t) if t is not None else None
        rows.append(f"{label} & {tv} & {PH if t is None else f'{t:.2f}'} & {f3(get(a, 'accuracy'))} & {pct(get(a, 'escalated'))}" + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("ablation_temp", "\n".join(rows))


def t_detailed():
    nj = seeds("nanojev", "test")
    rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"\textbf{Decision} & $n$ & $\tau$ & \textbf{Acc} & \textbf{Macro-F1} & \textbf{NLL} & \textbf{Brier} \\", r"\midrule"]
    for d in DECS:
        n = get(nj[0], d, "calibrated", "n") if nj else None
        rows.append(f"{d} & {PH if n is None else f'{n:,}'} & {f3(ms(nj, d, 'temperature')[0], 2)} & "
                    + " & ".join(pm(*ms(nj, d, "calibrated", k)) for k in ("accuracy", "macro_f1", "nll", "brier")) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("detailed", "\n".join(rows))


def t_perclass():
    nj = seeds("nanojev", "test")
    opts = ["irrelevant", "partially relevant", "directly answers"]
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\", r"\midrule"]
    for o in opts:
        sup = get(nj[0], "relevance", "per_class", o, "support") if nj else None
        rows.append(f"{o} & " + " & ".join(pm(*ms(nj, "relevance", "per_class", o, k)) for k in ("precision", "recall", "f1"))
                    + f" & {PH if sup is None else sup}" + r" \\")
    macro = [ms([{"m": float(np.mean([get(r, 'relevance', 'per_class', o, k) for o in opts]))} for r in nj
                 if get(r, "relevance", "per_class")], "m") for k in ("precision", "recall", "f1")]
    rows += [r"\midrule", r"\textbf{Macro avg} & " + " & ".join(pm(*m) for m in macro) + r" & \\",
             r"\bottomrule", r"\end{tabular}"]
    write("perclass", "\n".join(rows))


def t_latency():
    rows = [r"\begin{tabular}{lccccccc}", r"\toprule",
            r"& & \multicolumn{3}{c}{\textbf{GPU ms/decision}} & \multicolumn{3}{c}{\textbf{CPU ms/decision}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"\textbf{System} & \textbf{Params} & rel. & suf. & grd. & rel. & suf. & grd. \\", r"\midrule"]
    gpu, cpu = J("latency/gpu.json"), J("latency/cpu.json")

    def find(res, kind):
        for spec, v in (get(res, "systems") or {}).items():
            if v["kind"] == kind:
                return v
        return None

    for kind, label in (("nanojev", "CaliDec (K-pass)"), ("onepass", "1-pass classifier"),
                        ("reranker", "BGE-reranker-v2-m3"), ("llm", "Qwen3-4B-Instruct")):
        g, c = find(gpu, kind), find(cpu, kind)
        p = (g or c or {}).get("params")
        params = PH if p is None else f"{p / 1e6:.1f}" + r"\,M"
        rows.append(f"{label} & {params} & "
                    + " & ".join(f3(get(g, "ms_per_decision", d), 1) for d in DECS) + " & "
                    + " & ".join(f3(get(c, "ms_per_decision", d), 1) for d in DECS) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("latency", "\n".join(rows))
    nj, llm = find(gpu, "nanojev"), find(gpu, "llm")
    if nj and llm:
        sp = [llm["ms_per_decision"][d] / nj["ms_per_decision"][d] for d in DECS]
        num("SpeedupMin", f"{min(sp):.0f}")
        num("SpeedupMax", f"{max(sp):.0f}")
        num("NJMsMin", f"{min(nj['ms_per_decision'].values()):.1f}")
        num("NJMsMax", f"{max(nj['ms_per_decision'].values()):.1f}")
    else:
        for k in ("SpeedupMin", "SpeedupMax", "NJMsMin", "NJMsMax"):
            num(k, None)


def t_ablation_data():
    rows = [r"\begin{tabular}{lccc}", r"\toprule",
            r"\textbf{Training fraction} & \textbf{Rel. acc} & \textbf{Suf. acc} & \textbf{Grd. acc} \\", r"\midrule"]
    for frac in (10, 25, 50):
        r = J(f"ablation_data/frac{frac}/test.json")
        rows.append(f"{frac}\\% & " + " & ".join(f3(acc(r, d)) for d in DECS) + r" \\")
    s0 = J("nanojev/seed0/test.json")
    rows += [r"100\% (seed 0) & " + " & ".join(f3(acc(s0, d)) for d in DECS) + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    write("ablation_data", "\n".join(rows))


def t_onepass():
    nj_t, nj_h, op_t, op_h = (seeds("nanojev", "test"), seeds("nanojev", "heldout"),
                              seeds("onepass", "test"), seeds("onepass", "heldout"))
    cmp = J("stats/nanojev_vs_onepass_test.json") or {}
    gpu = J("latency/gpu.json")
    lat = {v["kind"]: v for v in (get(gpu, "systems") or {}).values()}
    rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"& \multicolumn{2}{c}{\textbf{K-pass (CaliDec)}} & \multicolumn{2}{c}{\textbf{1-pass}} & & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"\textbf{Decision} & Test & Held-out & Test & Held-out & $\Delta_{\text{test}}$ & McNemar $p$ \\", r"\midrule"]
    for d in DECS:
        c = cmp.get(d, {})
        diff = PH if "diff" not in c else f"{c['diff']:+.3f}"
        p = PH if "mcnemar_p" not in c else f"{c['mcnemar_p']:.2g}"
        accs = [pm(*ms(runs, d, "calibrated", "accuracy")) for runs in (nj_t, nj_h, op_t, op_h)]
        rows.append(f"{d} & " + " & ".join(accs) + f" & {diff} & {p}" + r" \\")
    rows += [r"\midrule",
             r"ECE$_{\mathrm{cal}}$ (test, rel/suf/grd) & \multicolumn{2}{c}{"
             + " / ".join(f3(ms(nj_t, d, "calibrated", "ece")[0]) for d in DECS) + r"} & \multicolumn{2}{c}{"
             + " / ".join(f3(ms(op_t, d, "calibrated", "ece")[0]) for d in DECS) + r"} & & \\",
             r"GPU ms/decision (rel/suf/grd) & \multicolumn{2}{c}{"
             + " / ".join(f3(get(lat.get("nanojev"), "ms_per_decision", d), 1) for d in DECS) + r"} & \multicolumn{2}{c}{"
             + " / ".join(f3(get(lat.get("onepass"), "ms_per_decision", d), 1) for d in DECS) + r"} & & \\",
             r"Option-order invariant & \multicolumn{2}{c}{yes} & \multicolumn{2}{c}{no} & & \\",
             r"Unseen option sets & \multicolumn{2}{c}{yes (Sec.~\ref{sec:custom})} & \multicolumn{2}{c}{no (retrain)} & & \\",
             r"\bottomrule", r"\end{tabular}"]
    write("onepass", "\n".join(rows))


def t_grounded_baselines():
    nj = {s: seeds("nanojev", s) for s in ("test", "heldout", "aggrefact")}
    op = {s: seeds("onepass", s) for s in ("test", "heldout", "aggrefact")}
    sets = [("data_clean", "test", ""), ("data_heldout", "heldout", "_heldout"), ("data_aggrefact", "aggrefact", "_aggrefact")]

    def gfile(slug, ds):
        return J(f"grounded/{slug}_{ds}.json")

    systems = [
        ("CaliDec (K-pass, trunc. 512)", "33\\,M", lambda ds, s, suf: ms(nj[s], "grounded", "calibrated", "accuracy")),
        ("CaliDec (K-pass, chunk-max)", "33\\,M", lambda ds, s, suf: ms([gfile("nanojev-runs_v2_nanojev_seed0", ds)], "grounded", "calibrated", "accuracy")),
        ("1-pass classifier", "33\\,M", lambda ds, s, suf: ms(op[s], "grounded", "calibrated", "accuracy")),
        ("DeBERTa-v3-base NLI", "184\\,M", lambda ds, s, suf: ms([gfile("nli-MoritzLaurer_DeBERTa-v3-base-mnli-fever-anli", ds)], "grounded", "calibrated", "accuracy")),
        ("DeBERTa-v3-large NLI", "435\\,M", lambda ds, s, suf: ms([gfile("nli-MoritzLaurer_DeBERTa-v3-large-mnli-fever-anli-ling-wanli", ds)], "grounded", "calibrated", "accuracy")),
        ("AlignScore-base", "125\\,M", lambda ds, s, suf: ms([gfile("alignscore-ckpts_AlignScore-base", ds)], "grounded", "calibrated", "accuracy")),
        ("MiniCheck-RoBERTa-L", "355\\,M", lambda ds, s, suf: ms([gfile("minicheck-roberta-large", ds)], "grounded", "calibrated", "accuracy")),
        ("MiniCheck-DeBERTa-L", "435\\,M", lambda ds, s, suf: ms([gfile("minicheck-deberta-v3-large", ds)], "grounded", "calibrated", "accuracy")),
        (r"BGE-reranker-v2-m3$^\dagger$", "568\\,M", lambda ds, s, suf: ms([J(f"baselines/reranker{suf}.json")], "grounded", "calibrated", "accuracy")),
        (r"Qwen3-4B-Instruct$^\dagger$", "4\\,B", lambda ds, s, suf: ms([J(f"baselines/llm{suf}.json")], "grounded", "calibrated", "accuracy")),
    ]
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"\textbf{System} & \textbf{Params} & \textbf{MultiNLI (test)} & \textbf{VitaminC} & \textbf{LLM-AggreFact} \\", r"\midrule"]
    for label, params, fn in systems:
        rows.append(f"{label} & {params} & " + " & ".join(pm(*fn(ds, s, suf)) for ds, s, suf in sets) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("grounded_baselines", "\n".join(rows))


def t_custom():
    tasks = [("agnews", "AG News", 4), ("yahoo", "Yahoo Answers", 10), ("emotion", "Emotion", 6)]
    probe = J("custom/probe_onepass.json") or {}
    rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"\textbf{Task} & $K$ & \textbf{Random} & \textbf{v0.1} & \textbf{v1.0} & \textbf{CaliDec (rev.)} & \textbf{Qwen3-4B} \\",
            r"\midrule"]
    for key, label, k in tasks:
        dec = f"custom:{key}"
        rows.append(f"{label} & {k} & {1 / k:.3f} & {f3(acc(J(f'custom/v0.1/{key}.json'), dec))} & "
                    f"{f3(acc(J(f'custom/v1.0/{key}.json'), dec))} & {f3(acc(J(f'custom/v2/{key}.json'), dec))} & "
                    f"{f3(acc(J(f'custom/llm_{key}.json'), dec))}" + r" \\")
    rows += [r"\midrule", r"\multicolumn{7}{l}{\textit{Supervised reference: linear probe on frozen 1-pass encoder, 300 labels}} \\"]
    for key, label, k in tasks:
        rows.append(f"{label} (probe) & {k} & & & & {f3(get(probe, key, 'accuracy'))} & " + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("custom", "\n".join(rows))
    for key, label, k in tasks:
        K = key.capitalize()
        num(f"Cust{K}VZero", f3(acc(J(f"custom/v0.1/{key}.json"), f"custom:{key}")))
        num(f"Cust{K}VOne", f3(acc(J(f"custom/v1.0/{key}.json"), f"custom:{key}")))
        num(f"Cust{K}Rev", f3(acc(J(f"custom/v2/{key}.json"), f"custom:{key}")))
        num(f"Cust{K}Rand", f"{1 / k:.3f}")


def t_truncation():
    rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"\textbf{Decision / source} & $n$ & \textbf{\% $>$512} & \textbf{\% gold cut} & \textbf{Acc trunc.} & \textbf{Acc not trunc.} & \textbf{Acc balanced} \\",
            r"\midrule"]
    for split, bal in (("test", "test_balanced"), ("heldout", "heldout_balanced")):
        t = J(f"truncation/{split}.json") or {}
        b = J(f"nanojev/seed0/{bal}.json")
        for key, e in t.items():
            dec = key.split("/")[0]
            multi = "pct_gold_cut" in e
            cells = [key.replace("_", r"\_"), str(e["n"]), pct(e.get("pct_truncated"), 1),
                     pct(e.get("pct_gold_cut"), 1) if multi else "---",
                     f"{f3(e.get('acc_truncated'))} ({e.get('n_truncated', 0)})",
                     f3(e.get("acc_not_truncated")),
                     f3(acc(b, dec)) if multi else "---"]
            rows.append(" & ".join(cells) + r" \\")
    if len(rows) == 4:
        rows.append(PH + r" & & & & & & \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("truncation", "\n".join(rows))
    t, h = J("truncation/test.json") or {}, J("truncation/heldout.json") or {}
    num("TruncInDistMax", pct(max((e["pct_truncated"] for e in t.values()), default=None), 1) if t else None)
    m = h.get("sufficient/musique", {})
    num("TruncMusSufPct", pct(m.get("pct_truncated"), 1) if m else None)
    num("TruncMusSufGold", pct(m.get("pct_gold_cut"), 1) if m else None)
    num("TruncMusSufAccT", f3(m.get("acc_truncated")))
    num("TruncMusSufAccNT", f3(m.get("acc_not_truncated")))
    num("TruncMusSufAccBal", f3(acc(J("nanojev/seed0/heldout_balanced.json"), "sufficient")))


def t_e2e():
    e = J("e2e/e2e.json") or {}
    rows = [r"\begin{tabular}{llcccccc}", r"\toprule",
            r"\textbf{Set} & \textbf{Filter} & \textbf{EM} & \textbf{F1} & \textbf{Gold rec.} & \textbf{Gold prec.} & \textbf{Kept} & \textbf{ms/q} \\",
            r"\midrule"]
    for s, lab in (("hotpot", "HotpotQA"), ("musique", "MuSiQue")):
        for f, fl in (("none", "none (top-5)"), ("bge", "BGE + LR"), ("nanojev", "CaliDec"), ("qwen", "Qwen3-4B judge")):
            r = e.get(f"{s}/{f}")
            rows.append(f"{lab} & {fl} & {f3(get(r, 'em'))} & {f3(get(r, 'f1'))} & {f3(get(r, 'gold_recall'))} & "
                        f"{f3(get(r, 'gold_precision'))} & {f3(get(r, 'kept'), 2)} & {f3(get(r, 'total_ms'), 0)}" + r" \\")
        if s == "hotpot":
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("e2e", "\n".join(rows))
    for s_, S_ in (("hotpot", "Hot"), ("musique", "Mus")):
        for f_, F_ in (("none", "None"), ("bge", "Bge"), ("nanojev", "NJ"), ("qwen", "Qwen")):
            r = e.get(f"{s_}/{f_}")
            num(f"EE{S_}{F_}Fone", f3(get(r, "f1")))
            num(f"EE{S_}{F_}Ms", f3(get(r, "total_ms"), 0))


def main():
    global R, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v2")
    ap.add_argument("--out", default="paper_revision/tables")
    args = ap.parse_args()
    R, OUT = Path(args.results), Path(args.out)

    t_datasplit(["data_heldout/test.jsonl", "data_aggrefact/test.jsonl"])
    for fn in (t_main, t_calibration, t_cascade_grounded, t_cascade_all, t_ablation_temp, t_detailed,
               t_perclass, t_latency, t_ablation_data, t_onepass, t_grounded_baselines, t_custom,
               t_truncation, t_e2e):
        fn()
    lines = ["% generated by scripts/make_tables.py - do not edit by hand"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(NUM.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = {f.name: f.read_text(encoding="utf-8").count(r"\ph{") for f in sorted(OUT.glob("*.tex"))}
    print("tables written to", OUT)
    for f, n in missing.items():
        print(f"  {f:<26} {'complete' if n == 0 else f'{n} value(s) still missing'}")


if __name__ == "__main__":
    main()
