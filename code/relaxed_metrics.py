"""All presentation metrics under the relaxed (fence-stripped) parser, for the ISPA camera-ready.

Reuses collect/relax/infer/holm from reparse.py unchanged. Adds: per-cell means of every metric,
per-model contrasts with Holm over the two registered tests per model (as in the paper), and
marginal 95% bootstrap CIs of per-cell coverage means for the figure (seed 20260929, 20,000
resamples, as plot_evidence.py).

Usage: python -I relaxed_metrics.py ATTEMPTS_JSONL P1_INPUTS_JSON BENCHMARK_JSON OUT_JSON
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reparse import collect, relax, infer, holm  # noqa: E402

METRICS = ("input_coverage", "silver_positive_coverage", "redundancy", "use_vocabulary_coverage",
           "outside_reference_fraction")
MODELS = ["claude-sonnet-4-6", "gpt-5.5"]


def ci(values):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260929)
    means = v[rng.integers(len(v), size=(20000, len(v)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def main():
    att_p, inp_p, bench_p, out_p = sys.argv[1:5]
    attempts = [json.loads(s) for s in open(att_p).read().splitlines()]
    inputs = json.load(open(inp_p))
    bench = json.load(open(bench_p))
    golds = {q["query_paper_id"]: {c["candidate_paper_id"]: c["silver_grade"] for c in q["candidates"]}
             for q in bench}
    out = {}
    for name, parser in (("strict", lambda t: t), ("relaxed", relax)):
        rows = collect(inputs, MODELS, attempts, golds, parser)
        cells = defaultdict(list)
        for r in rows:
            cells[(r["model"], r["condition"])].append(r)
        summary = {}
        for (model, cond), items in sorted(cells.items()):
            valid = [r for r in items if r["valid"]]
            s = dict(valid=len(valid), planned=len(items))
            for m in METRICS:
                vals = [r[m] for r in valid if r.get(m) is not None]
                s[m] = float(np.mean(vals)) if vals else None
                s[m + "_n"] = len(vals)
                if m == "input_coverage" and vals:
                    s["input_coverage_ci95"] = ci(vals)
            summary[f"{model}:{cond}"] = s
        lookup = {(r["query_id"], r["model"], r["condition"]): r for r in rows}
        qids = sorted({r["query_id"] for r in rows})
        tests = {}
        for model in MODELS:
            fam = {}
            for left, right in (("G", "F"), ("R", "G"), ("R", "W")):
                diffs = [lookup[q, model, left]["input_coverage"] - lookup[q, model, right]["input_coverage"]
                         for q in qids if lookup[q, model, left]["valid"] and lookup[q, model, right]["valid"]]
                if diffs:
                    fam[f"{left}-{right}"] = infer(diffs)
            primary = {k: v for k, v in fam.items() if k != "R-W"}
            if primary:
                holm(primary)
            tests[model] = fam
        out[name] = dict(summary=summary, tests=tests)
    Path(out_p).write_text(json.dumps(out, indent=2) + "\n")
    for name in out:
        print("==", name)
        for k, s in out[name]["summary"].items():
            print(k, s["valid"], *(f"{m[:6]}={s[m]:.3f}" for m in METRICS if s[m] is not None))
        for model, fam in out[name]["tests"].items():
            for k, t in fam.items():
                print(model, k, f"{t['delta']:+.4f}", [round(x, 3) for x in t["ci95"]], t["n"],
                      f"p={t['p']:.4f}", f"holm={t.get('p_holm', float('nan')):.4f}")


if __name__ == "__main__":
    main()
