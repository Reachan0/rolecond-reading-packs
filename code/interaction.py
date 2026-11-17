#!/usr/bin/env python3
"""Model x condition interaction for the presentation experiment.

Relaxed (fence-stripped) parse as in reparse.py; for each contrast, the per-query difference of paired
differences gpt-5.5 minus claude-sonnet-4-6 on queries complete for both models; inference = reparse.infer
(same bootstrap / sign-flip procedure and seed). Usage: interaction.py ATTEMPTS INPUTS BENCH OUT
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reparse as R

att_p, inp_p, bench_p, out_p = sys.argv[1:5]
attempts = [json.loads(s) for s in open(att_p).read().splitlines()]
inputs = json.load(open(inp_p))
bench = json.load(open(bench_p))
golds = {q["query_paper_id"]: {c["candidate_paper_id"]: c["silver_grade"] for c in q["candidates"]} for q in bench}
models = ["claude-sonnet-4-6", "gpt-5.5"]
rows = R.collect(inputs, models, attempts, golds, R.relax)
lookup = {(r["query_id"], r["model"], r["condition"]): r for r in rows}
qids = sorted({r["query_id"] for r in rows})
out = {}
for left, right in (("G", "F"), ("R", "G"), ("R", "W")):
    d = {}
    for m in models:
        d[m] = {}
        for q in qids:
            a, b = lookup[q, m, left], lookup[q, m, right]
            if a["valid"] and b["valid"] and a.get("input_coverage") is not None and b.get("input_coverage") is not None:
                d[m][q] = a["input_coverage"] - b["input_coverage"]
    common = sorted(set(d[models[0]]) & set(d[models[1]]))
    inter = [d["gpt-5.5"][q] - d["claude-sonnet-4-6"][q] for q in common]
    out[left + "-" + right] = dict(**R.infer(inter), n_common=len(common),
                                   gpt_mean_common=sum(d["gpt-5.5"][q] for q in common) / len(common),
                                   claude_mean_common=sum(d["claude-sonnet-4-6"][q] for q in common) / len(common))
json.dump(out, open(out_p, "w"), indent=1)
print(json.dumps(out, indent=1))
