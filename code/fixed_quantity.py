"""Fixed-quantity construction-order baselines (Table II fixed rows, Fig. 3).

The top-m control in the paper takes m from RoleCond's own return count per query, which a
deployed system cannot know. Here every query gets the same k (construction order, capped by
pool size), k = 1..8, compared with RoleCond on the same 90-query benchmark.

Read-only on inputs. Metrics and inference are copied from original/prepare_evidence.py (same
seed, same resampling counts), so the RoleCond and top-m rows reproduce evidence/P0_selection.json.

Usage: python fixed_quantity.py data/benchmark.json data/rolecond_outputs.jsonl OUT_JSON
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 20260915


def dcg(grades, k=8):
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def metrics(ids, gold):
    assert len(ids) == len(set(ids)) and set(ids) <= set(gold)
    selected = ids[:8]
    grades = [gold[i] for i in selected]
    ideal = dcg(sorted(gold.values(), reverse=True))
    positive = sum(g >= 2 for g in gold.values())
    return dict(ndcg=dcg(grades) / ideal if ideal else 0.,
                grade=sum(grades) / len(grades) if grades else 0.,
                recall=sum(g >= 2 for g in grades) / positive if positive else None,
                returned=len(selected))


def average(rows):
    return {k: float(np.mean(v)) if (v := [r[k] for r in rows if r[k] is not None]) else None
            for k in rows[0]}


def infer(diffs):
    a = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(SEED)
    boots = np.mean(rng.choice(a, (10000, len(a))), axis=1)
    obs = abs(float(a.mean()))
    exceed = 0
    for _ in range(50):
        vals = np.abs((rng.choice([-1, 1], (1000, len(a))) * a).mean(axis=1))
        exceed += int(np.sum(vals >= obs - 1e-14))
    return dict(n=len(a), delta=float(a.mean()), ci95=np.quantile(boots, [.025, .975]).tolist(),
                p=(exceed + 1) / 50001)


def holm(tests):
    prev = 0.
    order = sorted(tests, key=lambda k: tests[k]["p"])
    for i, key in enumerate(order):
        prev = max(prev, min(1., tests[key]["p"] * (len(order) - i)))
        tests[key]["p_holm"] = prev


def main():
    bench_p, e8_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    bench = json.loads(bench_p.read_text())
    queries = {q["query_paper_id"]: q for q in bench}
    golds = {qid: {c["candidate_paper_id"]: c["silver_grade"] for c in q["candidates"]}
             for qid, q in queries.items()}
    e8 = [json.loads(s) for s in e8_p.read_text().splitlines() if s.strip()]
    assert len(e8) == len(queries) == 90
    rows = {}
    counts = Counter()
    for row in e8:
        qid = row["query_id"]
        gold = golds[qid]
        order = [c["candidate_paper_id"] for c in sorted(queries[qid]["candidates"],
                 key=lambda c: (c["candidate_rank"], c["candidate_paper_id"]))]
        chosen = row["systems"]["rolecond_full"]["selected_ids"][:8]
        counts[len(chosen)] += 1
        r = {"rolecond": metrics(chosen, gold), "top_m": metrics(order[:len(chosen)], gold)}
        for k in range(1, 9):
            r[f"top_{k}"] = metrics(order[:k], gold)
        rows[qid] = r
    qids = sorted(rows)
    summary = {name: average([rows[q][name] for q in qids]) for name in rows[qids[0]]}
    tests = {}
    for k in (5, 6, 7):
        fam = {}
        for metric in ("ndcg", "grade", "recall"):
            diffs = [rows[q]["rolecond"][metric] - rows[q][f"top_{k}"][metric]
                     for q in qids if rows[q][f"top_{k}"][metric] is not None]
            fam[metric] = infer(diffs)
        holm(fam)
        tests[f"rolecond_minus_top_{k}"] = fam
    out.write_text(json.dumps(dict(rolecond_return_count_hist=dict(sorted(counts.items())),
                                   summary=summary, tests=tests), indent=2) + "\n")
    for name, s in summary.items():
        print(f"{name:10s} ndcg={s['ndcg']:.3f} grade={s['grade']:.3f} "
              f"recall={s['recall']:.3f} ret={s['returned']:.2f}")
    for name, fam in tests.items():
        for metric, t in fam.items():
            print(name, metric, f"{t['delta']:+.3f}", [round(x, 3) for x in t["ci95"]],
                  f"p={t['p']:.4f} holm={t['p_holm']:.4f}")
    print("RoleCond return-count histogram:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
