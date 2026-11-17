"""Read-only historical analysis; writes only next to this script. Requires numpy."""
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EXP = ROOT / "docs/papers/cikm_reading_packs/experiments"
ROUND = EXP / "reviewer_round3"
SEED = 20260915


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dcg(grades, k=8, linear=False):
    if linear:
        return sum(g / (1 if i == 0 else math.log2(i + 1))
                   for i, g in enumerate(grades[:k]))
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def metrics(ids, gold):
    assert len(ids) == len(set(ids)) and set(ids) <= set(gold)
    selected = ids[:8]
    grades = [gold[i] for i in selected]
    ideal = dcg(sorted(gold.values(), reverse=True))
    positive = sum(g >= 2 for g in gold.values())
    gain = sum(2**g - 1 for g in gold.values())
    recall = sum(g >= 2 for g in grades) / positive if positive else None
    return dict(ndcg=dcg(grades) / ideal if ideal else 0.,
                grade=sum(grades) / len(grades) if grades else 0.,
                recall=recall, recall_legacy_zero=recall or 0.,
                returned=len(selected), empty=int(not selected),
                gain_retention=sum(2**g-1 for g in grades)/gain if gain else None)


def average(rows):
    return {k: float(np.mean(v)) if (v := [r[k] for r in rows if r[k] is not None]) else None
            for k in rows[0]}


def infer(diffs):
    a = np.asarray(diffs, dtype=float)
    assert len(a)
    rng = np.random.default_rng(SEED)
    boots = np.mean(rng.choice(a, (10000, len(a))), axis=1)
    obs = abs(float(a.mean()))
    exceed = 0
    for _ in range(50):
        vals = np.abs((rng.choice([-1, 1], (1000, len(a))) * a).mean(axis=1))
        exceed += int(np.sum(vals >= obs - 1e-14))
    return dict(n=len(a), delta=float(a.mean()), ci95=np.quantile(boots, [.025, .975]).tolist(),
                p=(exceed+1)/50001, wins=int(sum(a > 1e-12)),
                ties=int(sum(abs(a) <= 1e-12)), losses=int(sum(a < -1e-12)))


def holm(tests):
    order = sorted(tests, key=lambda k: tests[k]["p"])
    prev = 0.
    for i, key in enumerate(order):
        prev = max(prev, min(1., tests[key]["p"] * (len(order)-i)))
        tests[key]["p_holm"] = prev


def main():
    sources = [ROOT / "src/core/literature_discovery_benchmark_silver.json",
               ROOT / "src/core/generate_literature_discovery_benchmark_seed.py",
               ROOT / "src/core/evaluate_query_discovery_pack_silver.py",
               EXP / "run_E1_csfcube.py", EXP / "run_E3_task_grounded.py",
               EXP / "run_E4_adaptive_budgets.py", EXP / "run_E2C_pair_baseline.py",
               ROUND / "results_E8.json", ROUND / "results_E9.json",
               ROUND / "raw/E8_queries.jsonl", ROUND / "raw/E9_grid_queries.jsonl",
               ROUND / "raw/E8_candidate_oof.jsonl"]
    for seed in (20260901, 20260902):
        sources += [ROUND/f"raw/E10_candidate_scores_seed{seed}.jsonl",
                    ROUND/f"results_E10_seed{seed}.json"]
    for facet in ("background", "method", "result"):
        sources.append(ROOT/f"external_benchmarks/csfcube_data/test-pid2anns-csfcube-{facet}.json")
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if (OUT/"input_manifest.json").exists():
        assert read(OUT/"input_manifest.json") == manifest, "Frozen input changed"
    else:
        save("input_manifest.json", manifest)
    benchmark = read(sources[0])
    queries = {q["query_paper_id"]: q for q in benchmark}
    golds = {qid: {c["candidate_paper_id"]: c["silver_grade"] for c in q["candidates"]}
             for qid, q in queries.items()}
    # Hand-calculated invariants catch selected-only normalization and empty output errors.
    assert metrics([2], {1: 3, 2: 1})["ndcg"] < .15
    assert metrics([1, 2], {1: 3, 2: 1})["ndcg"] == 1.
    assert metrics([], {1: 3})["ndcg"] == 0.
    assert metrics([1], {1: 0})["recall"] is None
    e8 = lines(ROUND/"raw/E8_queries.jsonl")
    assert len(e8) == len(queries) == len({x["query_id"] for x in e8})
    allrows = defaultdict(dict)
    historical_errors = []
    for row in e8:
        qid = row["query_id"]
        gold = golds[qid]
        order = [c["candidate_paper_id"] for c in sorted(queries[qid]["candidates"],
                  key=lambda c: (c["candidate_rank"], c["candidate_paper_id"]))]
        systems = dict(row["systems"])
        for family, content in row.get("learned", {}).items():
            for name, v in content["systems"].items():
                systems[family+"_"+name] = v
        for name, v in systems.items():
            m = metrics(v["selected_ids"], gold)
            allrows[name][qid] = m
            if "metrics" in v:
                historical_errors.extend([abs(m["grade"]-v["metrics"]["avg_grade@8"]),
                                          abs(m["recall_legacy_zero"]-v["metrics"]["positive_recall@8"])])
        chosen = systems["rolecond_full"]["selected_ids"][:8]
        for name, ids in {"construction_top_m": order[:len(chosen)],
                          "rolecond_original_order": sorted(chosen, key=order.index)}.items():
            allrows[name][qid] = metrics(ids, gold)
        assert allrows["construction_top_m"][qid]["returned"] == len(chosen)
        for k in ("grade", "recall", "gain_retention", "returned"):
            assert allrows["rolecond_full"][qid][k] == allrows["rolecond_original_order"][qid][k]
    assert max(historical_errors) < 1e-10
    summary = {name: average(list(rows.values())) for name, rows in allrows.items()}
    tests = {}
    for base in ("construction_top_m", "rolecond_original_order", "specter_flat"):
        for metric in ("ndcg", "grade", "recall"):
            diffs = [allrows["rolecond_full"][q][metric]-allrows[base][q][metric]
                     for q in sorted(queries) if allrows[base][q][metric] is not None]
            tests[base+":"+metric] = infer(diffs)
    holm(tests)
    save("P0_selection.json", dict(summary=summary, primary_tests=tests, per_query=allrows))
    grid = lines(ROUND/"raw/E9_grid_queries.jsonl")
    configs = defaultdict(list)
    gridout = []
    seen = set()
    for row in grid:
        key = (row["k"], row["order_index"], row["strategy"])
        assert (*key, row["query_id"]) not in seen
        seen.add((*key, row["query_id"]))
        m = metrics(row["selected_ids"], golds[row["query_id"]])
        m["nonempty_roles"] = sum(v > 0 for v in row["fill"].values())
        configs[key].append(m)
        gridout.append(dict(k=row["k"], order_index=row["order_index"], strategy=row["strategy"],
                            query_id=row["query_id"], metrics=m))
    assert len(configs) == 216 and all(len(v) == 90 for v in configs.values())
    save("P0_grid.json", dict(configs=[dict(k=k[0], order_index=k[1], strategy=k[2],
                                         **average(v)) for k, v in configs.items()],
                              per_query=gridout))
    external = {}
    for seed in (20260901, 20260902):
        raw = lines(ROUND/f"raw/E10_candidate_scores_seed{seed}.jsonl")
        groups = defaultdict(list)
        for r in raw:
            groups[(r["query_pid"], r["facet"])].append(r)
        assert len(raw) == 6244 and len(groups) == 50
        results = defaultdict(dict)
        for (qid, facet), rows in groups.items():
            official = read(ROOT/f"external_benchmarks/csfcube_data/test-pid2anns-csfcube-{facet}.json")[qid]
            gold = dict(zip(map(str, official["cands"]), official["relevance_adju"]))
            assert {r["candidate_pid"]: r["grade"] for r in rows} == gold
            assert len(rows) == len(gold)
            for col in ("specter_score", "heuristic_score", "scibert_score", "fusion_score", "logreg_score"):
                ranked = sorted(rows, key=lambda r: (-r[col], r["candidate_pid"]))
                grades = [r["grade"] for r in ranked]
                ideal = dcg(sorted(gold.values(), reverse=True), 10, True)
                hits = 0
                aps = 0.
                for i, g in enumerate(grades[:10], 1):
                    if g > 0:
                        hits += 1
                        aps += hits/i
                denom = min(sum(g > 0 for g in gold.values()), 10)
                results[col][qid+":"+facet] = dict(ndcg=dcg(grades,10,True)/ideal if ideal else 0.,
                                                   ap=aps/denom if denom else 0., facet=facet, query_id=qid)
        report = {}
        mapping = dict(specter_score="specter_flat", heuristic_score="heuristic_role",
                       scibert_score="scibert_pair", fusion_score="late_fusion", logreg_score="specter_logreg")
        historical = read(ROUND/f"results_E10_seed{seed}.json")["systems"]
        for col, vals in results.items():
            means = {f: {m: float(np.mean([r[m] for r in vals.values() if r["facet"] == f]))
                         for m in ("ndcg", "ap")} for f in ("background", "method", "result")}
            means["macro"] = {m: float(np.mean([v[m] for v in means.values()])) for m in ("ndcg","ap")}
            assert abs(means["macro"]["ndcg"]-historical[mapping[col]]["macro"]["ndcg@10"]) < 1e-6
            assert abs(means["macro"]["ap"]-historical[mapping[col]]["macro"]["map@10"]) < 1e-6
            report[col] = dict(summary=means, per_query_facet=vals)
        external[str(seed)] = report
    assert external["20260901"]["heuristic_score"] == external["20260902"]["heuristic_score"]
    save("P0_external.json", external)
    # Render all P1 inputs before any model invocation.
    bydiscipline = defaultdict(list)
    for q in benchmark:
        bydiscipline[q["query_discipline_l1"]].append(q["query_paper_id"])
    rng = random.Random(SEED)
    sampled = [qid for d in sorted(bydiscipline)
               for qid in sorted(rng.sample(sorted(bydiscipline[d]), 4))]
    role_keys = ["closely_related", "methodological_neighbors", "background_foundations", "cross_field_bridges"]
    titles = ["Closely related", "Methods", "Background", "Cross-field bridges"]
    derangements = [p for p in itertools.permutations(range(4)) if all(i != p[i] for i in range(4))]
    lookup = {r["query_id"]: r for r in e8}
    inputs = []
    for qid in sampled:
        q = queries[qid]
        v = lookup[qid]["systems"]["rolecond_full"]
        ids = v["selected_ids"]
        assert sum(v["fill"].values()) == len(ids)
        cmap = {c["candidate_paper_id"]: c for c in q["candidates"]}
        groups = []
        offset = 0
        for role in role_keys:
            groups.append(ids[offset:offset+v["fill"].get(role, 0)])
            offset += v["fill"].get(role, 0)
        permutation = rng.choice(derangements)
        for condition in ("F","G","R","W"):
            rendered = []
            for i, members in enumerate(groups):
                if not members:
                    continue
                if condition != "F":
                    label = (f"Group {chr(65+i)}" if condition == "G"
                             else titles[permutation[i] if condition == "W" else i])
                    rendered.append("## "+label)
                rendered.extend(f"[{cid}] {cmap[cid]['candidate_title']}" for cid in members)
            query_text = q["query_title"]+"\n"+q.get("query_summary_snippet","")[:500]
            prompt = ("You are writing the Related Work section outline for a research idea.\n\n"
                      f"Research idea: {query_text}\n\nYou are given the following retrieved papers:\n"
                      +"\n".join(rendered)+"\n\nProduce an outline with 3-5 thematic subsections. "
                      "Under each subsection, list the papers (by ID number) you would cite and one sentence on how each paper is used.\n"
                      'Only use the given papers. Output valid JSON only, no markdown:\n'
                      '{"sections":[{"theme":"...","papers":[{"id":NUMBER,"use":"..."}]}]}')
            inputs.append(dict(query_id=qid, discipline=q["query_discipline_l1"], condition=condition,
                               input_ids=ids, groups=groups, wrong_label_permutation=permutation,
                               prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    assert len(inputs) == 144 and len(set(sampled)) == 36
    save("P1_inputs.json", inputs)
    save("P1_sample.json", dict(seed=SEED, query_ids=sampled,
                                discipline_counts=dict(Counter(queries[q]["query_discipline_l1"] for q in sampled)),
                                models_locked=False, model_calls_completed=0))
    assert {str(p.relative_to(ROOT)): digest(p) for p in sources} == manifest
    save("verification.json", dict(input_hashes_unchanged=True, historical_grade_recall_max_error=max(historical_errors),
                                   e9_configs=len(configs), e9_rows=len(grid), external_full_pool_verified=True,
                                   external_historical_metrics_reproduced=True, p1_prompts=len(inputs),
                                   benchmark_pool_sizes=dict(Counter(len(q["candidates"]) for q in benchmark)),
                                   zero_positive_queries=[q for q,g in golds.items() if not any(v>=2 for v in g.values())]))
    report = ["# P0 初步结果\n", "基线是候选构造顺序，不是完整池 SPECTER2 排序。recall 排除零正例查询。\n",
              "|系统|Pool NDCG@8|返回平均 grade|Recall@8|返回数量|",
              "|---|---:|---:|---:|---:|"]
    for name, r in summary.items():
        report.append(f"|{name}|{r['ndcg']:.4f}|{r['grade']:.4f}|{r['recall']:.4f}|{r['returned']:.2f}|")
    report += ["\n## 主要配对检验\n", "|比较（RoleCond 减基线）|指标|差值|95% CI|Holm p|",
               "|---|---|---:|---|---:|"]
    for name, r in tests.items():
        base,m = name.split(":")
        report.append(f"|{base}|{m}|{r['delta']:+.4f}|[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]|{r['p_holm']:.5f}|")
    report.append("\n本脚本只生成离线证据和 P1 输入，不执行 API。外部聚类统计与失败案例见 P0_EXTERNAL_FAILURE_REPORT.md；主张边界见 WRITING_EVIDENCE.md；正式 P1 状态见 PREWRITING_STATUS.md。\n")
    (OUT/"P0_REPORT.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
