"""Complete cached external inference and failure audit without model calls."""
from collections import Counter
import numpy as np
from prepare_evidence import OUT, ROOT, ROUND, SEED, read, save, metrics, digest


def cluster_inference(rows):
    """Resample whole queries; recompute equal-facet macro means on each sample."""
    facets = ("background", "method", "result")
    qids = sorted({r["query_id"] for r in rows})
    numer = np.zeros((len(qids), 3))
    counts = np.zeros_like(numer)
    for r in rows:
        i, j = qids.index(r["query_id"]), facets.index(r["facet"])
        numer[i, j] += r["delta"]
        counts[i, j] += 1
    observed = float(np.mean(numer.sum(0) / counts.sum(0)))
    rng = np.random.default_rng(SEED)
    boots = []
    while len(boots) < 10000:
        sample = rng.integers(len(qids), size=(1000, len(qids)))
        den = counts[sample].sum(1)
        num = numer[sample].sum(1)
        valid = (den > 0).all(1)
        boots.extend((num[valid] / den[valid]).mean(1).tolist())
    exceed = 0
    for _ in range(50):
        signs = rng.choice([-1, 1], (1000, len(qids)))
        vals = ((signs @ numer) / counts.sum(0)).mean(1)
        exceed += int((abs(vals) >= abs(observed)-1e-14).sum())
    # Leave-query-out diagnostics show influence without selecting exclusions.
    loo = []
    for i, qid in enumerate(qids):
        den = counts.sum(0) - counts[i]
        loo.append(dict(query_id=qid,
                        delta=float(np.mean((numer.sum(0)-numer[i])/den)) if (den > 0).all() else None))
    return dict(n_queries=len(qids), n_query_facets=len(rows),
                delta=observed, ci95=np.quantile(boots[:10000],[.025,.975]).tolist(),
                p_sign_flip=(exceed+1)/50001, loo=loo,
                estimand="equal-facet macro mean; resample/sign-flip whole query clusters")


def main():
    manifest = read(OUT/"input_manifest.json")
    assert all(digest(ROOT/p) == h for p,h in manifest.items())
    data = read(OUT/"P0_external.json")["20260901"]
    a = data["heuristic_score"]["per_query_facet"]
    b = data["specter_score"]["per_query_facet"]
    results = {}
    for metric in ("ndcg", "ap"):
        rows = [dict(query_id=r["query_id"], facet=r["facet"], delta=r[metric]-b[k][metric])
                for k,r in a.items()]
        result = cluster_inference(rows)
        expected = data["heuristic_score"]["summary"]["macro"][metric]-data["specter_score"]["summary"]["macro"][metric]
        assert abs(result["delta"]-expected) < 1e-12
        results[metric] = result
    # Auxiliary diagnostic family = two metrics, distinct from OpenAlex primary family.
    previous = 0
    for i, key in enumerate(sorted(results, key=lambda k:results[k]["p_sign_flip"])):
        previous = max(previous, min(1, (2-i)*results[key]["p_sign_flip"]))
        results[key]["p_holm_two_external_metrics"] = previous
    save("P0_external_inference.json", results)
    queries = {q["query_paper_id"]:q for q in read(ROOT/"src/core/literature_discovery_benchmark_silver.json")}
    cases = read(ROUND/"results_E9.json")["top_failure_cases"]
    records = []
    for case in cases:
        qid = case["query_id"]
        gold = {c["candidate_paper_id"]:c["silver_grade"] for c in queries[qid]["candidates"]}
        systems = {k:metrics(v["selected_ids"],gold) for k,v in case["systems"].items()}
        reasons = dict(Counter(x["reason"] for x in case["omitted_positive_candidates"]))
        empty = [k for k,v in case["selected_by_role"].items() if not v]
        records.append(dict(query_id=qid,title=case["query_title"],omission_reasons=reasons,
                            empty_roles=empty,dedup_events=case["dedup_events"],
                            omitted_positive_candidates=case["omitted_positive_candidates"],
                            corrected_metrics=systems))
    save("P0_failure_audit.json",dict(
        selection="All ten historical selected failure traces; not a prevalence sample",
        caveat="Dedup events show eligibility overlap, not proof each event caused an omission",
        cases=records))
    grid = read(OUT/"P0_grid.json")["configs"]
    sweep = {}
    for k in (2,3,4):
        for strategy in sorted({r["strategy"] for r in grid}):
            rows = [r for r in grid if r["k"] == k and r["strategy"] == strategy]
            assert len(rows)==24
            sweep[f"k{k}:{strategy}"] = {m:dict(min=min(r[m] for r in rows),
                max=max(r[m] for r in rows),mean=float(np.mean([r[m] for r in rows])))
                for m in ("ndcg","grade","recall","returned","nonempty_roles")}
    save("P0_order_ranges.json",sweep)
    report = ["# 外部诊断与失败机制审计\n",
              "## 外部静态 scorer\n",
              "比较为文本适配启发式减 SPECTER；同 query 的 facet 一起重采样，统计保持三个 facet 等权。\n",
              "|指标|宏平均差|query-cluster 95% CI|两指标 Holm p|",
              "|---|---:|---|---:|"]
    for m,r in results.items():
        report.append(f"|{m}|{r['delta']:+.4f}|[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]|{r['p_holm_two_external_metrics']:.5f}|")
    report += ["\n启发式的外部表现为负面诊断。不能用另外的 SciBERT 正面结果替它背书。\n",
               "## 既有十个失败案例（全量保留）\n",
               "下表不估计总体失败比例。空组和跨角色 eligibility 不等于人工角色错误。\n",
               "|Query|漏选原因计数|空角色数|去重事件数|fixed recall|construction recall|",
               "|---|---|---:|---:|---:|---:|"]
    for r in records:
        m=r["corrected_metrics"]
        report.append(f"|{r['query_id']}|{r['omission_reasons']}|{len(r['empty_roles'])}|{len(r['dedup_events'])}|{m['B0_fixed']['recall']}|{m['SPECTER2_flat']['recall']}|")
    report += ["\n### 两个可追溯示例\n",
               "- Query 82890：grade=2 的 18111 在 closely-related 排第五，固定每角色三篇时漏选；比例配额可保留它，但该例非空角色从二变一。说明覆盖与角色分布的权衡，而非无损预算改善。",
               "- Query 15688：grade=3 的 35047 在 closely-related 排第四，固定配额漏选；恢复它使 recall 从 0.8 到 1，但比例配置非空角色从三变二。",
               "\n## 顺序扫描\n",
               "|配置|Pool NDCG 范围|grade 范围|recall 范围|",
               "|---|---|---|---|"]
    for k,r in sweep.items():
        report.append("|"+k+"|"+"|".join(f"{r[m]['min']:.4f}–{r[m]['max']:.4f}" for m in ("ndcg","grade","recall"))+"|")
    report.append("\n所有扫描结果为敏感性分析，不采用测试集最佳顺序。不声称总预算十二在最多八候选数据上受约束。")
    (OUT/"P0_EXTERNAL_FAILURE_REPORT.md").write_text("\n".join(report)+"\n")
    assert all(digest(ROOT/p) == h for p,h in manifest.items())
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
