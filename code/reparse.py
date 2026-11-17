"""Strict vs relaxed (markdown-fence-stripped) re-analysis of the archived presentation experiment.

Usage: python -I reparse.py ATTEMPTS_JSONL P1_INPUTS_JSON BENCHMARK_JSON ORIGINAL_P1_JSON OUT_JSON

Copied faithfully from the original pipeline (code/original/):
  validate_response / ROLE_USE_KEYWORDS : presentation_runner.py:15-60
  collect / summarize                   : analyze_presentation.py:12-100 (strict path unchanged)
  infer / holm                          : prepare_evidence.py:63-83 (SEED = 20260915, line 16)
Only additions: relax() fence stripping, and the failure-state classification.
"""
import json, re, sys
from collections import Counter, defaultdict
import numpy as np

SEED = 20260915
ROLE_USE_KEYWORDS = {
    "position": {"position", "compare", "contrast", "similar", "related", "overlap", "competitor", "rival", "nearby", "neighbor", "prior"},
    "compare_method": {"method", "technique", "approach", "algorithm", "framework", "pipeline", "baseline", "benchmark", "implementation", "architecture"},
    "ground": {"background", "foundation", "survey", "review", "introduction", "overview", "seminal", "fundamental", "grounding", "basis", "context"},
    "transfer": {"transfer", "bridge", "cross", "interdisciplinary", "borrow", "adapt", "inspire", "adjacent", "novel application", "different field"},
}


def validate_response(text, input_ids, positive_ids):  # presentation_runner.py:33
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return dict(valid=False, failure="invalid_json")
    if not isinstance(obj, dict) or not isinstance(obj.get("sections"), list) or not obj["sections"]:
        return dict(valid=False, failure="invalid_sections")
    refs = []
    for section in obj["sections"]:
        if not isinstance(section, dict) or not isinstance(section.get("theme"), str) or not isinstance(section.get("papers"), list):
            return dict(valid=False, failure="invalid_section_schema")
        for item in section["papers"]:
            if (not isinstance(item, dict) or type(item.get("id")) is not int
                    or not isinstance(item.get("use"), str)):
                return dict(valid=False, failure="invalid_citation_schema")
            refs.append(item["id"])
    valid_ids = set(input_ids); positives = set(positive_ids); cited = set(refs)
    role_hits = set()
    for section in obj["sections"]:
        for item in section["papers"]:
            if item["id"] in valid_ids:
                t = item["use"].lower()
                role_hits.update(role for role, words in ROLE_USE_KEYWORDS.items() if any(w in t for w in words))
    return dict(valid=True, section_count=len(obj["sections"]),
                use_vocabulary_coverage=len(role_hits)/4, use_vocabulary_hits=sorted(role_hits),
                input_coverage=len(cited & valid_ids)/len(valid_ids) if valid_ids else None,
                silver_positive_coverage=len(cited & positives)/len(positives) if positives else None,
                outside_reference_fraction=sum(x not in valid_ids for x in refs)/len(refs) if refs else 0.,
                redundancy=1-len(cited)/len(refs) if refs else 0.,
                citation_count=len(refs), unique_valid_citations=len(cited & valid_ids))


def infer(diffs):  # prepare_evidence.py:63
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


def holm(tests):  # prepare_evidence.py:78
    order = sorted(tests, key=lambda k: tests[k]["p"])
    prev = 0.
    for i, key in enumerate(order):
        prev = max(prev, min(1., tests[key]["p"] * (len(order)-i)))
        tests[key]["p_holm"] = prev


FENCE_OPEN = re.compile(r"^```[ \t]*(?:json)?[ \t]*\n", re.I)
FENCE_CLOSE = re.compile(r"\n[ \t]*```$")


def relax(text):
    """PROTOCOL.md rule: strip whitespace; remove exactly one leading and one trailing code fence."""
    t = text.strip()
    if FENCE_OPEN.match(t) and FENCE_CLOSE.search(t):
        t = FENCE_CLOSE.sub("", FENCE_OPEN.sub("", t, count=1), count=1).strip()
    return t


def relax_parses(text):
    try:
        json.loads(relax(text)); return True
    except ValueError:
        return False


def classify(text, finish):
    """Failure state of a strictly invalid response (descriptive only)."""
    t = text.strip()
    if not t:
        return "empty"
    if t.startswith("```"):
        return "fenced_parsable" if relax_parses(text) else "fenced_unparsable_" + ("truncated" if finish == "length" else "stop")
    return "other_non_json"


def collect(inputs, models, attempts, golds, parser):  # analyze_presentation.py:12
    expected = {f"{r['query_id']}:{r['condition']}:{m}": (r, m) for r in inputs for m in models}
    grouped = defaultdict(list)
    for a in attempts:
        if a["job"] not in expected:
            raise ValueError("Unexpected job")
        row, model = expected[a["job"]]
        if (a["query_id"], a["condition"], a["requested_model"]) != (row["query_id"], row["condition"], model):
            raise ValueError("Journal identity mismatch")
        grouped[a["job"]].append(a)
    out = []
    for job, (row, model) in expected.items():
        logs = grouped[job]
        succ = [a for a in logs if a["status"] == "response"]
        if len(logs) > 3 or len(succ) > 1:
            raise ValueError("More than three attempts or duplicate response")
        if succ and logs[-1] is not succ[0]:
            raise ValueError("Response followed by retry")
        res = dict(job=job, query_id=row["query_id"], condition=row["condition"], model=model,
                   attempts=len(logs), state="missing", valid=False)
        if succ:
            payload = succ[0]["response"]
            res["state"] = "invalid_response"
            text = payload["choices"][0]["message"]["content"]
            finish = payload["choices"][0].get("finish_reason")
            positive = [i for i in row["input_ids"] if golds[row["query_id"]][i] >= 2]
            res.update(validate_response(parser(text), row["input_ids"], positive))
            res["finish_reason"] = finish
            res["state"] = "valid" if res["valid"] else "invalid_response"
            if not res["valid"]:
                res["failure_class"] = classify(text, finish)
        elif logs:
            res["state"] = "exhausted_transport_failure" if len(logs) == 3 else "incomplete_transport_failure"
        out.append(res)
    return out


def summarize(rows, models):  # analyze_presentation.py:60 (logic unchanged; only input_coverage summary kept)
    groups = defaultdict(list)
    for r in rows:
        groups[r["model"] + ":" + r["condition"]].append(r)
    summaries = {}
    for key, items in groups.items():
        valid = [r for r in items if r["valid"]]
        summaries[key] = dict(planned=len(items), valid=len(valid))
        vals = [r["input_coverage"] for r in valid if r.get("input_coverage") is not None]
        summaries[key]["input_coverage"] = sum(vals)/len(vals) if vals else None
        summaries[key]["input_coverage_n"] = len(vals)
    lookup = {(r["query_id"], r["model"], r["condition"]): r for r in rows}
    qids = sorted({r["query_id"] for r in rows})
    tests, aux = {}, {}
    for left, right in (("G", "F"), ("R", "G"), ("R", "W")):
        target = aux if right == "W" else tests
        permodel = {}
        for model in models:
            diffs = {}
            for q in qids:
                a, b = lookup[q, model, left], lookup[q, model, right]
                if a["valid"] and b["valid"] and a.get("input_coverage") is not None and b.get("input_coverage") is not None:
                    diffs[q] = a["input_coverage"] - b["input_coverage"]
            permodel[model] = diffs
            if diffs:
                target[model + ":" + left + "-" + right] = dict(**infer(list(diffs.values())),
                                                              complete_queries=sorted(diffs), planned_queries=len(qids))
        common = set(qids)
        for v in permodel.values():
            common &= set(v)
        if common:
            target["both_models:" + left + "-" + right] = dict(
                **infer([sum(permodel[m][q] for m in models)/len(models) for q in sorted(common)]),
                complete_queries=sorted(common), planned_queries=len(qids))
    if tests:
        holm(tests)
    return dict(summary=summaries, primary_tests=tests, auxiliary_tests=aux,
                valid_jobs=sum(r["valid"] for r in rows))


def main():
    att_p, inp_p, bench_p, orig_p, out_p = sys.argv[1:6]
    attempts = [json.loads(s) for s in open(att_p).read().splitlines()]
    inputs = json.load(open(inp_p))
    bench = json.load(open(bench_p))
    golds = {q["query_paper_id"]: {c["candidate_paper_id"]: c["silver_grade"] for c in q["candidates"]} for q in bench}
    models = ["claude-sonnet-4-6", "gpt-5.5"]  # config order in RUN_LOCK.json
    orig = json.load(open(orig_p))
    report = {}
    for name, parser in (("strict", lambda t: t), ("relaxed", relax)):
        rows = collect(inputs, models, attempts, golds, parser)
        rep = summarize(rows, models)
        rep["failure_classes"] = {m: dict(Counter(r.get("failure_class") for r in rows if r["model"] == m and not r["valid"]))
                                  for m in models}
        rep["failure_classes_by_finish"] = {m: dict(Counter(f"{r.get('failure_class')}|{r.get('finish_reason')}"
                                  for r in rows if r["model"] == m and not r["valid"])) for m in models}
        rep["invalid_by_condition"] = {k: v["planned"] - v["valid"] for k, v in rep["summary"].items()}
        report[name] = rep
    s = report["strict"]
    chk = dict(valid_jobs=(s["valid_jobs"], orig["valid_jobs"]),
               valid_per_cell={k: (v["valid"], orig["summary"][k]["valid"]) for k, v in s["summary"].items()},
               gpt_RG_delta=(s["primary_tests"]["gpt-5.5:R-G"]["delta"], orig["primary_tests"]["gpt-5.5:R-G"]["delta"]))
    ok = (s["valid_jobs"] == orig["valid_jobs"]
          and all(a == b for a, b in chk["valid_per_cell"].values())
          and chk["gpt_RG_delta"][0] == chk["gpt_RG_delta"][1])
    for k in ("primary_tests", "auxiliary_tests"):
        for key, v in orig[k].items():
            for f in ("n", "delta", "p", "ci95", "wins", "ties", "losses", "p_holm"):
                if f in v and s[k][key].get(f) != v[f]:
                    ok = False; chk.setdefault("mismatch", []).append((k, key, f, s[k][key].get(f), v[f]))
    chk["reproduced"] = ok
    report["reproduction_check"] = chk
    json.dump(report, open(out_p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, indent=1))
    if not ok:
        sys.exit("STRICT RESULTS DO NOT REPRODUCE")


if __name__ == "__main__":
    main()
