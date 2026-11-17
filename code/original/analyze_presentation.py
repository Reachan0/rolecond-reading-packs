"""Analyze only journaled P1 responses. Missing/invalid jobs stay in denominators."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_evidence import OUT, ROOT, infer, holm, read, save
from presentation_runner import validate_response, sha


def collect(inputs, models, attempts, golds):
    expected = {f"{r['query_id']}:{r['condition']}:{m}": (r, m)
                for r in inputs for m in models}
    grouped = defaultdict(list)
    for a in attempts:
        if a["job"] not in expected:
            raise ValueError("Unexpected job in response journal")
        row, model = expected[a["job"]]
        if (a["query_id"],a["condition"],a["requested_model"]) != (row["query_id"],row["condition"],model):
            raise ValueError("Journal identity mismatch")
        grouped[a["job"]].append(a)
    output=[]
    for job,(row,model) in expected.items():
        logs=grouped[job]
        successes=[a for a in logs if a["status"]=="response"]
        if len(logs)>3 or len(successes)>1:
            raise ValueError("More than three attempts or duplicate response")
        if successes and logs[-1] is not successes[0]:
            raise ValueError("Response was followed by an unauthorized retry")
        result=dict(job=job,query_id=row["query_id"],condition=row["condition"],
                    model=model,attempts=len(logs),state="missing",valid=False)
        if successes:
            payload=successes[0].get("response")
            result["state"]="invalid_response"
            try:
                text=payload["choices"][0]["message"]["content"]
                if not isinstance(text,str):
                    raise TypeError()
                positive=[i for i in row["input_ids"] if golds[row["query_id"]][i]>=2]
                result.update(validate_response(text,row["input_ids"],positive))
                result["returned_model"]=payload.get("model")
                result["usage"]=payload.get("usage")
                result["finish_reason"]=payload["choices"][0].get("finish_reason")
                result["state"]="valid" if result["valid"] else "invalid_response"
            except (KeyError,IndexError,TypeError):
                result["failure"]="malformed_api_envelope"
        elif logs:
            result["state"]="exhausted_transport_failure" if len(logs)==3 else "incomplete_transport_failure"
        output.append(result)
    return output


def summarize(rows,models):
    groups=defaultdict(list)
    for r in rows:
        groups[r["model"]+":"+r["condition"]].append(r)
    summaries={}
    for key,items in groups.items():
        valid=[r for r in items if r["valid"]]
        summaries[key]=dict(planned=len(items),valid=len(valid),states=dict(Counter(r["state"] for r in items)),
                           valid_rate=len(valid)/len(items))
        for metric in ("input_coverage","silver_positive_coverage","outside_reference_fraction","redundancy",
                       "use_vocabulary_coverage"):
            vals=[r[metric] for r in valid if r.get(metric) is not None]
            summaries[key][metric]=sum(vals)/len(vals) if vals else None
            summaries[key][metric+"_n"]=len(vals)
    lookup={(r["query_id"],r["model"],r["condition"]):r for r in rows}
    qids=sorted({r["query_id"] for r in rows})
    tests={}
    auxiliary={}
    for left,right in (("G","F"),("R","G"),("R","W")):
        target=auxiliary if right=="W" else tests
        permodel={}
        for model in models:
            diffs={}
            for q in qids:
                a,b=lookup[q,model,left],lookup[q,model,right]
                if a["valid"] and b["valid"] and a.get("input_coverage") is not None and b.get("input_coverage") is not None:
                    diffs[q]=a["input_coverage"]-b["input_coverage"]
            permodel[model]=diffs
            if diffs:
                target[model+":"+left+"-"+right]=dict(**infer(list(diffs.values())),
                    complete_queries=sorted(diffs),planned_queries=len(qids))
        common=set(qids)
        for vals in permodel.values():
            common &= set(vals)
        if common:
            target["both_models:"+left+"-"+right]=dict(
                **infer([sum(permodel[m][q] for m in models)/len(models) for q in sorted(common)]),
                complete_queries=sorted(common),planned_queries=len(qids),
                unit="query; equally average model differences, requiring both model pairs")
    if tests:
        holm(tests)
    return dict(summary=summaries,primary_tests=tests,auxiliary_tests=auxiliary,
                planned_jobs=len(rows),valid_jobs=sum(r["valid"] for r in rows),
                operationally_terminal=all(r["state"] in ("valid","invalid_response","exhausted_transport_failure") for r in rows),
                warning="Conditional valid-pair analysis; missingness may bias effects. Coverage is not content quality.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=OUT/"presentation_run")
    args=parser.parse_args()
    lock=args.run_dir/"RUN_LOCK.json"
    if not lock.exists():
        print("No locked experiment exists; no results produced.")
        return
    identity=read(lock)
    inputs=read(OUT/"P1_inputs.json")
    assert identity["input_sha256"]==hashlib.sha256((OUT/"P1_inputs.json").read_bytes()).hexdigest()
    attempts_path=args.run_dir/"attempts.jsonl"
    attempts=[json.loads(s) for s in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
    config=identity["config"]
    lookup={f"{r['query_id']}:{r['condition']}:{m}":(r,m) for r in inputs for m in config["models"]}
    for a in attempts:
        r,m=lookup[a["job"]]
        body={"model":m,"messages":[{"role":"user","content":r["prompt"]}],"temperature":0,
              "max_tokens":config["max_completion_tokens"]}
        assert a["request_sha256"]==sha(body)
    benchmark=read(ROOT/"src/core/literature_discovery_benchmark_silver.json")
    golds={q["query_paper_id"]:{c["candidate_paper_id"]:c["silver_grade"] for c in q["candidates"]} for q in benchmark}
    rows=collect(inputs,config["models"],attempts,golds)
    report=summarize(rows,config["models"])
    report["pending_request_intent"]=(args.run_dir/"pending_request.json").exists()
    report["attempt_count"]=len(attempts)
    report["complete_run"]=report["operationally_terminal"] and not report["pending_request_intent"]
    save("P1_analysis.json",dict(**report,per_job=rows))
    print(json.dumps({k:report[k] for k in ("planned_jobs","valid_jobs","attempt_count","complete_run")}))


if __name__=="__main__":
    main()
