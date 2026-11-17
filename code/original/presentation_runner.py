"""Resumable, explicitly configured OpenAI-compatible presentation experiment.

Only --run performs network requests. Configuration must set base_url, two model
IDs, credential_env, max_completion_tokens, timeout_seconds, and request_limit.
Do not store credentials in configuration or outputs.
"""
import argparse
import hashlib
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
# Frozen historical E3 vocabulary, matched as lowercase substrings.
# Measures wording only; never role assignment or citation correctness.
ROLE_USE_KEYWORDS = {
    "position": {"position", "compare", "contrast", "similar", "related", "overlap", "competitor", "rival", "nearby", "neighbor", "prior"},
    "compare_method": {"method", "technique", "approach", "algorithm", "framework", "pipeline", "baseline", "benchmark", "implementation", "architecture"},
    "ground": {"background", "foundation", "survey", "review", "introduction", "overview", "seminal", "fundamental", "grounding", "basis", "context"},
    "transfer": {"transfer", "bridge", "cross", "interdisciplinary", "borrow", "adapt", "inspire", "adjacent", "novel application", "different field"},
}


def sha(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def validate_response(text, input_ids, positive_ids):
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return dict(valid=False, failure="invalid_json")
    if not isinstance(obj,dict) or not isinstance(obj.get("sections"),list) or not obj["sections"]:
        return dict(valid=False, failure="invalid_sections")
    refs = []
    for section in obj["sections"]:
        if not isinstance(section,dict) or not isinstance(section.get("theme"),str) or not isinstance(section.get("papers"),list):
            return dict(valid=False,failure="invalid_section_schema")
        for item in section["papers"]:
            if (not isinstance(item,dict) or type(item.get("id")) is not int
                    or not isinstance(item.get("use"),str)):
                return dict(valid=False,failure="invalid_citation_schema")
            refs.append(item["id"])
    valid_ids = set(input_ids)
    positives = set(positive_ids)
    cited = set(refs)
    role_hits = set()
    for section in obj["sections"]:
        for item in section["papers"]:
            if item["id"] in valid_ids:
                text = item["use"].lower()
                role_hits.update(role for role, words in ROLE_USE_KEYWORDS.items()
                                 if any(word in text for word in words))
    return dict(valid=True, section_count=len(obj["sections"]),
                use_vocabulary_coverage=len(role_hits)/4,
                use_vocabulary_hits=sorted(role_hits),
                input_coverage=len(cited & valid_ids)/len(valid_ids) if valid_ids else None,
                silver_positive_coverage=len(cited & positives)/len(positives) if positives else None,
                outside_reference_fraction=sum(x not in valid_ids for x in refs)/len(refs) if refs else 0.,
                redundancy=1-len(cited)/len(refs) if refs else 0.,
                citation_count=len(refs), unique_valid_citations=len(cited & valid_ids))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--run-dir",type=Path)
    args=parser.parse_args()
    inputs=json.loads((OUT/"P1_inputs.json").read_text())
    assert len(inputs)==144
    if not args.config:
        print(json.dumps(dict(input_conditions=len(inputs),queries=len({r["query_id"] for r in inputs}),
                              models_locked=False,requests_sent=0,
                              next_step="Supply a verified two-model configuration and credential environment")))
        return
    config=json.loads(args.config.read_text())
    required={"base_url","models","credential_env","max_completion_tokens","timeout_seconds","request_limit"}
    assert required <= config.keys()
    assert len(config["models"])==len(set(config["models"]))==2
    secure_url = config["base_url"].startswith("https://")
    insecure_opt_in = config.get("allow_insecure_http") is True
    assert secure_url or (config["base_url"].startswith("http://") and insecure_opt_in), \
        "HTTP requires explicit allow_insecure_http=true"
    assert type(config["request_limit"]) is int and config["request_limit"] >= 288
    assert config["max_completion_tokens"] > 0 and 0 < config["timeout_seconds"] <= 600
    key=os.environ.get(config["credential_env"])
    if not args.run:
        print(json.dumps(dict(config_valid=True,credential_present=bool(key),requests_sent=0,
                              planned_jobs=288,request_limit=config["request_limit"],
                              insecure_http=not secure_url)))
        return
    if not key:
        raise SystemExit("Required API credential environment variable is absent; no requests sent.")
    if not secure_url:
        print("WARNING: explicit insecure HTTP opt-in; API credential is transmitted without TLS.", flush=True)
    run_dir=args.run_dir or OUT/"presentation_run"
    run_dir.mkdir(exist_ok=True)
    lockpath=run_dir/"RUN_LOCK.json"
    identity={"config":config,"input_sha256":hashlib.sha256((OUT/"P1_inputs.json").read_bytes()).hexdigest()}
    if lockpath.exists():
        assert json.loads(lockpath.read_text())==identity, "Config/input changed; cannot mix runs"
    else:
        lockpath.write_text(json.dumps(identity,indent=2)+"\n")
    # Exclusive process marker. Stale files must be investigated, never automatically removed.
    process_lock=run_dir/"process.lock"
    fd=os.open(process_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        pending=run_dir/"pending_request.json"
        if pending.exists():
            raise RuntimeError("Unresolved prior request intent; inspect before resume.")
        log=run_dir/"attempts.jsonl"
        history=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
        jobs=[(r,m) for r in inputs for m in config["models"]]
        expected={f"{r['query_id']}:{r['condition']}:{m}": (r,m) for r,m in jobs}
        assert len(expected)==288, "Duplicate planned jobs"
        counts={}
        done=set()
        for row in history:
            if row["job"] not in expected or row["job"] in done:
                raise ValueError("Unknown job or attempts after terminal result")
            source,model=expected[row["job"]]
            body={"model":model,"messages":[{"role":"user","content":source["prompt"]}],
                  "temperature":0,"max_tokens":config["max_completion_tokens"]}
            if (row.get("query_id")!=source["query_id"] or row.get("condition")!=source["condition"]
                    or row.get("requested_model")!=model or row.get("request_sha256")!=sha(body)
                    or row.get("status") not in {"response","http_error","transport_or_decode_error"}):
                raise ValueError("Attempt journal does not match locked requests")
            counts[row["job"]]=counts.get(row["job"],0)+1
            if row["status"]=="response" or counts[row["job"]]>=3:
                done.add(row["job"])
        random.Random(20260915).shuffle(jobs)
        total=len(history)
        for row,model in jobs:
            job=f"{row['query_id']}:{row['condition']}:{model}"
            if job in done:
                continue
            while counts.get(job,0)<3:
                if total>=config["request_limit"]:
                    print("Configured total request ceiling reached; checkpoint retained.")
                    return
                body={"model":model,"messages":[{"role":"user","content":row["prompt"]}],
                      "temperature":0,"max_tokens":config["max_completion_tokens"]}
                # Intent journal precedes network call: a crash cannot silently duplicate an unlogged request.
                pending=run_dir/"pending_request.json"
                if pending.exists():
                    raise RuntimeError("Unresolved prior request intent; inspect before resume.")
                intent=dict(job=job,started_at=time.time(),request_sha256=sha(body))
                pending.write_text(json.dumps(intent)+"\n")
                started=time.monotonic()
                uncertain=False
                record=dict(**intent,query_id=row["query_id"],condition=row["condition"],requested_model=model)
                req=urllib.request.Request(config["base_url"].rstrip("/")+"/chat/completions",
                    data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
                try:
                    with urllib.request.urlopen(req,timeout=config["timeout_seconds"]) as response:
                        payload=json.loads(response.read())
                    # Store model response fields; never request headers, secrets, or server error body.
                    record.update(status="response",response=payload,
                                  elapsed_seconds=time.monotonic()-started)
                except urllib.error.HTTPError as exc:
                    record.update(status="http_error",http_status=exc.code)
                except Exception as exc:
                    record.update(status="transport_or_decode_error",error_type=type(exc).__name__)
                    uncertain=isinstance(exc,(TimeoutError,socket.timeout)) or (
                        isinstance(exc,urllib.error.URLError) and
                        isinstance(exc.reason,(TimeoutError,socket.timeout)))
                    if uncertain:
                        record["server_completion_unknown"]=True
                with log.open("a") as f:
                    f.write(json.dumps(record,ensure_ascii=False)+"\n")
                    f.flush(); os.fsync(f.fileno())
                if uncertain:
                    raise RuntimeError("Uncertain server completion after timeout; intent retained, inspect before resume.")
                pending.unlink()
                total+=1
                counts[job]=counts.get(job,0)+1
                print(json.dumps(dict(job=job,status=record["status"],total_requests=total)),flush=True)
                if record["status"]=="response":
                    break
                if record.get("http_status") in (401,403,404):
                    print("Authentication/endpoint/model error; stopped without exhausting other jobs.")
                    return
                time.sleep(2)
    finally:
        process_lock.unlink()


if __name__=="__main__":
    main()
