"""Audit P1 response formatting without changing the preregistered strict analysis."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analyze_presentation import collect, summarize
from prepare_evidence import OUT, ROOT, read, save
from presentation_runner import validate_response


def unwrap_json_fence(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text,
                         flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def classify_text(text):
    if not isinstance(text, str) or not text.strip():
        return {"format_class": "empty", "posthoc_json_parseable": False}
    try:
        json.loads(text)
        return {"format_class": "strict_json", "posthoc_json_parseable": True}
    except (TypeError, ValueError):
        pass
    unwrapped = unwrap_json_fence(text)
    if unwrapped is not None:
        try:
            json.loads(unwrapped)
            return {"format_class": "json_in_code_fence",
                    "posthoc_json_parseable": True}
        except (TypeError, ValueError):
            return {"format_class": "malformed_code_fence",
                    "posthoc_json_parseable": False}
    return {"format_class": "other_non_json", "posthoc_json_parseable": False}


def main():
    run_dir = OUT / "presentation_run_v2"
    attempts = [json.loads(line) for line in
                (run_dir / "attempts.jsonl").read_text().splitlines()]
    inputs = read(OUT / "P1_inputs.json")
    config = read(run_dir / "RUN_LOCK.json")["config"]
    successful = [row for row in attempts if row["status"] == "response"]
    by_job = {row["job"]: row for row in successful}
    assert len(by_job) == len(successful) == 288

    input_lookup = {(row["query_id"], row["condition"]): row for row in inputs}
    counts = Counter()
    by_model_condition = defaultdict(Counter)
    posthoc_schema_valid = Counter()
    transformed = []
    records = []
    for row in successful:
        text = row["response"]["choices"][0]["message"].get("content", "")
        classification = classify_text(text)
        key = (row["requested_model"], row["condition"])
        counts[classification["format_class"]] += 1
        by_model_condition[key][classification["format_class"]] += 1
        source = input_lookup[(row["query_id"], row["condition"])]
        candidate = unwrap_json_fence(text)
        candidate = candidate if candidate is not None else text
        positive_ids = []
        schema = validate_response(candidate, source["input_ids"], positive_ids)
        if schema["valid"]:
            posthoc_schema_valid[key] += 1
        copied = dict(row)
        copied["response"] = json.loads(json.dumps(row["response"]))
        copied["response"]["choices"][0]["message"]["content"] = candidate
        transformed.append(copied)
        records.append({
            "job": row["job"],
            "model": row["requested_model"],
            "condition": row["condition"],
            "format_class": classification["format_class"],
            "strict_schema_valid": validate_response(
                text, source["input_ids"], positive_ids)["valid"],
            "posthoc_schema_valid": schema["valid"],
            "finish_reason": row["response"]["choices"][0].get("finish_reason"),
        })

    benchmark = read(ROOT / "src/core/literature_discovery_benchmark_silver.json")
    golds = {q["query_paper_id"]: {
        c["candidate_paper_id"]: c["silver_grade"] for c in q["candidates"]}
        for q in benchmark}
    sensitivity_rows = collect(inputs, config["models"], transformed, golds)
    sensitivity = summarize(sensitivity_rows, config["models"])
    output = {
        "status": "posthoc_sensitivity_only",
        "warning": ("Removing complete Markdown JSON fences was specified only after "
                    "observing outputs. It does not replace strict preregistered validity."),
        "response_count": len(successful),
        "format_counts": dict(counts),
        "by_model_condition": {
            f"{model}:{condition}": dict(values)
            for (model, condition), values in sorted(by_model_condition.items())},
        "posthoc_schema_valid": {
            f"{model}:{condition}": posthoc_schema_valid[(model, condition)]
            for model in config["models"] for condition in ("F", "G", "R", "W")},
        "sensitivity": sensitivity,
        "per_job": records,
    }
    save("P1_format_audit.json", output)
    print(json.dumps({
        "responses": output["response_count"],
        "format_counts": output["format_counts"],
        "posthoc_valid_jobs": sensitivity["valid_jobs"],
    }))


if __name__ == "__main__":
    main()
