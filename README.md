# Role-Conditioned Reading Packs: Supplementary Material

Supplementary material for

> X. Chen, H. Zhu, and C. Yin, "Role-Conditioned Reading Packs for Scholarly Discovery: Selection, Presentation, and Trade-offs," in *Proc. IEEE ISPA 2026*.

It contains the mechanism parameters, the benchmark with its rule-based grades, the RoleCond outputs, the frozen evidence behind every number in the paper, all 288 saved model responses of the presentation experiment, and the analysis code. No model requests are needed to reproduce the analyses.

## Contents

| Path | What it is |
|---|---|
| `METHOD_PARAMETERS.md` | All admission gates, thresholds, and scoring coefficients (Section III). |
| `data/benchmark.json` | The 90-query OpenAlex benchmark: 656 query–candidate pairs in construction order, with metadata, rule-based score, grade (0–3), and the signals that produced it (Section IV-A). |
| `data/rolecond_outputs.jsonl` | Per-query selections of RoleCond and the control systems. |
| `evidence/` | Frozen per-query results and statistics of the selection, sensitivity, CSFCube, and failure-case analyses. |
| `presentation/inputs.json` | The 144 prompts (36 queries × 4 conditions), each sent to both models. |
| `presentation/raw_attempts.jsonl` | All 292 request attempts for the 288 jobs, with the full model responses. |
| `presentation/*.json` | Strict analysis, format audit, fence-stripped metrics, and the model × condition interaction. |
| `code/` | Standalone scripts that reproduce the camera-ready additions from the files above. |
| `code/original/` | The original pipeline that produced `evidence/` and the presentation run (see below). |
| `MANIFEST.sha256` | SHA-256 checksums of every file. |

## Paper element → evidence

| Paper | Evidence | Script |
|---|---|---|
| Table I, gates and coefficients | `METHOD_PARAMETERS.md` | — |
| Table II, Fig. 2: selection controls | `evidence/P0_selection.json`, `evidence/figure_statistics.json` | `code/original/prepare_evidence.py`, `code/original/plot_evidence.py` |
| Table II fixed-k rows, Fig. 3 | `evidence/fixed_quantity_results.json` | `code/fixed_quantity.py`, `code/plot_frontier.py` |
| Fig. 4, order / quota / allocation sensitivity | `evidence/P0_grid.json`, `evidence/P0_order_ranges.json` | `code/original/prepare_evidence.py` |
| Fig. 5, CSFCube diagnostic | `evidence/P0_external.json`, `evidence/P0_external_inference.json` | `code/original/prepare_evidence.py`, `code/original/analyze_external_failures.py` |
| Failure cases (Section V-B) | `evidence/P0_failure_audit.json` | `code/original/analyze_external_failures.py` |
| Tables III–IV, Fig. 6: presentation | `presentation/relaxed_metrics.json`, `presentation/interaction_results.json`, `presentation/reparse_results.json` | `code/relaxed_metrics.py`, `code/interaction.py`, `code/reparse.py`, `code/plot_presentation_both.py` |
| Strict format compliance | `presentation/analysis.json`, `presentation/format_audit.json` | `code/original/analyze_presentation.py`, `code/original/audit_presentation_formats.py` |

## Reproducing

Python 3.10+ with NumPy; the figure scripts also need Matplotlib.

```sh
mkdir -p out
python code/fixed_quantity.py data/benchmark.json data/rolecond_outputs.jsonl out/fixed_quantity.json
python code/plot_frontier.py out/fixed_quantity.json out/frontier

A="presentation/raw_attempts.jsonl presentation/inputs.json data/benchmark.json"
python code/reparse.py $A presentation/analysis.json out/reparse.json
python code/relaxed_metrics.py $A out/relaxed_metrics.json
python code/interaction.py $A out/interaction.json
python code/plot_presentation_both.py out/relaxed_metrics.json out/presentation
```

Each output JSON is identical to the corresponding file in `evidence/` or `presentation/`. All inference uses seed 20260915 (10,000 bootstrap resamples, 50,000 sign-flip permutations); figure intervals use seed 20260929 and 20,000 resamples.

The scripts in `code/original/` are kept as run, for provenance. They expect the directory layout of the original project, including inputs that are not redistributed here (the grid-sweep and CSFCube score files, and the CSFCube annotations, available from the CSFCube authors), and some of them write Chinese-language summary reports. Their outputs are frozen in `evidence/` and `presentation/`. `presentation_runner.py` is the client that sent the presentation requests; it reads the endpoint and credential from a local configuration that is not included.

## Notes

- Grades are computed by rules from metadata and citation links (Section IV-A). They are not human relevance judgments, and no human labels exist for the four reading roles.
- Candidate summaries from the source corpus are omitted from `data/benchmark.json`; paper IDs refer to the Aella Science Dataset Explorer corpus, enriched with OpenAlex metadata.
- Presentation responses were collected at temperature 0 with one response per query, condition, and model, through an API proxy that does not expose vendor snapshot identifiers.

## License

Code: MIT. Data and evidence files: CC BY 4.0.
