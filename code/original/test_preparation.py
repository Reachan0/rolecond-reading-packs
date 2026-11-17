"""Meaningful offline edge checks; no credentials or model calls."""
import json
import unittest
from presentation_runner import validate_response
from prepare_evidence import metrics, OUT
from audit_presentation_formats import unwrap_json_fence, classify_text
from analyze_external_failures import cluster_inference
from analyze_presentation import collect, summarize


class PreparationTests(unittest.TestCase):
    def test_format_audit_unwraps_only_complete_json_fences(self):
        raw='```json\n{"sections":[]}\n```'
        self.assertEqual(unwrap_json_fence(raw),'{"sections":[]}')
        self.assertIsNone(unwrap_json_fence('prefix ```json\n{}\n```'))
        self.assertIsNone(unwrap_json_fence('```json\n{}'))

    def test_format_audit_keeps_strict_and_posthoc_states_distinct(self):
        self.assertEqual(classify_text('{"sections":[]}')["format_class"],"strict_json")
        fenced=classify_text('```json\n{"sections":[]}\n```')
        self.assertEqual(fenced["format_class"],"json_in_code_fence")
        self.assertTrue(fenced["posthoc_json_parseable"])
        self.assertEqual(classify_text("")["format_class"],"empty")

    def test_missing_and_invalid_stay_in_denominator(self):
        inputs=[dict(query_id=1,condition=c,input_ids=[1,2]) for c in ("F","G","R","W")]
        records=[dict(job="1:F:m",query_id=1,condition="F",requested_model="m",status="response",
                      response={"choices":[{"message":{"content":"bad JSON"}}]})]
        rows=collect(inputs,["m"],records,{1:{1:2,2:0}})
        report=summarize(rows,["m"])
        self.assertEqual(report["planned_jobs"],4)
        self.assertEqual(report["valid_jobs"],0)
        self.assertFalse(report["operationally_terminal"])
        self.assertEqual(report["summary"]["m:F"]["states"],{"invalid_response":1})
        self.assertEqual(report["primary_tests"],{})

    def test_duplicate_and_unknown_jobs_rejected(self):
        inputs=[dict(query_id=1,condition="F",input_ids=[1])]
        record=dict(job="1:F:m",query_id=1,condition="F",requested_model="m",status="response")
        with self.assertRaises(ValueError):
            collect(inputs,["m"],[record,record],{1:{1:2}})
        with self.assertRaises(ValueError):
            collect(inputs,["other"],[record],{1:{1:2}})

    def test_two_models_are_one_query_cluster(self):
        rows=[]
        for q in range(3):
            for model in ("a","b"):
                for c,value in (("F",0.),("G",.5),("R",1.),("W",0.)):
                    rows.append(dict(query_id=q,model=model,condition=c,state="valid",valid=True,
                                     input_coverage=value))
        result=summarize(rows,["a","b"])
        self.assertEqual(result["primary_tests"]["both_models:R-G"]["n"],3)
        self.assertEqual(result["primary_tests"]["both_models:R-G"]["delta"],.5)

    def test_normalization_penalizes_omission(self):
        self.assertLess(metrics([2],{1:3,2:1})["ndcg"],.15)
        self.assertEqual(metrics([1,2],{1:3,2:1})["ndcg"],1.)

    def test_unknown_and_duplicate_candidates_fail(self):
        for ids in ([1,1],[2]):
            with self.assertRaises(AssertionError):
                metrics(ids,{1:1})

    def test_empty_and_zero_positive(self):
        self.assertEqual(metrics([],{1:2})["recall"],0)
        self.assertIsNone(metrics([1],{1:0})["recall"])

    def test_outside_citations_cannot_raise_input_coverage(self):
        text=json.dumps({"sections":[{"theme":"A","papers":[
            {"id":999,"use":"method background transfer"},{"id":1,"use":"comparison"}]}]})
        m=validate_response(text,[1,2],[2])
        self.assertEqual(m["input_coverage"],.5)
        self.assertEqual(m["silver_positive_coverage"],0.)
        self.assertEqual(m["outside_reference_fraction"],.5)

    def test_schema_failures_not_counted_as_valid(self):
        for text in ('not json','{"sections":[]}',
                     '{"sections":[{"theme":"a","papers":[{"id":true,"use":"a"}]}]}'):
            self.assertFalse(validate_response(text,[1],[1])["valid"])

    def test_vocabulary_ignores_outside_ids_and_counts_each_category_once(self):
        text=json.dumps({"sections":[{"theme":"background transfer","papers":[
            {"id":999,"use":"background transfer"},
            {"id":1,"use":"METHOD method baseline"},
            {"id":1,"use":"Compare with prior work"}]}]})
        result=validate_response(text,[1,2],[1])
        self.assertEqual(result["use_vocabulary_coverage"],.5)
        self.assertEqual(result["use_vocabulary_hits"],["compare_method","position"])

    def test_cluster_macro_preserves_facet_weights(self):
        rows=[dict(query_id="a",facet="background",delta=1),
              dict(query_id="b",facet="background",delta=0),
              dict(query_id="a",facet="method",delta=0),
              dict(query_id="b",facet="result",delta=0)]
        self.assertAlmostEqual(cluster_inference(rows)["delta"],1/6)

    def test_four_condition_inputs_fixed(self):
        inputs=json.loads((OUT/"P1_inputs.json").read_text())
        for qid in {r["query_id"] for r in inputs}:
            rows=[r for r in inputs if r["query_id"]==qid]
            self.assertEqual({r["condition"] for r in rows},{"F","G","R","W"})
            for r in rows:
                self.assertEqual(r["input_ids"],rows[0]["input_ids"])
                self.assertEqual(r["groups"],rows[0]["groups"])
                self.assertTrue(all(i!=v for i,v in enumerate(r["wrong_label_permutation"])))


if __name__=="__main__":
    unittest.main()
