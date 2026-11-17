"""Exercise transmission and restart safety with mocked HTTP only."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import presentation_runner as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "P1_inputs.json").write_bytes((runner.OUT / "P1_inputs.json").read_bytes())
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(dict(
            base_url="https://mock.invalid/v1", models=["mock-a", "mock-b"],
            credential_env="ISPA_TEST_KEY", max_completion_tokens=100,
            timeout_seconds=1, request_limit=864)))

    def invoke(self, side_effect=None, extra_args=None):
        with patch.object(runner, "OUT", self.root), \
             patch("sys.argv", ["runner", "--config", str(self.config), "--run"] + (extra_args or [])), \
             patch.dict(os.environ, {"ISPA_TEST_KEY": "synthetic-test-value"}), \
             patch.object(runner.urllib.request, "urlopen", side_effect=side_effect) as call, \
             patch.object(runner.time, "sleep"), patch("builtins.print"):
            runner.main()
        return call

    def response(self, *args, **kwargs):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "invalid output"}}]}).encode())

    def records(self):
        return [json.loads(line) for line in
                (self.root / "presentation_run/attempts.jsonl").read_text().splitlines()]

    def test_success_invalid_content_is_not_regenerated_and_resume_is_noop(self):
        self.assertEqual(self.invoke(self.response).call_count, 288)
        self.assertEqual(len(self.records()), 288)
        self.assertEqual(self.invoke(self.response).call_count, 0)
        self.assertFalse((self.root / "presentation_run/process.lock").exists())

    def test_transient_http_failure_retries_then_completes(self):
        attempts = 0
        def respond(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise urllib.error.HTTPError("https://mock.invalid", 429, "", {}, None)
            return self.response()
        self.assertEqual(self.invoke(respond).call_count, 289)
        self.assertEqual(self.records()[0]["http_status"], 429)
        self.assertEqual(self.records()[0]["job"], self.records()[1]["job"])

    def test_auth_failure_stops_and_can_resume(self):
        def unauthorized(*args, **kwargs):
            raise urllib.error.HTTPError("https://mock.invalid", 401, "", {}, None)
        self.assertEqual(self.invoke(unauthorized).call_count, 1)
        self.assertEqual(self.invoke(self.response).call_count, 288)

    def test_transport_failure_exhaustion_is_terminal(self):
        self.assertEqual(self.invoke(OSError("mock failure")).call_count, 864)
        self.assertEqual(self.invoke(self.response).call_count, 0)

    def test_pending_intent_blocks_even_after_all_jobs_complete(self):
        self.invoke(self.response)
        (self.root / "presentation_run/pending_request.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            self.invoke(self.response)
        self.assertEqual(len(self.records()), 288)

    def test_duplicate_journal_blocks_resume(self):
        self.invoke(self.response)
        log = self.root / "presentation_run/attempts.jsonl"
        with log.open("a") as stream:
            stream.write(json.dumps(self.records()[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.invoke(self.response)

    def test_interrupted_request_preserves_intent(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(KeyboardInterrupt())
        self.assertTrue((self.root / "presentation_run/pending_request.json").exists())
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            self.invoke(self.response)

    def test_existing_process_lock_is_preserved(self):
        directory = self.root / "presentation_run"
        directory.mkdir()
        marker = directory / "process.lock"
        marker.write_text("mock owner")
        with self.assertRaises(FileExistsError):
            self.invoke(self.response)
        self.assertEqual(marker.read_text(), "mock owner")

    def test_separate_run_directory_and_long_timeout(self):
        config=json.loads(self.config.read_text())
        config["timeout_seconds"]=600
        self.config.write_text(json.dumps(config))
        target=self.root/"new_run"
        call=self.invoke(self.response, ["--run-dir",str(target)])
        self.assertEqual(call.call_count,288)
        self.assertEqual(call.call_args.kwargs["timeout"],600)
        self.assertTrue((target/"attempts.jsonl").exists())
        self.assertFalse((self.root/"presentation_run").exists())

    def test_timeout_preserves_intent_and_prevents_duplicate_submission(self):
        with self.assertRaisesRegex(RuntimeError,"Uncertain"):
            self.invoke(TimeoutError())
        self.assertTrue((self.root/"presentation_run/pending_request.json").exists())
        self.assertEqual(len(self.records()),1)
        with self.assertRaisesRegex(RuntimeError,"Unresolved"):
            self.invoke(self.response)


if __name__ == "__main__":
    unittest.main()
