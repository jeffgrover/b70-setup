"""CPU-only checks for benchmark isolation and production-command selection."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_sycl
import validate_sycl


class ProfileTests(unittest.TestCase):
    def test_case_names_are_unique(self):
        for suite, count in [("comparison", 18), ("tuning", 6)]:
            cases = profile_sycl.cases(suite)
            self.assertEqual(len(cases), count)
            self.assertEqual(len({case["name"] for case in cases}), count)

    def test_saved_build_libraries_take_precedence(self):
        build = Path("/tmp/saved-sycl-build")
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/other-build/bin",
                                     "GGML_SYCL_ENABLE_GRAPH": "1"}):
            env = profile_sycl.runtime_env(build)
        self.assertEqual(env["LD_LIBRARY_PATH"],
                         "/tmp/saved-sycl-build/bin:/tmp/other-build/bin")
        self.assertEqual(env["GGML_SYCL_ENABLE_GRAPH"], "0")
        self.assertEqual(env["GGML_SYCL_FA_ONEDNN"], "1")
        self.assertEqual(profile_sycl.runtime_env(build, "mkl")["GGML_SYCL_FA_ONEDNN"], "0")

    def test_production_profile_variants(self):
        for kv, context in [("f16", "131072"), ("q8_0", "262144")]:
            args = argparse.Namespace(kv=kv, build=Path("/tmp/candidate"),
                                      port=19001, batch=None, ubatch=None,
                                      config=profile_sycl.ROOT / "llama-swap.yaml")
            for horizon in range(5):
                command = profile_sycl.profile_command(args, horizon)
                self.assertEqual(command[0], "/tmp/candidate/bin/llama-server")
                self.assertEqual(command[command.index("--port") + 1], "19001")
                self.assertEqual(command[command.index("-c") + 1], context)
                self.assertEqual(command[command.index("-ctk") + 1], kv)
                self.assertIn("--mmproj", command)
                if horizon:
                    self.assertEqual(command[command.index("--spec-draft-n-max") + 1], str(horizon))
                else:
                    self.assertNotIn("--spec-type", command)
                    self.assertNotIn("--spec-draft-n-max", command)

    def test_batch_overrides_are_independent(self):
        args = argparse.Namespace(kv="f16", build=Path("/tmp/candidate"),
                                  port=19001, batch=None, ubatch=1024,
                                  config=profile_sycl.ROOT / "llama-swap.yaml")
        command = profile_sycl.profile_command(args, 3)
        self.assertEqual(command[-2:], ["-ub", "1024"])
        args.batch = 4096
        command = profile_sycl.profile_command(args, 3)
        self.assertEqual(command[-4:], ["-b", "4096", "-ub", "1024"])

    def test_bench_overrides_match_names_and_recorded_parameters(self):
        with tempfile.TemporaryDirectory(prefix="sycl-profiler-test-") as temporary:
            directory = Path(temporary)
            args = argparse.Namespace(suite="comparison", fa_backend="auto", batch=4096,
                                      ubatch=1024, cases="qwen-f16-p8192-n0-d0-b4096-ub1024",
                                      build=directory, raw=directory, results=directory / "results.jsonl",
                                      label="test", repeat=False)
            response = SimpleNamespace(returncode=0, stdout=json.dumps([
                dict(n_prompt=8192, n_gen=0, avg_ts=1.0, stddev_ts=0.0)]))
            with patch("profile_sycl.subprocess.run", return_value=response) as run, patch("builtins.print"):
                profile_sycl.bench(args)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-b") + 1], "4096")
            self.assertEqual(command[command.index("-ub") + 1], "1024")
            recorded = json.loads(args.results.read_text())
            self.assertEqual(recorded["case"]["batch"], 4096)
            self.assertEqual(recorded["case"]["ubatch"], 1024)
            self.assertTrue((directory / "test-qwen-f16-p8192-n0-d0-b4096-ub1024.json").exists())


class ValidationTests(unittest.TestCase):
    def test_tool_round_trip_uses_matching_call_id(self):
        first = {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_test", "type": "function", "function": {"name": "read_status", "arguments": "{}"}}]}}]}
        second = {"choices": [{"message": {"content": "READY-2415"}}]}
        with patch("validate_sycl.request", side_effect=[first, second]) as request:
            result = validate_sycl.tool_round_trip(19001, "test-model")
        self.assertEqual(request.call_count, 2)
        body = request.call_args.args[2]
        self.assertEqual(body["messages"][-1]["tool_call_id"], "call_test")
        self.assertEqual(body["tool_choice"], "none")
        self.assertEqual(result["second"], second)

    def test_wrong_tool_is_not_a_success(self):
        wrong = {"choices": [{"message": {"tool_calls": [
            {"id": "call_test", "function": {"name": "other_tool", "arguments": "{}"}}]}}]}
        with patch("validate_sycl.request", return_value=wrong) as request:
            with self.assertRaises(AssertionError):
                validate_sycl.tool_round_trip(19001, "test-model")
        request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
