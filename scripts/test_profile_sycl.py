"""CPU-only checks for benchmark isolation and production-command selection."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import profile_sycl
import validate_sycl
import inspect_gguf


class ProfileTests(unittest.TestCase):
    def test_case_names_are_unique(self):
        for suite, count in [("comparison", 18), ("tuning", 6), ("swift", 8), ("swift-tuning", 14)]:
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

    def test_swift_alias_and_sampling_mode(self):
        args = argparse.Namespace(kv="f16", alias="swift-1.5-27b-think",
                                  draft_sampling="probabilistic", build=Path("/tmp/candidate"),
                                  port=19001, batch=None, ubatch=None,
                                  config=profile_sycl.ROOT / "llama-swap.yaml")
        command = profile_sycl.profile_command(args, 2)
        self.assertIn("Swift-1.5", command[command.index("-m") + 1])
        self.assertEqual(command.count("--spec-draft-sampling"), 1)
        self.assertEqual(command[command.index("--spec-draft-sampling") + 1], "probabilistic")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "2")
        plain = profile_sycl.profile_command(args, 0)
        self.assertNotIn("--spec-type", plain)
        self.assertNotIn("--spec-draft-n-max", plain)
        self.assertNotIn("--spec-draft-sampling", plain)
        args.kv = "q8_0"
        with self.assertRaises(ValueError):
            profile_sycl.profile_command(args, 2)

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

    def test_configured_sampling_can_be_overridden_or_removed(self):
        args = argparse.Namespace(kv="f16", draft_sampling="greedy", build=Path("/tmp/candidate"),
                                  port=19001, batch=None, ubatch=None, config=Path("/tmp/config"))
        configured = ["/tmp/server", "-ctk", "f16", "--spec-type", "draft-mtp",
                      "--spec-draft-n-max", "2", "--spec-draft-sampling", "probabilistic"]
        with patch("profile_sycl.model_command", side_effect=lambda *args: configured.copy()):
            command = profile_sycl.profile_command(args, 1)
            plain = profile_sycl.profile_command(args, 0)
        self.assertEqual(command.count("--spec-draft-sampling"), 1)
        self.assertEqual(command[command.index("--spec-draft-sampling") + 1], "greedy")
        self.assertNotIn("--spec-draft-sampling", plain)

    def test_mtp_resume_does_not_duplicate_recorded_samples(self):
        with tempfile.TemporaryDirectory(prefix='sycl-resume-test-') as temporary:
            directory = Path(temporary)
            args = argparse.Namespace(build=directory, raw=directory, results=directory / 'results.jsonl',
                                      label='test', kv='f16', alias='swift-1.5-27b-think', horizons='2',
                                      port=19001, fa_backend='auto', repeat=False, tokens=512,
                                      repetitions=2, padding_modules=128, draft_sampling=None,
                                      workloads='coding')
            prefix = 'test-mtp-f16-h2-b4096-ub1024-swift-1.5-27b-think-taskscoding'
            first = prefix + '-coding-r0'
            args.results.write_text(json.dumps({'sample': first}) + '\n')
            command = ['/tmp/fake-server', '-b', '4096', '-ub', '1024', '--alias', args.alias]
            response = {'timings': {'predicted_per_second': 30.0, 'prompt_n': 5500},
                        'tokens_predicted': 512, 'truncated': False, 'content': 'answer'}
            replies = [{'status': 'ok'}, {'prompt': 'test prompt'}, response, response]
            with patch('profile_sycl.socket.create_connection', side_effect=OSError), \
                 patch('profile_sycl.profile_command', return_value=command), \
                 patch('profile_sycl.subprocess.Popen') as start, \
                 patch('profile_sycl.request', side_effect=replies) as request, \
                 patch('builtins.print'):
                start.return_value.poll.return_value = None
                profile_sycl.mtp(args)
            recorded = [json.loads(line) for line in args.results.read_text().splitlines()]
            self.assertEqual(len(recorded), 2)
            self.assertEqual({row['sample'] for row in recorded}, {first, prefix + '-coding-r1'})
            self.assertEqual(request.call_count, 4)
            start.return_value.terminate.assert_called_once()


class ValidationTests(unittest.TestCase):
    def test_reasoning_off_probe_checks_final_answer_and_trace(self):
        response = dict(choices=[dict(message=dict(content='OFF-OK', reasoning_content=''), finish_reason='stop')])
        with patch('validate_sycl.request', return_value=response) as request:
            result = validate_sycl.reasoning_off_probe(19001, 'swift-test')
        self.assertEqual(request.call_args.args[2]['reasoning_effort'], 'none')
        self.assertTrue(result['correct'])
        self.assertEqual(result['reasoning_chars'], 0)

    def test_template_effort_rejection_is_recorded(self):
        def respond(port, path, body):
            if body['reasoning_effort'] == 'high':
                raise HTTPError('http://localhost', 400, 'Bad effort', {}, BytesIO(b'Use xhigh'))
            return {'prompt': body['reasoning_effort']}
        with patch('validate_sycl.request', side_effect=respond):
            outcomes = validate_sycl.template_efforts(19001)
        self.assertFalse(outcomes['high']['accepted'])
        self.assertTrue(all(outcomes[effort]['accepted'] for effort in ('low', 'medium', 'xhigh', 'none')))

    def test_complete_answer_samples_have_independent_correctness(self):
        def respond(port, path, body):
            expected = next(expected for prompt, expected in validate_sycl.CHAT_CASES.values()
                            if prompt == body['messages'][0]['content'])
            return {'choices': [{'message': {'content': json.dumps(expected),
                                            'reasoning_content': 'short thought'},
                                 'finish_reason': 'stop'}]}
        with patch('validate_sycl.request', side_effect=respond) as request:
            samples = validate_sycl.complete_answers(19001, 'test-model')
        self.assertEqual(request.call_count, 9)
        self.assertEqual(len(samples), 6)
        self.assertTrue(all(sample['correct'] for sample in samples))
        self.assertTrue(all(sample['suite'] == 'toy-json-v2' and len(sample['prompt_sha256']) == 64
                            for sample in samples))
        self.assertEqual({sample['workload'] for sample in samples}, set(validate_sycl.CHAT_CASES))

    def test_developer_role_rejection_is_not_advertised_as_supported(self):
        rejected = HTTPError('http://localhost', 400, 'Bad template role', {}, BytesIO(b'Unsupported role'))
        with patch('validate_sycl.request', side_effect=rejected):
            result = validate_sycl.developer_role(19001, 'test-model')
        self.assertFalse(result['supported'])
        self.assertEqual(result['http_status'], 400)

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


class GGUFTests(unittest.TestCase):
    @staticmethod
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data

    def test_metadata_tensor_inventory_and_skipped_tokens(self):
        encode = self.string
        payload = struct.pack('<4sIQQ', b'GGUF', 3, 1, 3)
        payload += encode('general.architecture') + struct.pack('<I', 8) + encode('qwen35')
        payload += encode('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 1) + encode('token')
        payload += encode('tokenizer.chat_template') + struct.pack('<I', 8) + encode('{{ messages }}')
        payload += encode('blk.64.nextn.eh_proj.weight') + struct.pack('<IQQIQ', 2, 10240, 5120, 2, 0)
        with tempfile.TemporaryDirectory(prefix='gguf-inspection-test-') as temporary:
            path = Path(temporary) / 'small.gguf'
            path.write_bytes(payload)
            result = inspect_gguf.inspect(path)
        self.assertEqual(result['metadata'], {'general.architecture': 'qwen35'})
        self.assertEqual(result['nextn_tensors'][0]['shape'], [10240, 5120])
        self.assertEqual(result['tensor_types'], {2: 1})
        self.assertEqual(result['chat_template'], '{{ messages }}')

    def test_truncated_skipped_field_is_rejected(self):
        payload = struct.pack('<4sIQQ', b'GGUF', 3, 0, 1)
        payload += self.string('tokenizer.ggml.tokens') + struct.pack('<IIQQ', 9, 8, 1, 100)
        with tempfile.TemporaryDirectory(prefix='gguf-inspection-test-') as temporary:
            path = Path(temporary) / 'truncated.gguf'
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                inspect_gguf.inspect(path)


class ClientConfigTests(unittest.TestCase):
    def test_swift_contexts_effort_mapping_and_preserved_defaults(self):
        import yaml
        config_path = profile_sycl.ROOT / 'llama-swap.yaml'
        aliases = list(yaml.safe_load(config_path.read_text())['models'])
        source = (profile_sycl.ROOT / 'bin/llm-swap').read_text()
        configure = source.split("python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory(prefix='swift-client-config-test-') as temporary:
            directory = Path(temporary)
            pi_path, auth_path, opencode_path = [directory / name for name in ('models.json', 'auth.json', 'opencode.json')]
            opencode_path.write_text(json.dumps({'model': 'local-b70/agents-a1', 'theme': 'existing'}))
            env = dict(MODELS_JSON=json.dumps(aliases), BASE_URL='http://127.0.0.1:8080/v1',
                       PROVIDER_ID='local-b70', PROVIDER_NAME='Local B70',
                       LLAMA_SWAP_CONFIG=str(config_path), PI_AGENT_CONFIG=str(pi_path),
                       PI_AGENT_AUTH=str(auth_path), OPENCODE_CONFIG=str(opencode_path))
            with patch.dict(os.environ, env), patch('builtins.print'):
                exec(compile(configure, str(profile_sycl.ROOT / 'bin/llm-swap'), 'exec'), {})
            pi_models = {model['id']: model for model in json.loads(pi_path.read_text())['providers']['local-b70']['models']}
            opencode = json.loads(opencode_path.read_text())
        self.assertEqual(set(pi_models), set(aliases))
        self.assertEqual(opencode['model'], 'local-b70/agents-a1')
        self.assertEqual(opencode['theme'], 'existing')
        for alias in ('swift-1.5-27b', 'swift-1.5-27b-mtp', 'swift-1.5-27b-think'):
            context = 131072 if alias.endswith('-think') else 262144
            self.assertEqual(pi_models[alias]['contextWindow'], context)
            self.assertEqual(pi_models[alias]['thinkingLevelMap']['high'], 'xhigh')
            model = opencode['provider']['local-b70']['models'][alias]
            self.assertEqual(model['limit']['context'], context)
            self.assertEqual(model['variants']['high']['reasoningEffort'], 'xhigh')
        self.assertNotIn('thinkingLevelMap', pi_models['qwen3.8-27b-think'])


if __name__ == "__main__":
    unittest.main()
