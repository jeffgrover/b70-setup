#!/usr/bin/env python3
"""Tool round-trip and optional near-limit KV stress checks for real profiles.

Source oneAPI and pause other GPU workloads before isolated checks. --proxy
tests the running llama-swap instead, without starting or stopping a server.
"""

import argparse
import json
from pathlib import Path
import random
import shlex
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError

from profile_sycl import ROOT, model_command, record, request, runtime_env


def tool_round_trip(port: int, alias: str) -> dict:
    tool = dict(type="function", function=dict(name="read_status", description="Read the test status.",
                parameters=dict(type="object", properties={}, additionalProperties=False)))
    messages = [dict(role="user", content="Call read_status. Then return only the status value from its result.")]
    body = dict(model=alias, messages=messages, tools=[tool],
                tool_choice=dict(type="function", function=dict(name="read_status")),
                max_tokens=1024, temperature=0, seed=42, reasoning_effort="low", stream=False)
    first = request(port, "/v1/chat/completions", body)
    message = first["choices"][0]["message"]
    calls = message.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "read_status":
        raise AssertionError(f"{alias}: expected one parsed read_status tool call")
    if json.loads(calls[0]["function"]["arguments"]) != {}:
        raise AssertionError(f"{alias}: unexpected tool arguments")
    messages += [dict(role="assistant", content=message.get("content"), tool_calls=calls),
                 dict(role="tool", tool_call_id=calls[0]["id"], content='{"status":"READY-2415"}')]
    body.update(messages=messages, tool_choice="none")
    second = request(port, "/v1/chat/completions", body)
    content = (second["choices"][0]["message"].get("content") or "").strip()
    if content != "READY-2415":
        raise AssertionError(f"{alias}: wrong final status: {content!r}")
    return dict(first=first, second=second)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "llama-swap.yaml")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--label", default="candidate-validation")
    parser.add_argument("--depth", type=int, help="Synthetic prompt token count for a single memory stress request")
    parser.add_argument("--port", type=int)
    parser.add_argument("--fa-backend", choices=["auto", "mkl"], default="auto")
    parser.add_argument("--proxy", action="store_true", help="Only test tool use through the existing llama-swap")
    args = parser.parse_args()
    if args.proxy and args.depth is not None:
        parser.error("--proxy cannot be combined with a direct native-completion stress request")
    args.port = args.port or (8080 if args.proxy else 19001)
    args.build = args.build.resolve()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    command = model_command(args.config, args.build, args.alias, args.port)
    # Diagnostic logging exposes offload/allocation details; it does not change inference settings.
    command += ["-lv", "4"]
    context = int(command[command.index("-c") + 1])
    if args.depth is not None:
        if not 0 < args.depth < context - 32:
            parser.error("--depth must leave at least 32 tokens within the configured context")
    key = f"{args.label}-{args.alias}-{'proxy' if args.proxy else 'isolated'}-d{args.depth or 0}"
    process = None
    started = time.monotonic()
    with (args.raw / f"{key}.log").open("w", encoding="utf-8") as log:
        try:
            if not args.proxy:
                try:
                    connection = socket.create_connection(("127.0.0.1", args.port), timeout=1)
                except OSError:
                    pass
                else:
                    connection.close()
                    raise RuntimeError(f"Port {args.port} is occupied")
                print(f"START {key}: {shlex.join(command)}", flush=True)
                process = subprocess.Popen(command, stdout=log, stderr=log,
                                           env=runtime_env(args.build, args.fa_backend))
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited during startup")
                    try:
                        if request(args.port, "/health", timeout=2).get("status") == "ok":
                            break
                    except (HTTPError, URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(1)
            props = None
            if not args.proxy:
                props = request(args.port, "/props")
                (args.raw / f"{key}-props.json").write_text(json.dumps(props, indent=2) + "\n")
                if props["default_generation_settings"]["n_ctx"] != context or props["total_slots"] != 1:
                    raise AssertionError("Server reduced context capacity or changed the slot count")
                if "--mmproj" in command and not props["modalities"]["vision"]:
                    raise AssertionError("Configured vision projector is not enabled")
            tools = tool_round_trip(args.port, args.alias)
            (args.raw / f"{key}-tools.json").write_text(json.dumps(tools, indent=2) + "\n")
            stress = None
            if args.depth is not None:
                print(f"STRESS {key}: {args.depth} synthetic input tokens", flush=True)
                rng = random.Random(42)
                prompt = [rng.randrange(1000, 20000) for _ in range(args.depth)]
                stress = request(args.port, "/completion", dict(prompt=prompt, n_predict=32,
                                 ignore_eos=True, seed=42, temperature=0, cache_prompt=False,
                                 stream=False), timeout=1800)
                (args.raw / f"{key}-stress.json").write_text(json.dumps(stress, indent=2) + "\n")
                if stress.get("truncated") or stress.get("tokens_predicted") != 32:
                    raise AssertionError("Stress request truncated input or did not generate 32 tokens")
                if stress["timings"]["prompt_n"] < args.depth:
                    raise AssertionError("Stress request reused cached input instead of filling KV")
            record(args, dict(kind="validation", alias=args.alias, proxy=args.proxy,
                              depth=args.depth, command=None if args.proxy else command,
                              fa_backend=args.fa_backend, tool_round_trip="READY-2415",
                              actual_context=None if props is None else props["default_generation_settings"]["n_ctx"],
                              modalities=None if props is None else props["modalities"],
                              stress_timings=None if stress is None else stress["timings"],
                              wall_seconds=time.monotonic() - started))
            print(f"PASS {key}: tool round trip" + (" and KV stress" if stress else ""), flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
