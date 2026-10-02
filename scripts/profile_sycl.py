#!/usr/bin/env python3
"""Reproducible, resumable SYCL throughput and native-MTP measurements.

Source oneAPI before running. Stop other GPU model services yourself; this
script only starts and terminates its own server on the requested port.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/home/jeff/.lmstudio/models")
MODELS = {
    "qwen": MODEL_ROOT / "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf",
    "agents": MODEL_ROOT / "InternScience/Agents-A1-Q4_K_M-GGUF/Agents-A1-Q4_K_M.gguf",
    "qwen36": MODEL_ROOT / "unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
    "nemotron": MODEL_ROOT / "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_S.gguf",
}
PROMPTS = {
    "coding": "Implement a Python asyncio work queue with bounded capacity, graceful shutdown, cancellation, exponential retry backoff, and exactly-once result delivery within one process. Explain the invariants, then provide executable code and tests for failed workers and shutdown races.",
    "reasoning": "Design a fair elevator scheduler for three elevators and twelve floors. Analyze starvation, direction changes, door timing, overload, and simultaneous calls. Give a concrete state machine, work through a conflicting-call example, and propose tests that detect subtle scheduling failures.",
    "agent": "You are maintaining an inference proxy. A long-context request stalls after a model swap, but a short request succeeds. Logs show a healthy HTTP listener, an idle GPU, and an outstanding tool call. Plan a read-only investigation, distinguish scheduling from inference failures, and specify the evidence required before changing configuration. Include concrete commands and expected observations.",
}


def cases(suite: str) -> list[dict]:
    result = []

    def add(model: str, kv: str, prompt: int, gen: int, depth: int = 0,
            batch: int = 2048, ubatch: int = 512, reps: int = 2) -> None:
        name = f"{model}-{kv}-p{prompt}-n{gen}-d{depth}-b{batch}-ub{ubatch}"
        result.append(dict(name=name, model=model, kv=kv, prompt=prompt,
                           gen=gen, depth=depth, batch=batch, ubatch=ubatch, reps=reps))

    if suite == "comparison":
        for model, kv in [("qwen", "q8_0"), ("qwen", "f16"), ("agents", "f16")]:
            add(model, kv, 512, 128, reps=3)
            for prompt in [8192, 64000]:
                add(model, kv, prompt, 0)
            add(model, kv, 0, 128, depth=64000)
        for kv in ["q8_0", "f16"]:
            add("qwen", kv, 512, 128, depth=16384, reps=3)
        for model in ["qwen", "agents"]:
            add(model, "f16", 126976, 0)
        add("qwen36", "q8_0", 512, 128, reps=3)
        add("nemotron", "f16", 512, 128, reps=3)
    elif suite == "tuning":
        for model, kv in [("qwen", "q8_0"), ("qwen", "f16"), ("agents", "f16")]:
            for prompt in [8192, 64000]:
                add(model, kv, prompt, 0, batch=4096, ubatch=1024)
    return result


def record(args: argparse.Namespace, value: dict) -> None:
    value.update(label=args.label, binary_dir=str(args.build / "bin"),
                 recorded_at=datetime.now(timezone.utc).isoformat())
    with args.results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def runtime_env(build: Path, fa_backend: str = "auto") -> dict[str, str]:
    env = os.environ.copy()
    previous_libraries = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(build / "bin") + (":" + previous_libraries if previous_libraries else "")
    env["GGML_SYCL_ENABLE_GRAPH"] = "0"
    env["GGML_SYCL_FA_ONEDNN"] = "0" if fa_backend == "mkl" else "1"
    return env


def bench(args: argparse.Namespace) -> None:
    for case in cases(args.suite):
        case = dict(case, fa_backend=args.fa_backend)
        if args.batch is not None:
            case["batch"] = args.batch
        if args.ubatch is not None:
            case["ubatch"] = args.ubatch
        case["name"] = (f"{case['model']}-{case['kv']}-p{case['prompt']}-n{case['gen']}"
                        f"-d{case['depth']}-b{case['batch']}-ub{case['ubatch']}")
        if args.cases and not any(part in case["name"] for part in args.cases.split(",")):
            continue
        key = f"{args.label}-{case['name']}"
        if args.fa_backend != "auto":
            key += f"-{args.fa_backend}"
        saved = args.raw / f"{key}.json"
        if saved.exists() and not args.repeat:
            print(f"SKIP {key}: already recorded", flush=True)
            continue
        command = [str(args.build / "bin/llama-bench"), "-m", str(MODELS[case["model"]]),
                   "-dev", "SYCL0", "-ngl", "999", "-fa", "on", "-ctk", case["kv"],
                   "-ctv", case["kv"], "-p", str(case["prompt"]), "-n", str(case["gen"]),
                   "-d", str(case["depth"]), "-b", str(case["batch"]), "-ub", str(case["ubatch"]),
                   "-r", str(case["reps"]), "-o", "json", "--progress"]
        print(f"START {key}: {shlex.join(command)}", flush=True)
        started = time.monotonic()
        with (args.raw / f"{key}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True,
                                     env=runtime_env(args.build, args.fa_backend), timeout=1800)
        if process.returncode:
            record(args, dict(kind="bench-error", case=case, returncode=process.returncode))
            raise RuntimeError(f"{key} failed; inspect its log")
        measured = json.loads(process.stdout)
        saved.write_text(process.stdout, encoding="utf-8")
        for row in measured:
            record(args, dict(kind="bench", case=case, measurement=row,
                              wall_seconds=time.monotonic() - started))
            print(f"RESULT {key}: p{row['n_prompt']} n{row['n_gen']} "
                  f"{row['avg_ts']:.2f} +/- {row['stddev_ts']:.2f} t/s", flush=True)


def request(port: int, path: str, body: dict | None = None, timeout: int = 600) -> dict:
    opener = build_opener(ProxyHandler({}))
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(f"http://127.0.0.1:{port}{path}", data=data,
                  headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def model_command(config_path: Path, build: Path, alias: str, port: int) -> list[str]:
    """Select real production arguments without evaluating the shell wrapper."""
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shell = shlex.split(config["models"][alias]["cmd"])[2]
    command = shlex.split(shell.split("&& exec ", 1)[1])
    command[0] = str(build / "bin/llama-server")
    command[command.index("--port") + 1] = str(port)
    return command


def profile_command(args: argparse.Namespace, horizon: int) -> list[str]:
    # Include the production projector, context allocation, and sampling defaults.
    alias = "qwen3.8-27b-think" if args.kv == "f16" else "qwen3.8-27b-mtp"
    command = model_command(args.config, args.build, alias, args.port)
    command[command.index("--spec-draft-n-max") + 1] = str(horizon)
    if args.batch is not None:
        command += ["-b", str(args.batch)]
    if args.ubatch is not None:
        command += ["-ub", str(args.ubatch)]
    if horizon == 0:
        index = command.index("--spec-type")
        del command[index:index + 2]
        index = command.index("--spec-draft-n-max")
        del command[index:index + 2]
    return command


def mtp(args: argparse.Namespace) -> None:
    try:
        connection = socket.create_connection(("127.0.0.1", args.port), timeout=1)
    except OSError:
        pass
    else:
        connection.close()
        raise RuntimeError(f"Port {args.port} is occupied; choose another --port")
    # Padding is deterministic repository-like text, not a repeating numeric sequence.
    context = "\n".join(
        f"module_{i:03d}.py: async def worker_{i}(queue, stop): "
        "item = await queue.get(); await process(item); queue.task_done(); "
        "# cancellation must preserve queue accounting and release owned resources"
        for i in range(128)
    )
    for horizon in map(int, args.horizons.split(",")):
        command = profile_command(args, horizon)
        batch, ubatch = 2048, 512
        for index, word in enumerate(command[:-1]):
            if word in ("-b", "--batch-size"):
                batch = int(command[index + 1])
            elif word in ("-ub", "--ubatch-size"):
                ubatch = int(command[index + 1])
        key = f"{args.label}-mtp-{args.kv}-h{horizon}-b{batch}-ub{ubatch}"
        if args.fa_backend != "auto":
            key += f"-{args.fa_backend}"
        if (args.raw / f"{key}.done").exists() and not args.repeat:
            print(f"SKIP {key}: already recorded", flush=True)
            continue
        print(f"START {key}: {shlex.join(command)}", flush=True)
        with (args.raw / f"{key}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                                       env=runtime_env(args.build, args.fa_backend))
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{key} server exited during startup")
                    try:
                        if request(args.port, "/health", timeout=2).get("status") == "ok":
                            break
                    except (HTTPError, URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{key} server startup timed out")
                    time.sleep(1)
                for name, task in PROMPTS.items():
                    messages = [
                        {"role": "system", "content": "You are a careful coding assistant. Think through the problem and provide a concrete, technically correct answer."},
                        {"role": "user", "content": "Repository notes:\n" + context + "\n\nTask:\n" + task},
                    ]
                    prompt = request(args.port, "/apply-template", {"messages": messages})["prompt"]
                    body = dict(prompt=prompt, n_predict=args.tokens, seed=42, temperature=1.0,
                                top_k=20, top_p=0.95, min_p=0, presence_penalty=0,
                                repeat_penalty=1.0, cache_prompt=False, stream=False)
                    # Discard one full request per workload to warm the tested paths.
                    request(args.port, "/completion", body)
                    for rep in range(args.repetitions):
                        started = time.monotonic()
                        response = request(args.port, "/completion", body)
                        elapsed = time.monotonic() - started
                        sample = f"{key}-{name}-r{rep}"
                        (args.raw / f"{sample}.json").write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
                        timings = response["timings"]
                        record(args, dict(kind="mtp", kv=args.kv, horizon=horizon, workload=name,
                                          repetition=rep, batch=batch, ubatch=ubatch,
                                          fa_backend=args.fa_backend,
                                          command=command,
                                          timings=timings, wall_seconds=elapsed,
                                          tokens_predicted=response.get("tokens_predicted"),
                                          truncated=response.get("truncated"),
                                          stopped_limit=response.get("stopped_limit", response.get("stop_type") == "limit"),
                                          stop_type=response.get("stop_type"),
                                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                          content_sha256=hashlib.sha256(response.get("content", "").encode()).hexdigest()))
                        print(f"RESULT {sample}: {timings['predicted_per_second']:.2f} t/s, "
                              f"{timings.get('draft_n_accepted', '?')}/{timings.get('draft_n', '?')} drafts, "
                              f"{timings['prompt_n']} prompt tokens, {elapsed:.2f}s wall", flush=True)
                (args.raw / f"{key}.done").touch()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["bench", "mtp"])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "llama-swap.yaml",
                        help="Production profile source for MTP tests; use a saved config to reproduce old runs")
    parser.add_argument("--suite", choices=["comparison", "tuning"], default="comparison")
    parser.add_argument("--cases", help="Comma-separated case-name substrings")
    parser.add_argument("--kv", choices=["f16", "q8_0"], default="f16")
    parser.add_argument("--fa-backend", choices=["auto", "mkl"], default="auto",
                        help="mkl bypasses oneDNN SDPA only; oneDNN matrix multiplication stays enabled")
    parser.add_argument("--horizons", default="1,2,3,4")
    parser.add_argument("--port", type=int, default=19001)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--batch", type=int, help="Override batch size for bench or MTP tests")
    parser.add_argument("--ubatch", type=int, help="Override microbatch size for bench or MTP tests")
    parser.add_argument("--repeat", action="store_true", help="Append fresh measurements even if raw results exist")
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "bench":
        bench(args)
    else:
        mtp(args)


if __name__ == "__main__":
    main()
