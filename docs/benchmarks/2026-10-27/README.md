# Intel SYCL update, tuning, and Flint integration — 2026-10-27

## Summary

Updated the clean llama.cpp checkout from `18b5f8b18` (build 2546) to `8ca4060b7` (build 3014, `0.7.0-dev`) and built a separate SYCL release directory. The build uses IntelLLVM 2026.2.0, oneMKL 2026.2, oneDNN 2026.1, Level Zero direct allocation, oneDNN, FP16 kernels, host-memory fallback, and compile-time SYCL graph support. Runtime graphs were explicitly disabled in the profiling runner unless noted.

The most material new Intel optimization is upstream [PR #29864](https://github.com/ggml-org/llama.cpp/pull/29864), merged October 22: multi-column Q4_K_S and Q8_0 XMX ESIMD matrix-vector paths intended for speculative verification and small prompt batches. The upstream B70 report showed 38.7 to 47.6 tokens/s for a Q4_K_S speculative case; ordinary single-token decode was essentially unchanged. Other relevant upstream work includes [Q8 fast dequantization](https://github.com/ggml-org/llama.cpp/commit/0756e83fc), [SYCL graph record/replay support](https://github.com/ggml-org/llama.cpp/commit/7377bada8), improved oneDNN flash-attention scheduling, bounded reorder temporaries, and Q4/Q8 reorder corrections. See the upstream [SYCL backend guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md).

Selected backend correctness passed 2,273/2,273 matrix-multiply cases. Swift real-text profiling showed a useful XMX gain with probabilistic MTP. Flint loaded and generated through its embedded Jinja template at 32K context. A 262K Flint load caused OS memory pressure, so the registered profile remains at 32K; the full model context is not validated on this machine.

## Build and host

- GPU: Intel Arc Pro B70, device `0xe223`, Level Zero reported 1.14.37020.
- Host RAM: 32 GB installed, 8 GiB swap. The 40B Q4_K_M Flint file is 23 GB on disk.
- oneAPI packages selected by `/opt/intel/oneapi/setvars.sh`: IntelLLVM 2026.2.0, oneMKL 2026.2, oneDNN 2026.1. The Level Zero version matches the recorded October 7 value; compiler and math libraries changed, so historical October comparisons are not source-only comparisons.
- Source fast-forwarded from `18b5f8b186` to `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`.
- Build directory: `llama.cpp/build-sycl-20261027`; active symlink: `llama.cpp/build`.
- CMake enabled SYCL/Intel, oneDNN, F16, graph support, Level Zero API, host-memory fallback, server, examples, tests, Release. XMX ESIMD was enabled by this upstream build; `GGML_SYCL_XMX_MAX_COLS=80` and GLU max columns 16.
- Built `llama-server`, `llama-bench`, and `test-backend-ops` with four jobs. CMake could not download llama.cpp web UI assets on the restricted network; the server API is functional, but the candidate binary has no embedded web dashboard.

## Paired throughput checks

These old/new measurements used the same host, Intel runtime, model, prompt shapes, runner, and batch parameters during this pass. They compare the saved October 7 executable to the October 27 candidate under today's toolchain. Values are mean tokens/s; large error bars at pp512 make that short prompt result noisy. The full machine-readable runner log is in [`results.jsonl`](results.jsonl).

| Model/cache | Workload | Saved Oct 7 build | Oct 27 build | Change |
|---|---|---:|---:|---:|
| Swift Q4_K_S / q8_0 | pp8192, 4096/1024 | 756.25 | 871.48 | +15.2% |
| Swift Q4_K_S / q8_0 | pp64000, 4096/1024 | 621.21 | 688.67 | +10.9% |
| Swift Q4_K_S / f16 | pp8192, 4096/1024 | 785.15 | 839.05 | +6.9% |
| Swift Q4_K_S / f16 | pp64000, 4096/1024 | 604.62 | 707.31 | +17.0% |
| Swift Q4_K_S / q8_0 | tg128, 64K KV depth | 8.77 | 8.77 | flat |
| Swift Q4_K_S / f16 | tg128, 64K KV depth | 11.00 | 10.50 | -4.5% |

The production 4096/2048 prompt batch measured 930.80 q8 and 928.26 f16 tokens/s at 8K, versus 871.48 and 839.05 with 4096/1024. Keep 4096/2048 for Swift; this is a batching effect and has not been extended to unrelated model aliases.

Compared with the October 7 report's earlier rates, several new values are lower. That is not a clean source regression comparison: the compiler and oneAPI libraries have changed since then, and only the saved-binary run here controls source under the current environment.

### Swift real-text speculative decoding

Three fixed ~5.5K-token prompts (coding, reasoning, agent), two 512-token outputs each, warm model, Q4_K_S, q8 KV, probabilistic MTP, production batch 4096/2048. Each horizon result is the mean of six output-rate samples:

| Executable / draft horizon | Mean output rate | Relative to saved build h2 |
|---|---:|---:|
| Saved Oct 7 / 2 | 25.21 t/s | baseline |
| Oct 27 / 1 | 25.62 t/s | +1.6% |
| Oct 27 / 2 | 30.80 t/s | +22.2% |
| Oct 27 / 3 | 32.44 t/s | +28.7% |
| Oct 27 / 4 | 29.43 t/s | +16.7% |

Three drafts were fastest overall and on coding/agent prompts; two drafts narrowly led on reasoning. Four drafts regressed. The `swift-1.5-27b-mtp` alias now uses three drafts; the f16 thinking alias remains at two because only q8 KV was swept. Three drafts are a measured throughput choice, not a universal quality or full-agent win. Compared with the historical October 7 h2 figure (33.35 t/s), the new h3 result is 2.7% lower under today's environment.

Graph-capture A/B on the new build with three drafts measured 31.93 t/s overall versus 32.44 with runtime graphs disabled (-1.6%). Keep runtime graphs off. Compile-time graph support remains built in.

## Flint local model

The downloaded file is `~/.lmstudio/models/lmstudio-community/flint-GGUF/flint-Q4_K_M.gguf`, 23 GB. Its GGUF identifies architecture `flint`, 40B model size, 49 blocks, 256 experts with top 6 routed experts, 2048 embedding, 262,144 maximum context, and the Flint embedded chat template. It is text-only; no projector or MTP sidecar is needed. Model metadata recommends temperature 1.0, top-p 0.95, and top-k 20. The template uses Jinja, tool tags, and a separate reasoning channel; the server response confirmed `reasoning_content` and the model returned the requested `FLINT` text.

The 262K context profile encountered OS memory pressure during initial loading. A staged sweep then completed up to a 64K q8 prompt/decode workload without host memory pressure. At 32K decode depth, f16 KV was 23% faster than q8 and consumes about the same KV capacity as q8 at 64K, so the alias now uses 32,768 context, f16 key/value cache, flash attention, `-b 2048 -ub 1024`, one slot, full GPU offload, and reasoning enabled. The model passed a short chat completion. Through llama-swap, a required-tool request returned a correctly parsed `lookup_weather` call and JSON city argument; executing the tool and returning its result was not tested. VRAM peak and sustained sessions have not been measured. The alias remains capped at 32K pending a separate staged context-capacity check.

### Flint throughput and parameter sweep

All rows use build 3014 on SYCL0, full GPU offload, flash attention, oneDNN attention, runtime graphs disabled, and two or three repetitions. Reported values are mean tokens/s ± standard deviation. The prompt sweep fixes model and q8 KV while changing batch/microbatch; the later f16 comparison tests cache type at the same depth and batches.

| Flint workload | KV | Batch / microbatch | Mean tokens/s |
|---|---|---:|---:|
| pp512 | q8_0 | 2048 / 512 | 1,305.5 ± 76.3 |
| pp8192 | q8_0 | 2048 / 512 | 979.4 ± 17.0 |
| pp8192 | q8_0 | 2048 / 1024 | 1,120.5 ± 40.3 |
| pp8192 | q8_0 | 4096 / 512 | 974.7 ± 14.3 |
| pp8192 | q8_0 | 4096 / 1024 | 1,104.6 ± 12.4 |
| pp8192 | f16 | 2048 / 1024 | 1,511.8 ± 0.2 |
| pp64000 | q8_0 | 2048 / 1024 | 640.0 ± 2.3 |
| tg128 at 32K depth | q8_0 | 2048 / 1024 | 52.2 ± 1.0 |
| tg128 at 32K depth | f16 | 2048 / 1024 | 64.0 ± 1.1 |
| tg128 at 64K depth | q8_0 | 2048 / 1024 | 37.9 ± 0.5 |
| tg128 at empty KV | q8_0 | 2048 / 512 | 63.5 ± 7.2 |

At q8 KV, changing 2048/512 to 2048/1024 increased pp8192 by 14.4%. Raising batch to 4096 did not help at either tested microbatch, so Flint keeps batch 2048. At matched 32K depth, f16 improved decode by 22.6%; f16 also improved pp8192 by 34.9% versus q8 at the same 2048/1024 setting. Flint f16 at 32K therefore becomes the active profile. The q8 pp64000 result is close to Swift's q8 688.7 t/s and f16 707.3 t/s at that shape, though Swift used 4096/1024 and a different architecture. At 64K depth Flint q8 measured 37.9 t/s, between Qwen3.8 q8 (12.4) and Agents-A1 f16 (61.6) in the prior October report; quantization and architecture differ, so treat this as context, not a ranking.

The 11 machine-readable per-run measurements and llama-bench metadata are in [`flint/results.jsonl`](flint/results.jsonl). Console logs remain local and are excluded by the repository's `*.log` ignore rule.

Primary model references: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), [local GGUF publisher](https://huggingface.co/lmstudio-community/flint-GGUF), and the [Flint architecture support PR](https://github.com/ggml-org/llama.cpp/pull/30928).

## Deployment and rollback

The active `build` link now points to `build-sycl-20261027`. The previous target is preserved at `build-link-before-20261027`, and its binaries remain intact. The llama-swap service was restarted with the tuned Flint profile; `/health` returned `OK`, `/v1/models` listed `flint`, a chat request passed at 32K f16 KV, and the required-tool request passed through port 8080. The service was restarted after the smoke test to unload Flint and release GPU memory. The model file remains untouched.

Rollback the executable by restoring `llama.cpp/build-link-before-20261027` to `llama.cpp/build`, and remove the Flint entry from `llama-swap.yaml` if reverting the config as well. The prior build directory remains available. Do not delete either build while preserving rollback.
