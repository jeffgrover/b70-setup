# B70 SYCL update and profiling, 2026-10-01–02

## Environment and scope

- Baseline: `3057bb66c`, release tag `b10931`, local binary build 1999.
- Candidate: `5fc4f3c8c`, `b11337-10-g5fc4f3c8c`, local binary build 2415. This is 416 commits newer than the baseline.
- GPU: Intel Arc Pro B70, `SYCL0`, device ID `0xe223`, Level Zero driver `1.14.37020`, approximately 32 GiB VRAM.
- CPU: AMD Ryzen 9 7940HS. Kernel: `7.0.0-38-generic`.
- Compiler: IntelLLVM 2026.1.1; oneMKL 2026.1; oneDNN 2026.0 (`libdnnl.so.3.11`).
- Both builds enable SYCL, FP16 kernels, oneDNN, direct Level Zero allocation, native CPU tuning, host-memory fallback, and compiled SYCL graphs. Runtime graph capture stays disabled. Device architecture remains unset for the large-buffer JIT path.
- The original build was copied to `llama.cpp/build-b10931-20260912/` before updating the clean source checkout. The candidate was compiled separately in `llama.cpp/build-sycl-20261001/`.

The baseline's binaries contain an absolute runtime library path pointing at `build/bin`. The profiling script prepends the selected build's `bin` directory to `LD_LIBRARY_PATH`, so the saved baseline cannot accidentally use candidate libraries after deployment.

## Upstream changes worth testing

- [MKL attention softmax coalescing](https://github.com/ggml-org/llama.cpp/pull/28918): upstream B70 measurements on Qwen3.8 UD-Q4_K_XL report roughly 7%, 33%, and 49% faster prompt processing at 9K, 64K, and 127K tokens. These are not measurements of this repository's Q4_K_S model. The default oneDNN SDPA path can take precedence, so an MKL-only attention comparison is needed.
- [SSM convolution and SiLU fusion](https://github.com/ggml-org/llama.cpp/pull/28929) and [normalization/mixed-quant fusion](https://github.com/ggml-org/llama.cpp/pull/28931): relevant to hybrid Qwen architectures. The added mixed-quant gate/up path covers Q5_K and IQ4_XS combinations; same-type Q4_K fusion already existed in the baseline.
- [B70 large-allocation workaround](https://github.com/ggml-org/llama.cpp/pull/28953): caps individual allocation sizes, not total usable VRAM.
- [oneDNN scratchpad allocation ordering](https://github.com/ggml-org/llama.cpp/pull/28704): fixes crashes with the VMM allocator.
- [GPU radix top-k](https://github.com/ggml-org/llama.cpp/pull/28670) and [sparse attention](https://github.com/ggml-org/llama.cpp/pull/28796): useful for additional architectures and GPU operators. Sparse attention remains disabled; it is not a generic switch to make dense Qwen attention sparse. This does not justify changing sampling `--top-k`.
- [Speculative-decoding batch ordering](https://github.com/ggml-org/llama.cpp/pull/29019): correctness work; the reported DFlash concurrency gains are not directly applicable to the single-slot native-MTP profiles here.

The [oneAPI 2026.1 CI update](https://github.com/ggml-org/llama.cpp/pull/29273) reports a large improvement over 2025.3, but this host already used 2026.1.1. The [GLM MLA prefill optimization](https://github.com/ggml-org/llama.cpp/pull/29171) was still unmerged when checked; it was not cherry-picked, and the existing q8 GLM profile does not match its f16-only gate.

## Method

The raw machine-readable measurements are in [results.jsonl](results.jsonl). Complete native outputs and server logs remain local under the ignored `llama.cpp/build-profile-20261001/` directory. Benchmarks run sequentially with llama-swap stopped and no compilation competing for CPU dispatch time. The service is restarted through a shell exit trap after each stage.

Direct `llama-bench` measurements use the existing GGUFs, full GPU offload, flash attention, eight default CPU threads, and unchanged polling. Short tests have three measured repetitions; long tests have two. Prompt tests include a full discarded warmup; context-depth tests preload and restore the recorded KV depth. These tests exclude tokenization and sampling, and use synthetic tokens rather than real conversation text.

The native-MTP tests use the actual Qwen profile commands from `llama-swap.yaml`, including the vision projector, context allocation, reasoning configuration, and production sampling settings. The script applies the model's chat template to coding, reasoning, and diagnostic-planning tasks with deterministic repository-like padding. Requests disable prompt reuse, use seed 42, and cap generation at 512 tokens. One full request per workload is discarded as warmup, then two repetitions are recorded. This measures bounded throughput and acceptance, not complete-task quality or an entire agent session. Actual prompt lengths and output hashes are recorded.

The old-build real-text baseline uses the original f16 thinking profile, horizon 3, at 2048/512. The candidate sweeps use 4096/1024 and include horizon 0 (no speculation) as a control. Thus the old-to-tuned comparison measures the combined deployed configuration; source-only differences are isolated by the direct paired benchmarks. Prompt hashes match across these server tests: 5,522 coding tokens, 5,525 reasoning tokens, and 5,542 diagnostic-planning tokens. A fixed seed does not guarantee identical GPU output or draft acceptance; report cross-sample variability, not just the best request.

Attention `auto` uses the backend's ordinary dispatch, including oneDNN SDPA. Attention `mkl` sets `GGML_SYCL_FA_ONEDNN=0`; it leaves oneDNN matrix multiplication enabled. Both keep runtime graphs off.

## Paired results with defaults unchanged

All figures below are tokens/second, baseline → candidate, at `-b 2048 -ub 512`. These compare the two builds on the same host during this pass, not against an unmatched historical run.

| Model / KV | pp512 | tg128, empty KV | pp8192 | pp64000 | tg128 at 64000 depth |
|---|---:|---:|---:|---:|---:|
| Qwen3.8 / q8_0 | 833.33 → 834.56 | 26.58 → 26.67 | 853.18 → 864.72 | 721.86 → 728.40 | 12.41 → 12.42 |
| Qwen3.8 / f16 | 868.60 → 879.78 | 26.84 → 26.92 | 858.75 → 869.75 | 726.36 → 732.84 | 14.93 → 14.98 |
| Agents-A1 / f16 | 1127.92 → 1125.00 | 99.29 → 100.15 | 1053.97 → 1060.81 | 923.56 → 930.07 | 61.44 → 61.62 |

Qwen's first 16K-depth decode round measured `20.45 → 20.09` with q8_0 and `22.09 → 22.23` with f16. The apparent q8 regression did not reproduce: a second old/new round measured `20.48 → 20.56`. At 126,976 prompt tokens, Qwen f16 measured `617.26 → 622.08` (+0.8%); Agents-A1 measured `819.55 → 825.35` (+0.7%).

Default-setting prompt gains are generally about 0–1.4%; decoding is mostly unchanged. The candidate still decodes roughly 20% faster with f16 than q8 at 64K depth. Keep the 128K f16 / 256K q8 context trade-off; these measurements do not justify reducing cache precision, increasing the context limit, or changing the model's sampling recipe.

### Historical short controls and repeatability

| Model | September 12 pp512 / tg128 | Old build rerun | Candidate | Old / candidate check round |
|---|---:|---:|---:|---:|
| Qwen3.8 q8_0 | 838.80 / 26.80 | 833.33 / 26.58 | 834.56 / 26.67 | 16K-depth decode: 20.48 / 20.56 |
| Agents-A1 f16 | 1140.24 / 100.97 | 1127.92 / 99.29 | 1125.00 / 100.15 | — |
| Qwen3.6 q8_0 | 1157.01 / 78.85 | 1145.59 / 77.88 | 1161.28 / 78.27 | — |
| Nemotron f16 | 1061.92 / 58.64 | 986.23 / 57.34 | 923.16 / 57.68 | 929.88 / 57.58 → 1027.76 / 58.06 |

The Nemotron pp512 gap reversed in the second round; its source-only prefill change is not established. Cross-round variation is larger than some within-round standard deviations. Small differences should not be treated as guaranteed speedups or regressions. The old and new builds use the same installed compiler/runtime, but September's measurements were made on an earlier host/kernel state.

## Batch and attention-path tuning

At 8,192 prompt tokens, the candidate produced:

| Model / KV | 2048/512, auto | 4096/1024, auto | 4096/1024, MKL attention |
|---|---:|---:|---:|
| Qwen3.8 / q8_0 | 864.72 | 1068.64 | 1026.74 |
| Qwen3.8 / f16 | 869.75 | 1071.18 | 1026.60 |
| Agents-A1 / f16 | 1060.81 | 1423.96 | 1369.24 |

Larger batching improves these prompt tests by 23–34%. Disabling oneDNN SDPA is about 4% slower at the same larger batch size. These are configuration gains, not evidence that the source update alone delivers the upstream MKL percentages.

The saved old build also benefits from larger batching at 8K: `1053.79` (Qwen q8), `1051.71` (Qwen f16), and `1278.69` (Agents-A1). This separates batching gains from source/configuration interactions; the latter should not be inferred from an old small-batch versus new large-batch comparison.

A second matched 8K Agents-A1 round at 4096/1024 measured `1407.34 → 1418.12` (old → candidate), versus `1278.69 → 1423.96` initially. The apparent 11% source-only gain did not reproduce. Larger batching remains a useful configuration change, but cross-round variation rules out assigning that initial difference to one particular upstream kernel.

At 64K, the candidate's comparison is:

| Model / KV | 2048/512, auto | 4096/1024, auto | 4096/1024, MKL attention | Larger-batch auto gain |
|---|---:|---:|---:|---:|
| Qwen3.8 / q8_0 | 728.40 | 862.32 | 721.95 | +18.4% |
| Qwen3.8 / f16 | 732.84 | 863.52 | 721.71 | +17.8% |
| Agents-A1 / f16 | 930.07 | 1188.52 | 1067.12 | +27.8% |

Keep normal attention dispatch: MKL-only is 10–16% slower at 64K too. No oneDNN disable flag or guessed long-context crossover threshold is warranted by these results.

## Real-text native-MTP sweep

Each row has six measured 512-token outputs (two per workload), with 4096/1024 batching and normal attention dispatch. Generation rate is the mean of the native API's reported rate; acceptance is total accepted / total drafted tokens. Request time includes prompt processing and generation on an already-loaded server, not model-swap startup.

| KV | Maximum drafts | Mean generation t/s | Sample range t/s | Draft acceptance | Mean request seconds |
|---|---:|---:|---:|---:|---:|
| f16 | 0, non-speculative | 24.66 | 24.55–24.78 | — | 26.27 |
| f16 | 1 | 30.29 | 29.12–31.49 | 64.6% | 22.93 |
| f16 | 2 | 30.96 | 28.10–33.97 | 49.7% | 22.60 |
| f16 | 3 | 27.84 | 25.32–29.58 | 37.4% | 24.48 |
| f16 | 4 | 23.18 | 21.21–24.81 | 30.7% | 28.15 |
| q8_0 | 0, non-speculative | 23.94 | 23.93–23.95 | — | 26.90 |
| q8_0 | 1 | 29.05 | 28.74–29.73 | 62.1% | 23.65 |
| q8_0 | 2 | 28.78 | 26.63–30.99 | 45.4% | 23.87 |
| q8_0 | 3 | 27.94 | 26.10–29.83 | 39.1% | 24.38 |
| q8_0 | 4 | 22.82 | 21.60–24.46 | 30.6% | 28.50 |

The old f16/three-draft/2048–512 profile averaged `27.59 t/s` (range `25.11–30.26`, acceptance `37.4%`) and `26.02 s` per request. The selected new f16/two-draft/4096–1024 combination averaged `30.96 t/s` and `22.60 s`: about 12% higher generation rate and 13% lower warm request time. This is a combined configuration result, not a source-only speedup.

Select two drafts for the 128K f16 thinking profile: it beats three in all three workload means and has the best overall mean. One draft is close and better on the reasoning workload; this is not a universal optimum. Select one for the 256K q8 MTP profile: one/two are nearly tied overall, but one is more consistent and better on reasoning. Four is worse than non-speculative decoding here. Acceptance percentage alone is not the objective: f16/one has the highest acceptance, but f16/two is slightly faster overall.

September's three-draft `47–50 t/s` figures used a short repetitive numeric control with much higher acceptance. They are not comparable to these real-text `23–31 t/s` averages. All 60 candidate and six baseline measured real-text outputs reached the 512-token cap with no input truncation. These are predominantly bounded reasoning-throughput tests, not complete code-generation correctness tests, full agent runs, or proof of the best MTP horizon at 64K–256K occupied context.

## Configuration decisions

- Add `-b 4096 -ub 1024` only to `qwen3.8-27b`, `qwen3.8-27b-mtp`, `qwen3.8-27b-think`, and `agents-a1`.
- Change the f16 thinking profile and its standalone launcher from three MTP drafts to two; change the q8 MTP profile from three to one.
- Keep model files, projectors, all eleven aliases, context limits, cache types, sampling, reasoning budgets, and single-slot operation unchanged. Other models keep their existing batching and experimental status.
- Keep ordinary oneDNN attention dispatch, graphs disabled by default, and polling unchanged. The compiler was already 2026.1.1; no driver or oneAPI package upgrade was needed.
- Correct the current config comment that incorrectly presented the older 43% f16 advantage at 16K as a September/current measurement. The paired October advantage is about 8–11% at 16K and 20% at 64K.

## Validation and kernel profiling

The candidate passed 2,037/2,037 selected `ADD`, `L2_NORM`, `RMS_NORM`, `RMS_NORM_SCALE`, `MUL_MAT`, `GLU`, `SSM_CONV`, `SSM_CONV_BIAS_SILU`, and `TOP_K` cases. Production-relevant head-width-256, unpermuted f16/q8 attention cases with no sparse-KV hint passed 117/117 under normal dispatch and another 117/117 with oneDNN SDPA disabled to exercise the MKL path.

The wider attention subset passed 122/123. The failure was `hsk=256,hsv=256,nh=1,nr23=[12,2],kv=8192,nb=67,type_K=f16,type_V=f16,n_kv_max=512`, with normalized error about 1.01. This is a two-sequence sparse-mask case, unlike the single-slot dense Qwen/Agents configurations. Increasing the prompt's token batch does not increase the number of simultaneous sequences. The old executable does not enumerate that exact case: filtering it produced 0/0 tests, which is not evidence that the old build passes it. No claim is made that this failure is pre-existing or fixed. Keep sparse attention disabled and validate again before expanding to such layouts. Complete logs are retained locally.

Standalone convolution profiling confirms the new tiled kernel improves prefill work substantially, while the one-token case is effectively unchanged:

| SSM_CONV input shape | Baseline microseconds | Candidate microseconds | Speedup |
|---|---:|---:|---:|
| `[515,3328,1,1]` | 53.11 | 19.43 | 2.73x |
| `[937,8192,1,1]` | 279.94 | 131.21 | 2.13x |
| `[4,3328,1,1]` | 3.02 | 3.00 | 1.01x |

These are operation timings, not whole-model speedups.

### Production-profile memory and API checks

The proposed live commands passed isolated `read_status` tool-call / simulated tool-result / exact `READY-2415` final-answer round trips, followed by a single synthetic near-limit cache-fill request generating 32 tokens. `/props` confirmed the requested context and one slot; both Qwen projectors reported vision enabled. Diagnostic verbosity exposed the actual batching and full GPU offload (66/66 Qwen layers, 41/41 Agents layers).

| Profile | Reported context | Stress input tokens | Target KV | Separate MTP draft KV | Device free before cleanup |
|---|---:|---:|---:|---:|---:|
| Qwen f16, two drafts | 131072 | 126976 | 8192 MiB | 512 MiB f16 | 5857 MiB |
| Qwen q8, one draft | 262144 | 253952 | 8704 MiB | 1024 MiB f16 | 3504 MiB |
| Agents-A1 f16 | 262144 | 253952 | 5120 MiB | — | 6327 MiB |

All three completed without input truncation, context reduction, or allocation-fallback warnings. These are text memory/compatibility stress checks, not repeated throughput benchmarks or an image/video stress test. End-of-run free memory comes from the driver's reported 32656 MiB total and includes background/driver usage; it is not a measured minimum over the entire run.

The q8 target cache's block-scale overhead matters: it is 8.5 GiB at 256K, versus 8 GiB for f16 at 128K. The native MTP draft cache remains default-f16 independently of the target cache flags, adding 1 GiB or 0.5 GiB respectively. Do not assume identical total KV footprints, or that the B70 single-allocation workaround creates more total VRAM. Recurrent state, projector weights, compute buffers, and attention temporaries also consume memory.

After deployment, complete tool round trips passed through `127.0.0.1:8080` for `qwen3.8-27b`, `qwen3.8-27b-mtp`, `qwen3.8-27b-think`, `agents-a1`, `qwen3.6-35b-a3b`, `nemotron-3.5-lightning`, and `gemma-4-e4b`. The experimental Nemotron MTP profile and the remaining models were not newly profiled or retuned. llama-swap remains active; the existing 600-second idle unload still applies.

## Deployment and rollback

`llama.cpp/build` now points to `build-sycl-20261001`. Native version output is `0.5.0-dev`, local build 2415, commit `5fc4f3c8c`, IntelLLVM 2026.1.1. Library resolution was checked: the llama/ggml libraries come from the candidate directory, with the existing oneAPI libraries under `/opt/intel/oneapi`. llama-swap was restarted and `/health` returned `OK`. The documentation, measurement dataset, reproduction helpers, and tuned configurations are versioned in this repository; compiled binaries, model weights, and full native logs remain local and ignored.

A final proxy round trip reloaded the thinking profile after its normal idle unload. The live process had `-c 131072 -b 4096 -ub 1024 --spec-draft-n-max 2`; every mapped llama/ggml library came from the candidate, with no inherited `GGML_SYCL_*` overrides. It was left warm for handoff, subject to the usual idle TTL. The saved dataset contains 75 direct throughput measurements, 66 real-text measurements, and 11 validation records (three isolated near-limit checks and eight proxy round trips, including that final repeat).

Retained local recovery artifacts:

- `llama.cpp/build-original-20261001/`: the original build directory, moved intact before installing the stable link.
- `llama.cpp/build-b10931-20260912/`: the independent old-build benchmark snapshot; prepend its `bin` to `LD_LIBRARY_PATH` when running it.
- `llama.cpp/build-profile-20261001/llama-swap.before.yaml` and `start_server.before.sh`: pre-tuning configs.

To roll back, first preserve any subsequent config edits. These commands intentionally restore the pre-update configs and binaries, but leave the source checkout at the new commit. The link is moved rather than deleted, and the exact current target is checked before changing it:

```bash
cd /home/jeff/Code/intel
test -L llama.cpp/build
test "$(readlink llama.cpp/build)" = build-sycl-20261001
test -d llama.cpp/build-original-20261001
test ! -e llama.cpp/build-updated-link-20261002
systemctl --user stop llama-swap.service
mv llama.cpp/build llama.cpp/build-updated-link-20261002
mv llama.cpp/build-original-20261001 llama.cpp/build
cp llama.cpp/build-profile-20261001/llama-swap.before.yaml llama-swap.yaml
cp llama.cpp/build-profile-20261001/start_server.before.sh start_server.sh
systemctl --user start llama-swap.service
```

For a future update, build in a fresh named directory and switch only after validation. Do not reconfigure or overwrite the active `build` symlink target during a running model request.

## Reproduction

From the repository root, with other GPU model workloads paused:

```bash
source /opt/intel/oneapi/setvars.sh > /dev/null
python3 scripts/profile_sycl.py bench \
  --build llama.cpp/build-b10931-20260912 --label baseline \
  --results docs/benchmarks/2026-10-01/results.jsonl \
  --raw llama.cpp/build-profile-20261001
python3 scripts/profile_sycl.py bench \
  --build llama.cpp/build-sycl-20261001 --label candidate \
  --results docs/benchmarks/2026-10-01/results.jsonl \
  --raw llama.cpp/build-profile-20261001
python3 scripts/profile_sycl.py bench \
  --build llama.cpp/build-sycl-20261001 --label candidate --suite tuning \
  --results docs/benchmarks/2026-10-01/results.jsonl \
  --raw llama.cpp/build-profile-20261001
python3 scripts/profile_sycl.py mtp \
  --build llama.cpp/build-sycl-20261001 --label candidate --kv f16 \
  --results docs/benchmarks/2026-10-01/results.jsonl \
  --raw llama.cpp/build-profile-20261001 --horizons 0,1,2,3,4 \
  --batch 4096 --ubatch 1024
python3 scripts/validate_sycl.py \
  --build llama.cpp/build-sycl-20261001 --alias qwen3.8-27b-think \
  --depth 126976 --results docs/benchmarks/2026-10-01/results.jsonl \
  --raw llama.cpp/build-profile-20261001
python3 scripts/test_profile_sycl.py
```

Use `--fa-backend mkl` for the attention-path comparison and `--kv q8_0` for the 256K MTP sweep. Batch overrides work for both direct and server tests. Use `--config llama.cpp/build-profile-20261001/llama-swap.before.yaml` to reproduce pre-tuning profile arguments; all original sampling/context arguments were preserved, and full native commands are in the logs. Saved throughput/MTP cases are skipped unless `--repeat` is supplied. Prefer a fresh `--label`/results path for new experiments to preserve earlier raw outputs.

For the other memory checks, select `--alias qwen3.8-27b-mtp` or `--alias agents-a1` with `--depth 253952`. For live proxy compatibility checks, omit `--depth` and add `--proxy`; those checks do not start/stop servers themselves. Isolated validation always reruns, checks that its port is unused, and terminates only its own child server. PyYAML is required for production-profile commands; direct throughput tests otherwise use the Python standard library. The seven CPU-only helper checks passed in this pass.
