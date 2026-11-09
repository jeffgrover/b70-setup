# Intel SYCL refresh and Ripple integration - 2026-11-08

## Summary

Fast-forwarded the clean `llama.cpp` checkout from `8ca4060b7` (build 3014) to
`7d7f02875` (build 3279, `0.9.0-dev`) and rebuilt the SYCL server and benchmark
tools. The live `llama.cpp/build` link now selects
`build-sycl-20261108-make`; the previous build remains available through
`build-link-before-20261108`.

The refresh includes upstream Ripple (`ricursive`) model support and newer
Intel SYCL work. Matched Qwen controls against the October 27 binary were
within measurement noise. Ripple's AtomicChat AD-IQ3_XXS GGUF loaded at 8K
context on the B70, and a forced tool call succeeded through the live
llama-swap endpoint. This is a conservative initial agent profile, not a
complete Ripple tuning pass.

## Host and build

- GPU: Intel Arc Pro B70, PCI ID `0xe223`, 32 GB; USB4 eGPU.
- SYCL reports 32,530 MiB global device memory and Level Zero 1.14.37020.
- IntelLLVM 2026.2.0, oneMKL 2026.2, oneDNN 2026.1; the installed toolchain
  and driver stack did not change in this refresh.
- Source: `7d7f02875`, build 3279. `llama-server --version` reports
  `0.9.0-dev` and IntelLLVM 2026.2.0.
- New build directory: `llama.cpp/build-sycl-20261108-make`.
- CMake: Release, shared libraries, native CPU, SYCL target `INTEL`, oneDNN,
  FP16 kernels, SYCL graph support, host-memory fallback, direct Level Zero
  API; `GGML_SYCL_DEVICE_ARCH` remains unset for the JIT path.
- Built `llama-server`, `llama-bench`, and `test-backend-ops` with two jobs.
  The backend-ops binary was built but its test suite was not run in this
  update.
- The restricted build environment could not fetch the matching web UI
  assets. The llama-server API works; its embedded web UI is absent. The
  llama-swap dashboard remains available.

### Relevant upstream changes

- [DPAS flash attention for quantized KV cache](https://github.com/ggml-org/llama.cpp/pull/31125)
  adds an Intel matrix-engine path for supported quantized-cache attention.
- [Vectorized stores for reordered quantized weights](https://github.com/ggml-org/llama.cpp/pull/30976)
  covers Q4_0, K-quants, IQ3, and MXFP4.
- [Faster device exchange on B70 multi-device setups](https://github.com/ggml-org/llama.cpp/pull/30956)
  improves the SYCL communication path when more than one device participates.
- [Refuse allocations larger than VRAM](https://github.com/ggml-org/llama.cpp/pull/31285)
  improves failure handling for oversized device buffers.
- [Exact SYCL quantized-copy rounding](https://github.com/ggml-org/llama.cpp/pull/31465)
  fixes rounding for Q4_1, Q5_1, and Q8_0 copies.
- [Ripple architecture support](https://github.com/ggml-org/llama.cpp/pull/31692)
  is included in upstream from b12167 onward. The current source is b12211.

XMX ESIMD, DPAS flash attention, oneDNN, Level Zero direct allocation,
reordered-weight optimization, and VMM were enabled/reported by the built
binary. Runtime graph capture was disabled during the throughput comparison,
matching the October 27 profiling setup.

## Paired Qwen throughput check

These direct `llama-bench` runs compare the saved October 27 build
(`8ca4060b7`) with the November 8 build (`7d7f02875`) on the same B70,
IntelLLVM/oneAPI stack, model file, SYCL device, full GPU offload, flash
attention, q8 KV, and normal oneDNN dispatch. Runtime graphs were off. Short
cases use two repetitions and have limited statistical power.

| Workload | Build 3014 | Build 3279 | Change |
|---|---:|---:|---:|
| Qwen3.8-27B prompt, 512 tokens, batch 2048 / microbatch 512 | 1016.66 t/s (±20.05) | 1014.15 t/s (±17.91) | -0.25% |
| Qwen3.8-27B prompt, 8192 tokens, batch 4096 / microbatch 1024 | 1179.57 t/s (±1.57) | 1182.20 t/s (±1.31) | +0.22% |
| Qwen3.8-27B decode, 128 tokens, batch 2048 / microbatch 512 | 26.974 t/s (±0.005) | 27.319 t/s (±0.048) | +1.28% |

The throughput changes are too small to call a general performance win. The
update is useful for upstream model support, correctness fixes, and newer
kernel paths. Ripple has no earlier local baseline, so its result below is a
starting point rather than a before/after comparison.

## Ripple profile and validation

Model file:
`~/.lmstudio/models/AtomicChat/Ripple-65B-Preview-GGUF/Ripple-65B-Preview-AD-IQ3_XXS.gguf`
(27,964,982,752 bytes; 28 GB decimal). The file embeds its chat template;
`--jinja` is enabled for tool and reasoning parsing.

Initial llama-swap profile:

| Parameter | Value |
|---|---|
| Context | 8192 |
| KV | q8_0 for K and V |
| GPU offload | `-ngl 999` |
| Flash attention | on |
| Batch / microbatch | 512 / 128 |
| Parallel slots | 1 |
| Reasoning budget | 2048 |
| Sampling | temperature 0.7, top-p 0.95, top-k 40, min-p 0, repeat penalty 1.0, presence penalty 0 |

The model card publishes no default sampling settings; the sampler above is a
conservative project starting point, not a model-specific tuning result. The
65B model is dense, and the 8K context is far below its advertised maximum.
The profile loaded successfully at both 4K and 8K in isolated runs. A forced
OpenAI function call completed through both the candidate server and the live
llama-swap port 8080. The live Pi registry now lists `ripple-65b-preview` at
8192 context and 8192 maximum tokens. It remains selectable and was not made
Pi's default model. A Pi 1.5.0 JSON-mode one-shot request through the same
provider returned `READY`; its prompt used 3,670 tokens and generated two.
See the [AtomicChat GGUF model card](https://huggingface.co/AtomicChat/Ripple-65B-Preview-GGUF)
for quantization details and upstream support requirements.

One warm tool-call response generated 46 tokens at 11.6 tokens/s; its 178-token
uncached prompt processed at 41.5 tokens/s. A separate first request included
initial JIT overhead and is excluded. Cold load took about 31 seconds in this
test. Peak VRAM, sustained load, long-context operation, batch sweeps, coding
task quality, and full agent sessions were not measured. Retain the current
small batches until a longer monitored profile establishes safe headroom.

## Reproduction

The paired checks used:

```bash
source /opt/intel/oneapi/setvars.sh > /dev/null
export GGML_SYCL_ENABLE_GRAPH=0
export GGML_SYCL_FA_ONEDNN=1
llama-bench -m "$MODEL" -dev SYCL0 -ngl 999 -fa on \
  -ctk q8_0 -ctv q8_0 -p 8192 -n 0 \
  -b 4096 -ub 1024 -r 2 -o json
```

For the Qwen 512-token prompt and decode controls, use `-p 512 -n 0` and
`-p 0 -n 128`, respectively, with `-b 2048 -ub 512`. Run the same command
against `build-sycl-20261027/bin/llama-bench` and
`build-sycl-20261108-make/bin/llama-bench`, with each build's `bin` directory
first in `LD_LIBRARY_PATH`.

## Deployment and rollback

The live service was restarted after switching `llama.cpp/build`; `/v1/models`
lists `ripple-65b-preview`, and a forced function call returned a valid
`tool_calls` object through port 8080. The configured live Pi model registry
was synchronized. The model may unload after its normal 600-second idle TTL.

The previous binary directory is intact. To roll back, point `llama.cpp/build`
back to `build-sycl-20261027` and restart the user service. The October 27
binary predates Ripple support, so remove or disable the Ripple model stanza
before using that rollback build.
