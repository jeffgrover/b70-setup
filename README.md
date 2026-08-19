# Local LLM stack on Intel Arc Pro B70

Single-port OpenAI-compatible chat-completions endpoint at `http://127.0.0.1:8080/v1`, backed by `llama.cpp` (SYCL/Level-Zero → XMX) and fronted by `llama-swap` for transparent model switching. Used by `opencode` and `pi`.

## Hardware & OS

| | |
|---|---|
| Box | Minisforum Venus mini-PC, AMD CPU, 32 GB RAM, 64 GB swap |
| GPU | Intel Arc Pro B70 (Battlemage, 32 GB VRAM, device id `0xe223`) |
| Connection | USB4 → eGPU enclosure |
| OS | Ubuntu 26.04 LTS "resolute", kernel 7.0.0-x, `xe` driver |
| Compute runtime | Intel oneAPI compiler/MKL 2026.1, oneDNN 2026.0, Level-Zero/OpenCL packages `26.05.37020.3-1` |

## Architecture

```
opencode / pi
      ↓  POST /v1/chat/completions  { "model": "qwen3.6-35b-a3b" | "qwen3.8-27b" | "qwen3.8-27b-mtp" | "qwen3.8-27b-think" | "agents-a1" | "nemotron-3.5-lightning" | "nemotron-3.5-lightning-mtp" | "muse-glimmer-30b" | "gemma-4-e4b" | "gemma-4-31b-qat" | "glm-4.7-flash" }
http://127.0.0.1:8080
  llama-swap                          ← model registry: ~/Code/intel/llama-swap.yaml
      ↓  spawns/kills based on requested model
http://127.0.0.1:9000
  llama-server (SYCL build) ─────→ Intel Arc Pro B70 (XMX matmul)
                                    GGUFs: ~/.lmstudio/models/...
```

Only one llama-server runs at a time. First request to a different model triggers a swap (~20–30 s cold load over USB4); subsequent requests stay warm. Models go idle and unload after `ttl: 600` s of no traffic.

## Models

| Model | Path | Quant | Context | VRAM |
|---|---|---|---|---|
| `qwen3.6-35b-a3b` | `~/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` + `mmproj-F32.gguf` | UD-Q4_K_S | 256 K (q8_0 KV) | ~24.3 GB |
| `qwen3.8-27b` | `~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf` + `mmproj-F16.gguf` | Q4_K_S | 256 K (q8_0 KV) | ~24.2 GiB (78%) |
| `qwen3.8-27b-mtp` | Same MTP-preserving GGUF + projector | Q4_K_S + MTP | 256 K (q8_0 KV) | ~24.2 GiB (78%) |
| `qwen3.8-27b-think` | Same MTP-preserving GGUF + projector | Q4_K_S + MTP | 128 K (f16 KV) | ~24.2 GiB (same KV footprint) |
| `agents-a1` | `~/.lmstudio/models/InternScience/Agents-A1-Q4_K_M-GGUF/Agents-A1-Q4_K_M.gguf` | Q4_K_M | 256 K (f16 KV) | ~25.1 GiB |
| `nemotron-3.5-lightning` | `~/.lmstudio/models/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_S.gguf` | Q4_K_S | 256 K (f16 KV) | ~24.3 GB |
| `nemotron-3.5-lightning-mtp` | Same MTP-preserving GGUF | Q4_K_S + MTP | 256 K (f16 KV) | ~24.3 GB |
| `muse-glimmer-30b` | `~/.lmstudio/models/lmstudio-community/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-17gb.gguf` + `mmproj-kquant.gguf` | K-Quant-17GB | 128 K (q8_0 KV) | ~18.6 GB |
| `gemma-4-e4b` | `~/.lmstudio/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | Q4_K_M | 128 K (q8_0 KV) | ~6 GB |
| `gemma-4-31b-qat` | `~/.lmstudio/models/lmstudio-community/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf` | Q4_0 QAT | 128 K (q8_0 KV) | ~20 GB |
| `glm-4.7-flash` | `~/.lmstudio/models/lmstudio-community/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf` | Q4_K_M | 128 K (q8_0 KV) | ~19 GB |

GGUFs live under `~/.lmstudio/models/` so LM Studio sees them too — both stacks coexist.

### Measured performance

| Model | Prompt processing | Generation |
|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_S | 220.0 t/s in tool-call smoke test | 55.0 t/s |
| Qwen3.8-27B Q4_K_S | 836.5 t/s at 512 tokens; 864.7 t/s at 2,048 | 25.9 t/s baseline; 31.1 t/s MTP in the earlier 256K-profile tests |
| Qwen3.8-27B depth study | ~844 t/s at 512 tokens; ~67 t/s for tiny uncached prompts | q8_0: 26.6 t/s empty, 15.5 at 16K; f16: 22.2 at 16K; f16 + MTP: 33.8 with 86% acceptance |
| Nemotron 3.5 Lightning Q4_K_S | 342.4 t/s baseline; 336.7 t/s MTP | 56.5 t/s baseline; 30.8 t/s MTP |
| Muse Glimmer 30B K-Quant-17GB | 22.5 t/s in tool-call smoke test | 24.2 t/s |
| Agents-A1 Q4_K_M | 277.8 t/s at 32 tokens | 86.6 t/s bench; 81.6 t/s sustained server decode |
| Gemma-4 E4B Q4_K_M | 1710 t/s | 76.5 t/s |
| Gemma-4 31B QAT Q4_0 | 283.8 t/s | 10.9 t/s |
| GLM-4.7-Flash Q4_K_M | 496.2 t/s | 20.4 t/s |

### Agentic evaluation: Qwen3.8 before and after tuning

On 2026-08-19, the same office/elevator tasks were rerun through the real
`llama-server`/llama-swap path after switching agent work to
`qwen3.8-27b-think` (128K f16 KV, one-token MTP, an 8192-token reasoning
ceiling, and zero presence penalty). The archived baseline used the older
`qwen3.8-27b` profile, so this is an end-to-end profile comparison rather
than a single-variable benchmark.

| Run | Result | Tokens (input + output) | Turns | Wall time | Quality checks |
|---|---|---:|---:|---:|---|
| Pi standard, baseline `qwen3.8-27b` | success | 214,639 + 155,999 | 102 | not recorded by the harness | static/runtime clean |
| Pi standard, tuned `qwen3.8-27b-think` | success | 143,422 + 99,289 | 63 | approximately 1h40m service window | static/runtime clean |
| Pi-Wiggum, baseline `qwen3.8-27b` | pass | 95,341 + 40,455 | 42 | 2h12m04s | 7/7 logic tests; no browser errors |
| Pi-Wiggum, tuned `qwen3.8-27b-think` | pass | 161,872 + 109,356 | 60 | 1h45m17s | 7/7 logic tests; no browser errors |

The tuned Wiggum run finished 20.3% faster (26m47s saved) while producing
roughly twice as many tokens. Both runs produced the same 1,794 scene objects;
the tuned run observed 160 animation frames and 1,326 dynamic changes versus
149 and 1,194 before tuning. The standard run also passed its static and
runtime probes, although its generated scene was smaller than the baseline,
so those lightweight probes do not establish semantic parity by themselves.

The harness did not retain separate prompt-processing and generation rates for
these long runs. The short controlled measurements above remain the reliable
throughput data: f16 KV recovered the deep-context slowdown, and one-token MTP
reached about 33.8 t/s with 86% draft acceptance. The practical conclusion is
to keep `qwen3.8-27b-think` as the 128K daily agent profile, keep
`qwen3.8-27b-mtp` for 256K contexts, and retain the non-speculative
`qwen3.8-27b` alias as a fallback.

### Choosing a model

- Prefer `agents-a1` for substantial coding, research, and tool-driven work. It combines 35B-class capacity, verified native tool use, a 256K context, and about 81.6 t/s sustained generation on this machine. It can spend many tokens reasoning, so simple tasks may take longer than its raw token rate suggests.
- Use `gemma-4-e4b` for quick questions, summaries, transformations, and routine edits. Its small VRAM footprint, 1710 t/s prompt processing, and 76.5 t/s generation make it the fast path when the task does not need a larger model.
- Use `qwen3.6-35b-a3b` as the balanced general-purpose option. Its Unsloth UD-Q4_K_S quant is the fastest large Qwen configuration measured here so far, and its projector, developer-role handling, reasoning extraction, and tool calls are validated.
- Use `qwen3.8-27b-think` as the daily Qwen profile for OpenCode, Pi, and other reasoning-heavy agent work when 128K context is enough. Its f16 KV cache avoids the severe q8_0 slowdown measured at depth, and one-token MTP improved the recorded reasoning run to 33.8 t/s. Use `qwen3.8-27b-mtp` when a 256K q8_0 context is required and `qwen3.8-27b` as the conservative non-speculative fallback.
- Use `nemotron-3.5-lightning` to try NVIDIA's new text-only reasoning and agent model. Use the explicit `nemotron-3.5-lightning-mtp` alias only for MTP experiments; the current SYCL speculative path is slower and can intermittently stop making progress on longer generations.
- Try `muse-glimmer-30b` for agentic and multimodal work. Its profile includes the perception projector, native ATEM tool-call parsing, reasoning extraction, and the model authors' sampling defaults. Generation is usable at about 24.2 t/s, though prompt ingestion was relatively slow in the first local test.
- Keep `glm-4.7-flash` as an independent second opinion.

The measured models hit ~75 % of the B70's GDDR6 bandwidth ceiling. Token-gen rate degrades as the context fills (more KV state to attend per step). Don't expect more without quantizing the model further or using a smaller one — the bottleneck is VRAM bandwidth, not compute.

Qwen3.6-35B-A3B tuning notes:

- The previous LM Studio Q4_K_M quant validated at 262K q8 KV and left about 8.7 GiB free on the B70. It measured `37.2 t/s` with q8 KV and `40.1 t/s` with f16 KV; flash attention was required.
- The active alias now targets Unsloth's UD-Q4_K_S quant and attaches its F32 multimodal projector. It keeps q8 KV to preserve the 256K agent context. The profile validated at about 24.3 GB VRAM, leaving roughly 7.7 GB free, and produced 55.0 t/s in the first short tool-call test.

Qwen3.8-27B tuning and usage notes:

- [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) is a dense native vision-language model with 27B parameters, 64 layers, and a repeating hybrid layout of three Gated DeltaNet blocks followed by one full-attention block. It has a native 262,144-token context and can be extended toward 1M with YaRN; the standard profiles stay at the native limit, while `-think` deliberately trades half the context for f16 KV.
- The selected Q4_K_S GGUF is 16,121,359,328 bytes and attaches the 927,607,488-byte F16 vision projector. Across the 16 full-attention layers, f16 KV consumes exactly 64 KiB per token (`16 layers × 4 KV heads × 256 dimensions × K/V × 2 bytes`): 8 GiB at 128K or 16 GiB at 256K. The two 256K profiles therefore use q8_0 KV (roughly 8 GiB), while `-think` spends the same cache footprint on higher-precision f16 at 128K.
- Thinking is enabled by default. Temperature and truncation sampling follow the published thinking recipe: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=0`, and `repetition_penalty=1`. The 256K profiles use `--reasoning-budget 2048`, while `-think` raises the hard sampler cap to 8192; all three use `--reasoning-preserve`. Request-level `reasoning_effort` is independently passed into the embedded template and supports `low`, `medium`, and `xhigh` (`high` maps to `xhigh`), but it does not replace the server's hard reasoning-token cap. The 8192 value is a ceiling, not a latency-free default: consuming it fully would take about four minutes at 34 t/s, so use `low` for time-sensitive work. Request sampling parameters can still override the server defaults.
- Both downloads match their published SHA-256 values (`22200efc…a3eb3f9` for the model and `cbb841a9…0b4e43e` for the projector). The 866-tensor model retains a Q8_0 MTP projection and the required normalization tensors at `blk.64.nextn.*`. The 256K q8_0 profiles and the 128K f16 profile have the same nominal 8 GiB KV footprint and occupy roughly 24.2 GiB of the B70's 31.0 GiB exposed memory.
- The stable alias does not enable speculation. Matched warm 256-token runs averaged 25.87 t/s normally. MTP with one, two, and three draft tokens averaged 31.13, 30.62, and 29.25 t/s respectively, so the final alias uses `--spec-type draft-mtp --spec-draft-n-max 1`. Its 351/476 acceptance rate was 73.7%, and its 20.3% speedup is the first local MTP win measured on this SYCL stack.
- Sustained 1,024-token runs completed at 25.47 t/s baseline and 30.81 t/s MTP, with the latter accepting 464/620 drafts (74.8%). Both paths averaged 230.2 W package power and settled near 2.5 GHz under the card's power limit; unlike stalled Nemotron MTP, Qwen3.8 converts the same full power draw into higher useful throughput.
- At approximately 16K tokens of context, the q8_0 flash-attention path fell from 26.6 t/s at empty context to 15.5 t/s. Changing only the cache to f16 restored 22.2 t/s (+43% at depth); f16 plus MTP reached 33.8 t/s with 86% draft acceptance on the recorded reasoning prompt. Combining `ngram-mod` with MTP was worse at 31.4 t/s and 68% acceptance, so it is deliberately omitted.
- Qwen3.8 has one trained MTP block, but llama.cpp can reuse that block autoregressively to draft more than one token. Higher `--spec-draft-n-max` values are therefore not ignored; one token remains selected because the measured one-, two-, and three-token runs favored one.
- With `--parallel 1`, llama-server can reuse a matching prompt prefix within the warm slot. This helps normal growing agent conversations, but a model swap discards that cache and changes to the serialized prefix can prevent a hit; it should not be treated as guaranteed caching across every turn.
- A seeded code-review quality comparison used the same prompt, `reasoning_effort=low`, and a 3,200-token allowance for each presence penalty. Penalty 0 completed correctly in 2,733 tokens with all requested tests; 0.5 also completed but used 3,050 tokens and omitted one explicit edge-case test; 1.5 exhausted all 3,200 tokens, never reached the requested tests, and returned an implementation that failed to deduplicate tags within the first record. All Qwen3.8 profiles therefore use `--presence-penalty 0`.
- Do not use `reasoning_effort=none` with this GGUF/template as a latency shortcut. In a direct control at presence penalty 0, it spent the full 2,200-token allowance in `reasoning_content` and returned no final answer. `low` is the verified short-reasoning setting; the hard `--reasoning-budget` remains the reliable upper bound.
- Required-tool and complete tool-result round trips returned parsed `tool_calls`, separate `reasoning_content`, and a correct final answer with `reasoning_effort=low`. The F16 projector correctly identified the local llama.cpp logo, Pi returned the exact requested sentinel, and a real OpenCode run completed its `read` call and final response through the MTP alias.

Agents-A1 tuning and usage notes:

- [Agents-A1](https://internscience.github.io/Agents-A1/) is a Qwen3.5-architecture 35B-A3B hybrid MoE trained for long-horizon tool use. Its full 35B weight set must still reside in memory even though roughly 3B parameters are active per token. The 262K f16 KV profile uses about 25.1 GiB of VRAM on the B70, leaving roughly 6.9 GiB free.
- The default profile keeps thinking enabled and separates it into the OpenAI-compatible `reasoning_content` field. Do not give this model tiny output limits: a trivial arithmetic smoke test consumed 278 completion tokens before producing its three-character final answer. The reported 45K-token training trajectories span reasoning, tool calls, observations, and multiple turns; they do not imply that each response should be a 45K-token monologue.
- The model authors recommend `temperature=0.85`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=1.1`, and `repetition_penalty=1.0`. These are the server defaults for this profile, but request parameters from clients can override them.
- Keep the embedded Jinja template. It supplies the Qwen3-Coder XML tool format, supports parallel calls, and preserves tool observations across turns. Local smoke tests verified parsed `tool_calls`, `reasoning_content`, and a complete call → tool response → final answer round trip.
- The official Q4_K_M GGUF has no MTP/NextN tensors, so this profile deliberately does not enable speculative MTP. The optional 899 MB `Agents-A1-mmproj.gguf` was not downloaded and is not attached; the alias is text-only. Add a separate multimodal alias with `--mmproj` if vision is needed later.
- Use the native 262K context without RoPE scaling. Keep `--parallel 1` and flash attention. Controlled generation measured `80.51 t/s` with q8 KV and `86.56 t/s` with f16 KV; a matching 1,024-token server decode improved from `75.46` to `81.58 t/s` (+8.1%). The f16 profile also held the compute engine near 100% busy at 2.8 GHz, so higher polling and experimental SYCL graphs are not enabled.

Nemotron 3.5 Lightning tuning and usage notes:

- [NVIDIA Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) is a text-only hybrid MoE model with 30B total and 3B active parameters, configurable reasoning, native tool use, and a 1M maximum context. Both B70 aliases use 256K to leave practical memory headroom on its 32 GB card.
- The profile follows NVIDIA's recommended `temperature=1.0` and `top_p=0.95`. It explicitly disables llama.cpp's otherwise-active top-k and min-p filters so those extra samplers do not change the published recipe.
- Keep the GGUF's embedded Jinja template. It enables thinking by default, emits Qwen3-Coder-style XML tool calls, folds tool results back into user messages as expected by the model, and truncates earlier reasoning traces in multi-turn history. `--reasoning-format deepseek` exposes the current trace as OpenAI-compatible `reasoning_content`.
- [Bartowski's Q4_K_S quant](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) includes the native MTP draft layer as `blk.52.nextn.*`, with the draft projection quantized at Q4_0. The stable alias ignores that optional head; the experimental `-mtp` alias uses one draft token with a 0.75 probability floor. The suggested additional `ngram-mod` drafter remains omitted because it was pathological on this SYCL backend.
- The downloaded Q4_K_S matches Bartowski's published SHA-256 (`82d6eb30…869c752`). Both aliases loaded at 256K and used about 76% of the B70's VRAM. Required-tool smoke tests returned parsed `reasoning_content` and correct OpenAI `tool_calls` through both paths.
- In those matched short tests, normal decoding delivered 56.5 t/s while MTP delivered 30.8 t/s despite accepting all 30 draft tokens. The previous gbuzhf quant showed the same direction: 66.24 t/s without speculation and 38.97 t/s with MTP. Keep the non-MTP alias for agent work; MTP remains experimental on this SYCL backend due to the regression and earlier longer-run stalls. This is consistent with upstream reports of [MTP overhead on an Arc Pro B70](https://github.com/ggml-org/llama.cpp/issues/23533) and [intermittent speculative-decoding hangs](https://github.com/ggml-org/llama.cpp/issues/23268).

Muse Glimmer tuning and usage notes:

- [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a dense agentic model with a separate perception encoder, native 128K context, reasoning output, and ATEM-format tool calls. The current llama.cpp checkout includes dedicated model, multimodal, and chat-parser support.
- The profile loads the 17 GB dynamic K-Quant GGUF and its 1.4 GB k-quant projector, advertises text-and-image input to Pi and OpenCode, and uses q8 KV. Server defaults follow the authors' `temperature=1.0`, `top_p=0.95`, and `top_k=64` recommendation.
- Reasoning strength defaults to `high` in the embedded template. It can be changed through a system prompt such as `Reasoning strength: medium.`
- The validated 128K profile used about 18.6 GB VRAM. A required-tool smoke test returned parsed `reasoning_content` and a correct OpenAI `tool_calls` object at 24.2 t/s generation; a real OpenCode `build` request also completed successfully.

Retired profiles:

- `qwen3.6-27b`, `qwen3.6-27b-mtp`, and `nemotron-3-nano-omni` were removed from llama-swap and both harnesses on 2026-08-10. Their model files were already absent from `~/.lmstudio/models/`.
- Historical B70 generation measurements were 21.6 t/s for Qwen3.6-27B Q4_K_M, 21.9 t/s for its MTP Q4_K_S profile, and 23.1 t/s for Nemotron. These records explain the retirement decision without leaving dead runnable profiles.

For comparison, LM Studio's bundled Vulkan llama.cpp measured ~9 t/s in the historical Qwen3.6-27B test — Vulkan doesn't use XMX. **Always run through this stack, not LM Studio's Vulkan, for performance work.**

## llama.cpp update notes

Current local build: `2e92ecd02` (`b10502-9-g2e92ecd02`, binary build 1579), built with IntelLLVM 2026.1.1.

### Maintenance record: 2026-08-19

- Fast-forwarded the clean llama.cpp checkout by 77 commits from `7e4c0a968` to `2e92ecd02`, then rebuilt `llama-server` and `llama-bench` with the existing IntelLLVM/SYCL configuration.
- The directly relevant upstream fixes are corrected thread/block counts for quantized SYCL copy kernels (`f275595dd`) and support for the Hadamard source hint in SYCL (`0882c7bc8`). The resulting binaries report build 1579.
- Rechecked the current CMake cache: direct Level Zero allocation, oneDNN, FP16 kernels, SYCL graphs, host-memory fallback, and native CPU tuning remain enabled; `GGML_SYCL_DEVICE_ARCH` remains intentionally unset.
- Added the `qwen3.8-27b-think` daily-driver profile with 128K f16 KV, one-token MTP, an 8192-token reasoning cap, and the same vision, tool-use, and sampling metadata exposed to Pi and OpenCode.

### Maintenance record: 2026-08-14

- Fast-forwarded the clean llama.cpp checkout by 76 commits from `030ebb558` (`b10356-2-g030ebb558`) to `7e4c0a968` (`b10434`) before profiling Qwen3.8-27B.
- Incrementally rebuilt `llama-server`, `llama-bench`, and `test-backend-ops` with the existing Level Zero, oneDNN, FP16-kernel, SYCL-graph, and host-memory-fallback configuration. The matching embedded web UI is build 1502.
- The update adds directly relevant SYCL optimizations for fused Q4_K dense-FFN gate/up projections and Gated DeltaNet state writeback, enables host-pinned memory, improves recurrent-state rollback, auto-detects MTP draft types, and passes OpenAI `reasoning_effort` into chat templates.
- Focused post-build validation passed all 40 Gated DeltaNet and gated-linear-attention correctness cases on `SYCL0` (Arc Pro B70, Level Zero driver `1.14.37020`).

### Maintenance record: 2026-08-10

- Fast-forwarded the local llama.cpp checkout by 233 commits from `720d7fa40` (`b10121-4-g720d7fa40`) to `030ebb558` (`b10356-2-g030ebb558`).
- Reconfigured from a fresh CMake cache and rebuilt `llama-server`, `llama-bench`, and `test-backend-ops` with direct Level Zero allocation, oneDNN, FP16 kernels, SYCL graph support, and host-memory fallback.
- Rebuilt and embedded the matching llama.cpp web UI (build 1426) instead of retaining the previous cached assets.
- Verified the post-upgrade runtime on kernel `7.0.0-29-generic`: the `xe` driver owns the B70, Level Zero reports driver `1.14.37020`, and SYCL exposes 31023 MiB of device memory.
- Ran the focused SYCL matrix-multiplication suite on the B70: 1015/1015 tests passed. An end-to-end `gemma-4-e4b` chat completion through llama-swap also loaded and generated successfully.

The intervening SYCL work includes oneMKL/XMX flash attention for prompt processing, oneDNN flash-attention support for quantized and FP32 KV caches, fused RMS norm plus multiply, faster SSM/convolution and non-contiguous concat paths, and several quantization, copy, and device-memory correctness fixes.

### Maintenance record: 2026-07-25

- Fast-forwarded the local llama.cpp checkout from `4fc4ec554` (`b9859`) to `720d7fa40` (`b10121-4-g720d7fa40`) and rebuilt `llama-server` and `llama-bench` against the refreshed Intel oneAPI stack.
- Installed the Level Zero development package and rebuilt with direct Level Zero allocation, oneDNN, FP16 kernels, SYCL graph support, and host-memory fallback compiled in. Runtime graph capture remains disabled by its upstream default.
- Rebuilt and embedded the llama.cpp web UI, then verified the llama-swap dashboard on port 8080 and the transient llama.cpp UI/API on port 9000.
- Changed the standard Qwen3.6-27B profile to f16 KV at its existing 131K context, enabling the Battlemage oneDNN/XMX flash-attention path without changing or redownloading its Q4_K_M GGUF.
- Benchmarked the optional DMMV preference and left it disabled: it reduced Qwen3.6-27B generation from `23.62 t/s` to `18.04 t/s`.
- Updated `llm-swap configure` so Pi and opencode receive model context and output limits derived from each profile's `-c` value.

This update matters for Intel Arc because upstream llama.cpp now includes oneDNN/XMX flash attention, Battlemage flash-attention tuning, fused top-k MoE dispatch, `Q2_K` reorder support, and several SYCL quantization and copy correctness fixes. It also retains the earlier reorder optimizations for `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`.

Rebuild command used here:

```bash
cd ~/Code/intel/llama.cpp
git fetch --tags origin master
git checkout master
git pull --ff-only origin master
source /opt/intel/oneapi/setvars.sh > /dev/null
cmake --fresh -S . -B build \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server llama-bench test-backend-ops -j 6
```

Notes from the rebuild:

- Ubuntu package `libze-dev` is installed, so CMake enables `GGML_SYCL_SUPPORT_LEVEL_ZERO_API` and links the direct Level Zero allocation path.
- `GGML_SYCL_F16=ON` remains enabled. Upstream recommends testing both modes because FP16 can improve prompt processing depending on the model.
- Keep `GGML_SYCL_DEVICE_ARCH` unset for this build. The current upstream CMake logic skips `-ze-intel-greater-than-4GB-buffer-required` for `spir64_gen` AOT builds, which made the local large-model configuration unsuitable for the attempted `bmg-g21` AOT build. The normal JIT path is cached after its initial device compilation.
- The embedded llama.cpp web UI was built from the checked-out sources with npm and linked as gzip-compressed assets. The initial UI dependency install requires network access.
- Host validation reports `SYCL0: Intel(R) Graphics [0xe223]` with 31023 MiB and `llama-server --version` reports IntelLLVM 2026.1.1.

### Historical MTP speculative decoding (retired)

The current `llama-server` supports MTP speculative decoding via `--spec-type draft-mtp`. The retired `qwen3.6-27b-mtp` profile used:

```bash
--spec-type draft-mtp
--spec-draft-n-max 1
```

Unsloth documents MTP as roughly 1.5-2x faster inference on supported models. The retired profile used `--parallel 1` without a projector.

Local B70 SYCL testing did not show a speedup. On a controlled 512-token, temperature-0 request:

| Alias | Config | Generation | Draft acceptance |
|---|---|---:|---:|
| `qwen3.6-27b` | no speculation | 23.10 t/s | n/a |
| `qwen3.6-27b-mtp` | `--spec-draft-n-max 2` | 20.27 t/s | 298 / 424 |
| `qwen3.6-27b-mtp` | `--spec-draft-n-max 1` | 21.85 t/s | 226 / 285 |

The MTP path worked, but on this Intel SYCL backend the extra draft-context work cost more than the accepted draft tokens saved, so both 27B aliases were retired.

## Service control

```bash
# Status / start / stop / restart
systemctl --user status  llama-swap.service
systemctl --user start   llama-swap.service
systemctl --user stop    llama-swap.service
systemctl --user restart llama-swap.service        # do this after editing llama-swap.yaml

# Live logs
journalctl --user -u llama-swap.service -f

# Survive logout / boot without login (one-time, requires sudo)
sudo loginctl enable-linger jeff
```

The service runs as a user unit (no system root needed). Without `enable-linger`, it stops when you log out.

## Web interfaces and status

| Interface | URL | Availability | Purpose |
|---|---|---|---|
| llama-swap dashboard | `http://127.0.0.1:8080/ui/` | While `llama-swap.service` is active | Always-available Models, Activity, and Logs views |
| Embedded llama.cpp UI | `http://127.0.0.1:9000/` | Only while a model is loaded | Chat, connection and model status, context usage, live prompt progress, generation speed, and per-message token statistics |

Opening `http://127.0.0.1:8080/` redirects to the llama-swap dashboard. The embedded llama.cpp UI is served by the transient `llama-server` child; port 9000 disappears when the model is swapped or unloaded after its 600-second idle TTL.

Status endpoints:

| Endpoint | Availability | Contents |
|---|---|---|
| `http://127.0.0.1:8080/health` | While `llama-swap.service` is active | Proxy health |
| `http://127.0.0.1:9000/health` | While a model is loaded | llama-server health |
| `http://127.0.0.1:9000/slots` | While a model is loaded | Per-slot state, processed tokens, and timings |
| `http://127.0.0.1:9000/metrics` | While a model is loaded | Prometheus-format request, token, prompt, and cache metrics (`--metrics` is enabled in each model profile) |

These interfaces do not provide Intel GPU utilization, temperature, power, or complete VRAM telemetry. Use `nvtop` interactively for the B70 telemetry it exposes; package-energy counters are available under `/sys/class/drm/card0/device/hwmon/hwmon*/energy*_input`. The installed `intel_gpu_top` does not support the `xe` driver, and `xpu-smi` is not installed.

## Using opencode

Config: `~/.config/opencode/opencode.json` — provider `local-b70`; the existing default model selection is preserved when the provider list is regenerated.

```bash
opencode                                     # interactive TUI, default model
opencode -m local-b70/qwen3.6-35b-a3b        # interactive TUI, balanced Qwen MoE
opencode -m local-b70/qwen3.8-27b             # interactive TUI, stable Qwen3.8 profile
opencode -m local-b70/qwen3.8-27b-mtp         # interactive TUI, 256K Qwen3.8 + MTP
opencode -m local-b70/qwen3.8-27b-think       # interactive TUI, recommended 128K agent profile
opencode -m local-b70/agents-a1               # interactive TUI, long-horizon agent model
opencode -m local-b70/nemotron-3.5-lightning  # interactive TUI, stable Nemotron profile
opencode -m local-b70/nemotron-3.5-lightning-mtp # interactive TUI, experimental MTP
opencode -m local-b70/muse-glimmer-30b        # interactive TUI, agentic + vision model
opencode -m local-b70/gemma-4-e4b            # interactive TUI, gemma
opencode -m local-b70/glm-4.7-flash          # interactive TUI, GLM
opencode run "summarize this file" @file.py  # one-shot, default model
opencode run -m local-b70/qwen3.6-35b-a3b "..." # one-shot, balanced Qwen MoE
opencode run -m local-b70/qwen3.8-27b "..."  # one-shot, stable Qwen3.8
opencode run -m local-b70/qwen3.8-27b-mtp "..." # one-shot, 256K Qwen3.8 + MTP
opencode run -m local-b70/qwen3.8-27b-think "..." # one-shot, recommended 128K agent profile
opencode run -m local-b70/agents-a1 "..."    # one-shot, long-horizon agent model
opencode run -m local-b70/nemotron-3.5-lightning "..." # one-shot, Nemotron
opencode run -m local-b70/nemotron-3.5-lightning-mtp "..." # one-shot, experimental MTP
opencode run -m local-b70/muse-glimmer-30b "..." # one-shot, Muse Glimmer
opencode run -m local-b70/gemma-4-e4b "..."  # one-shot, gemma
opencode run -m local-b70/glm-4.7-flash "..." # one-shot, GLM
```

Inside the TUI, `/model` switches between the configured `local-b70` models transparently.

## Using pi

Config: `~/.pi/agent/models.json` — provider `local-b70`, all active llama-swap models listed.

```bash
pi --provider local-b70 --model qwen3.6-35b-a3b
pi --provider local-b70 --model qwen3.8-27b
pi --provider local-b70 --model qwen3.8-27b-mtp
pi --provider local-b70 --model qwen3.8-27b-think
pi --provider local-b70 --model agents-a1
pi --provider local-b70 --model nemotron-3.5-lightning
pi --provider local-b70 --model nemotron-3.5-lightning-mtp
pi --provider local-b70 --model muse-glimmer-30b
pi --provider local-b70 --model gemma-4-e4b -p "one-shot prompt"
pi --provider local-b70 --model glm-4.7-flash -p "one-shot prompt"
```

`pi config` opens a TUI for enabling/disabling extensions.

## Syncing client configs

`llm-swap configure` reads the live `llama-swap` model list from `LLAMA_SWAP_URL` and repairs the local clients:

```bash
~/Code/intel/bin/llm-swap configure
```

It updates:

| | |
|---|---|
| Pi agent models and per-model context/output limits | `~/.pi/agent/models.json` |
| Pi agent auth | `~/.pi/agent/auth.json` |
| opencode provider and per-model context/output limits | `~/.config/opencode/opencode.json` |

For both clients, context and maximum-output limits are set from each model profile's `-c` or `--ctx-size` value. Existing per-model fields are preserved; maintained metadata exposes reasoning, tool use, and the appropriate text/vision modalities for Qwen, Agents-A1, Nemotron, and Muse. The command also removes the retired Qwen 27B and Nemotron IDs from old local-provider entries. Run it after adding, renaming, removing, or changing a model.

The repository keeps sanitized snapshots in `configs/pi-models.json` and `configs/opencode.json`. Refresh them from the live registry before committing configuration changes:

```bash
PI_AGENT_CONFIG="$PWD/configs/pi-models.json" \
PI_AGENT_AUTH=/tmp/local-b70-auth.json \
OPENCODE_CONFIG="$PWD/configs/opencode.json" \
./bin/llm-swap configure
```

## Adding another model

1. Drop the GGUF under `~/.lmstudio/models/<owner>/<repo>/`.
2. Add a stanza to `~/Code/intel/llama-swap.yaml` (copy a `cmd:` block, change `-m`, `-c`, `--alias`).
3. `systemctl --user restart llama-swap`.
4. `~/Code/intel/bin/llm-swap configure`.

## Troubleshooting

**`502 Bad Gateway` on first request** — the spawned llama-server crashed. Check `journalctl --user -u llama-swap -n 50`. Common causes: bad `cmd:` quoting in YAML, missing GGUF, OOM (KV cache too big for free VRAM).

**Server won't start, "out of memory" / "free memory target"** — VRAM is fragmented from prior process churn. Reboot is the cleanest fix; the `xe` driver doesn't always release Level-Zero allocations promptly. (`rmmod xe` won't work while gnome-shell holds it.)

**Slow first response after switching models** — expected. The new model is loading from NVMe → over USB4 → into VRAM. Expect tens of seconds for the large models and roughly 10 seconds for the small Gemma. It stays warm after that.

**Wrong-looking GPU usage in nvtop** — nvtop normalizes each GPU to its own VRAM pool. The AMD 780M iGPU showing "50 %" is just gnome-shell using ~1 GB of its 2 GB UMA share for desktop compositing. The B70 is the only thing running model weights.

**Where's `icpx` / `sycl-ls`?** — `source /opt/intel/oneapi/setvars.sh` before invoking llama.cpp tools directly. Each llama-swap model command already does this.

**Things to avoid (lessons from the hard way):**
- Ollama on Intel: `OLLAMA_VULKAN=1`, `OLLAMA_NUM_GPU=999` etc. are not real env vars — pure hallucination.
- IPEX-LLM Docker (`intelanalytics/ipex-llm-serving-xpu`): XPU device count goes to zero inside the container on this combo. Skip.
- The checked-in `export_qwen.py` and `serve_ov.py` files are historical prototypes from the abandoned Qwen3.6/OpenVINO path, not part of the active stack. Revalidate current library and architecture support before reviving them.
- `kobuk-team/intel-graphics` PPA: hasn't published a Release file for `resolute`. Don't add it.

## File locations

| | |
|---|---|
| llama.cpp build | `~/Code/intel/llama.cpp/build/bin/` |
| llama-swap binary | `~/Code/intel/bin/llama-swap` |
| llama-swap config | `~/Code/intel/llama-swap.yaml` |
| Standalone Qwen3.8 thinking-profile launcher | `~/Code/intel/start_server.sh` (stop llama-swap first) |
| systemd unit | `~/.config/systemd/user/llama-swap.service` |
| opencode config | `~/.config/opencode/opencode.json` |
| opencode config snapshot | `~/Code/intel/configs/opencode.json` |
| pi config | `~/.pi/agent/models.json` |
| pi config snapshot | `~/Code/intel/configs/pi-models.json` |
| GGUFs | `~/.lmstudio/models/` |
| Intel oneAPI | `/opt/intel/oneapi/` (source `setvars.sh`) |
