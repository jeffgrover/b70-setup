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
      ↓  POST /v1/chat/completions  { "model": "qwen3.6-27b" | "qwen3.6-35b-a3b" | "nemotron-3-nano-omni" | "gemma-4-e4b" | "gemma-4-31b-qat" | "glm-4.7-flash" }
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
| `qwen3.6-27b` | `~/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf` | Q4_K_M | 128 K (f16 KV) | ~26 GB |
| `qwen3.6-27b-mtp` | `~/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_K_S.gguf` | Q4_K_S + MTP | 128 K (q8_0 KV) | ~20 GB |
| `qwen3.6-35b-a3b` | `~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf` | Q4_K_M | 256 K (q8_0 KV) | ~21.6 GiB |
| `nemotron-3-nano-omni` | `~/.lmstudio/models/lmstudio-community/nemotron-3-nano-omni-30b-a3b-reasoning-gguf/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Q4_K_M.gguf` | Q4_K_M | 256 K (f16 KV) | ~23.7 GiB |
| `gemma-4-e4b` | `~/.lmstudio/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | Q4_K_M | 128 K (q8_0 KV) | ~6 GB |
| `gemma-4-31b-qat` | `~/.lmstudio/models/lmstudio-community/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf` | Q4_0 QAT | 128 K (q8_0 KV) | ~20 GB |
| `glm-4.7-flash` | `~/.lmstudio/models/lmstudio-community/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf` | Q4_K_M | 128 K (q8_0 KV) | ~19 GB |

GGUFs live under `~/.lmstudio/models/` so LM Studio sees them too — both stacks coexist.

### Measured performance

| Model | Prompt processing | Generation |
|---|---|---|
| Qwen3.6-27B Q4_K_M | 292 t/s | 21.6 t/s |
| Qwen3.6-27B MTP Q4_K_S | 340.9 t/s baseline | 21.9 t/s configured MTP (`--spec-draft-n-max 1`) |
| Qwen3.6-35B-A3B Q4_K_M | 534.0 t/s short prompt; 503.3 t/s at 4K prompt | 37.2 t/s with q8 KV; 40.1 t/s with f16 KV |
| Nemotron-3 Nano Omni 30B-A3B Q4_K_M | 483.1 t/s short prompt; 484.2 t/s at 4K prompt | 23.1 t/s with f16 KV; 22.6 t/s with q8 KV |
| Gemma-4 E4B Q4_K_M | 1710 t/s | 76.5 t/s |
| Gemma-4 31B QAT Q4_0 | 283.8 t/s | 10.9 t/s |
| GLM-4.7-Flash Q4_K_M | 496.2 t/s | 20.4 t/s |

The measured models hit ~75 % of the B70's GDDR6 bandwidth ceiling. Token-gen rate degrades as the context fills (more KV state to attend per step). Don't expect more without quantizing the model further or using a smaller one — the bottleneck is VRAM bandwidth, not compute.

Qwen3.6-27B tuning notes:

- The 131K f16 KV profile loads successfully on the B70. It uses roughly 1 GiB more VRAM than q8 KV because only 16 of the model's 64 layers use full-attention KV.
- f16 KV enables the Battlemage oneDNN/XMX flash-attention path. In controlled `llama-bench` tests, 512-token prompt processing improved from `701.07 t/s` with q8 KV to `794.83 t/s` with f16 KV, while 128-token generation improved from `23.62 t/s` to `24.16 t/s`.

Qwen3.6-35B-A3B tuning notes:

- The 262K q8 KV profile loads successfully and leaves about 8.7 GiB free on the B70.
- f16 KV is slightly faster in short generation tests (`40.1 t/s` vs `37.2 t/s`) but uses more KV memory. The `llama-swap` profile uses q8 KV to preserve maximum context for coding-agent runs.
- Flash attention is required for the tested profile; `-fa off` failed context creation during tuning.

Nemotron tuning notes:

- The 262K f16 KV profile loads successfully and leaves about 6.8 GiB free on the B70.
- f16 KV is slightly faster than q8 KV for this model and still has enough memory margin because only 6 of 52 layers use attention KV cache; many layers are recurrent/SSM.
- The model offloads 53/53 layers to SYCL, but `token_embd.weight` is CPU-mapped (`231 MiB`). Its `nemotron_h_moe` architecture mixes MoE and SSM/Gated Delta Net work, so instantaneous GPU utilization can vary more than dense Qwen/Gemma models. During the measured prompt/generation kernels it can still hit full GPU utilization.
- The downloaded `mmproj-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.gguf` is not attached in `llama-swap`; it would spend VRAM on multimodal support that coding-agent text prompts do not need.

For comparison, LM Studio's bundled Vulkan llama.cpp gets ~9 t/s on the same Qwen model — Vulkan doesn't use XMX. **Always run through this stack, not LM Studio's Vulkan, for performance work.**

## llama.cpp update notes

Current local build: `720d7fa40` (`b10121-4-g720d7fa40`), built with IntelLLVM 2026.1.0.

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
cmake --build build --config Release --target llama-server llama-bench -j 6
```

Notes from the rebuild:

- Ubuntu package `libze-dev` is installed, so CMake enables `GGML_SYCL_SUPPORT_LEVEL_ZERO_API` and links the direct Level Zero allocation path.
- `GGML_SYCL_F16=ON` remains enabled. Upstream recommends testing both modes because FP16 can improve prompt processing depending on the model.
- The embedded llama.cpp web UI was built from the checked-out sources with npm and linked as gzip-compressed assets. The initial UI dependency install requires network access.
- Host validation reports `SYCL0: Intel(R) Graphics [0xe223]` with 31023 MiB and `llama-server --version` reports IntelLLVM 2026.1.0.

### MTP speculative decoding

The current `llama-server` supports MTP speculative decoding via `--spec-type draft-mtp`. A separate `qwen3.6-27b-mtp` alias is configured for the local Unsloth MTP GGUF:

```bash
--spec-type draft-mtp
--spec-draft-n-max 1
```

Unsloth documents MTP as roughly 1.5-2x faster inference on supported models. Their current llama.cpp guidance also says `--parallel` values greater than 1 and `--mmproj` are not yet supported with MTP; this stack uses `--parallel 1` and does not attach the downloaded `mmproj-F32.gguf` for the MTP alias.

Local B70 SYCL testing did not show a speedup. On a controlled 512-token, temperature-0 request:

| Alias | Config | Generation | Draft acceptance |
|---|---|---:|---:|
| `qwen3.6-27b` | no speculation | 23.10 t/s | n/a |
| `qwen3.6-27b-mtp` | `--spec-draft-n-max 2` | 20.27 t/s | 298 / 424 |
| `qwen3.6-27b-mtp` | `--spec-draft-n-max 1` | 21.85 t/s | 226 / 285 |

The MTP path works, but on this Intel SYCL backend the extra draft-context work currently costs more than the accepted draft tokens save. Treat `qwen3.6-27b-mtp` as experimental and prefer `qwen3.6-27b` for throughput-sensitive work unless a future llama.cpp/SYCL update changes this.

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

These interfaces do not provide Intel GPU utilization, temperature, power, or complete VRAM telemetry. Use the installed `intel_gpu_top` for hardware monitoring; `xpu-smi` is another option if installed separately.

## Using opencode

Config: `~/.config/opencode/opencode.json` — provider `local-b70`, default model `qwen3.6-27b`.

```bash
opencode                                     # interactive TUI, default model
opencode -m local-b70/qwen3.6-35b-a3b        # interactive TUI, large Qwen MoE
opencode -m local-b70/nemotron-3-nano-omni   # interactive TUI, Nemotron hybrid MoE/SSM
opencode -m local-b70/gemma-4-e4b            # interactive TUI, gemma
opencode -m local-b70/glm-4.7-flash          # interactive TUI, GLM
opencode run "summarize this file" @file.py  # one-shot, default model
opencode run -m local-b70/qwen3.6-35b-a3b "..." # one-shot, large Qwen MoE
opencode run -m local-b70/nemotron-3-nano-omni "..." # one-shot, Nemotron
opencode run -m local-b70/gemma-4-e4b "..."  # one-shot, gemma
opencode run -m local-b70/glm-4.7-flash "..." # one-shot, GLM
```

Inside the TUI, `/model` switches between the configured `local-b70` models transparently.

## Using pi

Config: `~/.pi/agent/models.json` — provider `local-b70`, all active llama-swap models listed.

```bash
pi --provider local-b70 --model qwen3.6-27b
pi --provider local-b70 --model qwen3.6-35b-a3b
pi --provider local-b70 --model nemotron-3-nano-omni
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

For both clients, context and maximum-output limits are set from each model profile's `-c` or `--ctx-size` value. Existing Pi model fields are preserved. Run the command after adding, renaming, removing, or changing the context size of a model in `llama-swap.yaml`.

## Adding another model

1. Drop the GGUF under `~/.lmstudio/models/<owner>/<repo>/`.
2. Add a stanza to `~/Code/intel/llama-swap.yaml` (copy a `cmd:` block, change `-m`, `-c`, `--alias`).
3. `systemctl --user restart llama-swap`.
4. `~/Code/intel/bin/llm-swap configure`.

## Troubleshooting

**`502 Bad Gateway` on first request** — the spawned llama-server crashed. Check `journalctl --user -u llama-swap -n 50`. Common causes: bad `cmd:` quoting in YAML, missing GGUF, OOM (KV cache too big for free VRAM).

**Server won't start, "out of memory" / "free memory target"** — VRAM is fragmented from prior process churn. Reboot is the cleanest fix; the `xe` driver doesn't always release Level-Zero allocations promptly. (`rmmod xe` won't work while gnome-shell holds it.)

**Slow first response after switching models** — expected. The new model is loading from NVMe → over USB4 → into VRAM. ~20–30 s for the 27B, ~10 s for gemma. Keeps warm after that.

**Wrong-looking GPU usage in nvtop** — nvtop normalizes each GPU to its own VRAM pool. The AMD 780M iGPU showing "50 %" is just gnome-shell using ~1 GB of its 2 GB UMA share for desktop compositing. The B70 is the only thing running model weights.

**Where's `icpx` / `sycl-ls`?** — `source /opt/intel/oneapi/setvars.sh` before invoking llama.cpp tools directly. The systemd unit already does this.

**Things to avoid (lessons from the hard way):**
- Ollama on Intel: `OLLAMA_VULKAN=1`, `OLLAMA_NUM_GPU=999` etc. are not real env vars — pure hallucination.
- IPEX-LLM Docker (`intelanalytics/ipex-llm-serving-xpu`): XPU device count goes to zero inside the container on this combo. Skip.
- OpenVINO IR conversion via `optimum-cli`: doesn't recognize bleeding-edge architectures like `qwen3_5` yet. Not worth fighting unless you have a reason.
- `kobuk-team/intel-graphics` PPA: hasn't published a Release file for `resolute`. Don't add it.

## File locations

| | |
|---|---|
| llama.cpp build | `~/Code/intel/llama.cpp/build/bin/` |
| llama-swap binary | `~/Code/intel/bin/llama-swap` |
| llama-swap config | `~/Code/intel/llama-swap.yaml` |
| systemd unit | `~/.config/systemd/user/llama-swap.service` |
| opencode config | `~/.config/opencode/opencode.json` |
| pi config | `~/.pi/agent/models.json` |
| GGUFs | `~/.lmstudio/models/` |
| Intel oneAPI | `/opt/intel/oneapi/` (source `setvars.sh`) |
