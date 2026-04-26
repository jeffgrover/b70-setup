# Local LLM stack on Intel Arc Pro B70

Single-port OpenAI-compatible chat-completions endpoint at `http://127.0.0.1:8080/v1`, backed by `llama.cpp` (SYCL/Level-Zero → XMX) and fronted by `llama-swap` for transparent model switching. Used by `opencode` and `pi`.

## Hardware & OS

| | |
|---|---|
| Box | Minisforum Venus mini-PC, AMD CPU, 32 GB RAM, 64 GB swap |
| GPU | Intel Arc Pro B70 (Battlemage, 32 GB VRAM, device id `0xe223`) |
| Connection | USB4 → eGPU enclosure |
| OS | Ubuntu 26.04 LTS "resolute", kernel 7.0.0-x, `xe` driver |
| Compute runtime | Intel oneAPI 2026.0 (`intel-basekit`), Level-Zero |

## Architecture

```
opencode / pi
      ↓  POST /v1/chat/completions  { "model": "qwen3.6-27b" | "gemma-4-e4b" }
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
| `qwen3.6-27b` | `~/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf` | Q4_K_M | 256 K (q8_0 KV) | ~25 GB |
| `gemma-4-e4b` | `~/.lmstudio/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | Q4_K_M | 128 K (q8_0 KV) | ~6 GB |

GGUFs live under `~/.lmstudio/models/` so LM Studio sees them too — both stacks coexist.

### Measured performance

| Model | Prefill (pp512) | Generation (tg128) |
|---|---|---|
| Qwen3.6-27B Q4_K_M | 292 t/s | 21.6 t/s |
| Gemma-4 E4B Q4_K_M | 1710 t/s | 76.5 t/s |

Both hit ~75 % of the B70's GDDR6 bandwidth ceiling. Token-gen rate degrades as the context fills (more KV state to attend per step). Don't expect more without quantizing the model further or using a smaller one — the bottleneck is VRAM bandwidth, not compute.

For comparison, LM Studio's bundled Vulkan llama.cpp gets ~9 t/s on the same Qwen model — Vulkan doesn't use XMX. **Always run through this stack, not LM Studio's Vulkan, for performance work.**

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

## Using opencode

Config: `~/.config/opencode/opencode.json` — provider `local-b70`, default model `qwen3.6-27b`.

```bash
opencode                                     # interactive TUI, default model
opencode -m local-b70/gemma-4-e4b            # interactive TUI, gemma
opencode run "summarize this file" @file.py  # one-shot, default model
opencode run -m local-b70/gemma-4-e4b "..."  # one-shot, gemma
```

Inside the TUI, `/model` switches between the two transparently.

## Using pi

Config: `~/.pi/agent/models.json` — provider `local-b70`, both models listed.

```bash
pi --provider local-b70 --model qwen3.6-27b
pi --provider local-b70 --model gemma-4-e4b -p "one-shot prompt"
```

`pi config` opens a TUI for enabling/disabling extensions.

## CLI helper: `llm-swap`

For software that doesn't speak llama-swap's auto-routing (e.g. tools written for LM Studio or cloud APIs that hard-code a model name), use the `bin/llm-swap` helper to preload a model first, then run your tool. llama-swap will keep that model warm until something else asks for a different one.

```bash
llm-swap qwen3.6-27b   # preload (returns when warm — ~22 s cold, instant if already loaded)
llm-swap gemma-4-e4b
llm-swap list          # configured models
llm-swap status        # currently loaded model
llm-swap unload        # free VRAM
```

Symlink it onto PATH: `ln -s ~/Code/intel/bin/llm-swap ~/.local/bin/llm-swap`. Honors `LLAMA_SWAP_URL` (default `http://127.0.0.1:8080`).

Note: llama-swap routes requests by the `model` field in the request body. If your software sends a model name that isn't in the registry (e.g. `gpt-4o`), it will be rejected — preloading doesn't change that. Either configure your software to send `qwen3.6-27b` / `gemma-4-e4b`, or add `aliases:` entries to `llama-swap.yaml`.

## Adding another model

1. Drop the GGUF under `~/.lmstudio/models/<owner>/<repo>/`.
2. Add a stanza to `~/Code/intel/llama-swap.yaml` (copy a `cmd:` block, change `-m`, `-c`, `--alias`).
3. Add the model to `~/.config/opencode/opencode.json` (under `provider.local-b70.models`).
4. Add the model to `~/.pi/agent/models.json` (under the `local-b70` provider's `models` array).
5. `systemctl --user restart llama-swap`.

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
