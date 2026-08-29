#!/usr/bin/env bash
set -euo pipefail

# Run LocalAI beside llama-swap (which owns port 8080).
# Override LOCALAI_MODEL_DIR or LOCALAI_PORT when needed.
MODEL_DIR="${LOCALAI_MODEL_DIR:-/home/jeff/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF}"
PORT="${LOCALAI_PORT:-8081}"

if [[ ! -d "$MODEL_DIR" ]]; then
  printf 'Model directory does not exist: %s\n' "$MODEL_DIR" >&2
  exit 1
fi

exec podman run --rm -it --name local-ai \
  --device /dev/dri \
  -p "${PORT}:8080" \
  -e DEBUG=true \
  -e THREADS=1 \
  -e LOCALAI_FORCE_META_BACKEND_CAPABILITY=intel \
  -v "${MODEL_DIR}:/models" \
  quay.io/go-skynet/local-ai:master-gpu-intel \
  --models-path /models
