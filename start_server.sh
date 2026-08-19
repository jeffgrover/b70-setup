#!/usr/bin/env bash
set -e

# Standalone Qwen3.8 thinking-profile launcher. This bypasses llama-swap and
# listens on its usual port, so stop llama-swap.service before using it.

# Kill any existing standalone server
pkill -f llama-server || true
sleep 3

# Source oneAPI env
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

# Truncate log
: > /tmp/llama_server.log

# Launch detached
setsid /home/jeff/Code/intel/llama.cpp/build/bin/llama-server \
  -m /home/jeff/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf \
  --mmproj /home/jeff/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf \
  -ngl 999 \
  -c 131072 \
  -ctk f16 -ctv f16 \
  -fa on \
  --parallel 1 \
  --metrics \
  --host 127.0.0.1 --port 8080 \
  --jinja \
  --reasoning on \
  --reasoning-format deepseek \
  --reasoning-budget 8192 \
  --reasoning-preserve \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0 \
  --alias qwen3.8-27b-think \
  > /tmp/llama_server.log 2>&1 < /dev/null &

echo "PID=$!"
disown
