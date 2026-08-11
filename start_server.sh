#!/usr/bin/env bash
set -e

# Kill any existing server
pkill -f llama-server || true
sleep 3

# Source oneAPI env
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

# Truncate log
: > /tmp/llama_server.log

# Launch detached
setsid /home/jeff/Code/intel/llama.cpp/build/bin/llama-server \
  -m /home/jeff/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --mmproj /home/jeff/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F32.gguf \
  -ngl 999 \
  -c 262144 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --parallel 1 \
  --host 127.0.0.1 --port 8080 \
  --jinja \
  --alias qwen3.6-35b-a3b \
  > /tmp/llama_server.log 2>&1 < /dev/null &

echo "PID=$!"
disown
