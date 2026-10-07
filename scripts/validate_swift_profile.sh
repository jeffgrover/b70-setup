#!/usr/bin/env bash
# Run after throughput tuning and final Swift settings have been selected.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_build=llama.cpp/build-sycl-20261007
task_results=docs/benchmarks/2026-10-07/results.jsonl
task_raw=llama.cpp/build-profile-20261007
task_config="${1:-llama-swap.yaml}"
test -x "$task_build/bin/test-backend-ops"
mkdir -p "$task_raw"
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  set +u
  source /opt/intel/oneapi/setvars.sh > /dev/null
  set -u
fi
export LD_LIBRARY_PATH="$task_root/$task_build/bin:${LD_LIBRARY_PATH:-}"
export GGML_SYCL_ENABLE_GRAPH=0
export GGML_SYCL_FA_ONEDNN=1
trap 'systemctl --user start llama-swap.service' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
systemctl --user stop llama-swap.service

"$task_build/bin/test-backend-ops" test -b SYCL0 \
  -o ADD,L2_NORM,RMS_NORM,RMS_NORM_SCALE,MUL_MAT,GLU,SSM_CONV,SSM_CONV_BIAS_SILU,TOP_K \
  > "$task_raw/candidate-correctness-core.log" 2>&1
rg -q ' [1-9][0-9]*/[1-9][0-9]* tests passed' "$task_raw/candidate-correctness-core.log"
printf '%s\n' 'PASS: selected core backend correctness'
"$task_build/bin/test-backend-ops" test -b SYCL0 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,.*type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],.*n_kv_max=0' \
  > "$task_raw/candidate-correctness-fa-dense.log" 2>&1
rg -q ' [1-9][0-9]*/[1-9][0-9]* tests passed' "$task_raw/candidate-correctness-fa-dense.log"
printf '%s\n' 'PASS: selected dense attention correctness; verify nonzero test count in log'
if ! "$task_build/bin/test-backend-ops" test -b SYCL0 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,nh=1,nr23=\[12,2\],kv=8192,nb=67,.*type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,.*n_kv_max=512' \
  > "$task_raw/candidate-correctness-fa-sparse-edge.log" 2>&1; then
  printf '%s\n' 'NOTE: the wider sparse-mask edge check failed; see retained log'
fi

python3 scripts/validate_sycl.py --build "$task_build" --alias swift-1.5-27b-think \
  --config "$task_config" \
  --depth 126976 --chat-checks --vision-image llama.cpp/media/llama0-logo.png \
  --results "$task_results" --raw "$task_raw"
python3 scripts/validate_sycl.py --build "$task_build" --alias swift-1.5-27b-mtp \
  --config "$task_config" \
  --depth 253952 --vision-image llama.cpp/media/llama0-logo.png \
  --results "$task_results" --raw "$task_raw"
python3 scripts/validate_sycl.py --build "$task_build" --alias swift-1.5-27b \
  --config "$task_config" \
  --vision-image llama.cpp/media/llama0-logo.png --results "$task_results" --raw "$task_raw"
python3 scripts/validate_sycl.py --build "$task_build" --alias qwen3.8-27b-think \
  --config "$task_config" \
  --chat-checks --results "$task_results" --raw "$task_raw"
