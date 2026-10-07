#!/usr/bin/env bash
# October 7 scoped Swift measurements; do not run alongside other GPU work.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_baseline=llama.cpp/build-sycl-20261001
task_candidate=llama.cpp/build-sycl-20261007
task_results=docs/benchmarks/2026-10-07/results.jsonl
task_raw=llama.cpp/build-profile-20261007
task_initial_config=docs/benchmarks/2026-10-07/initial-swift-profiles.yaml
test -x "$task_candidate/bin/llama-server"
test -x "$task_candidate/bin/llama-bench"
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  set +u
  source /opt/intel/oneapi/setvars.sh > /dev/null
  set -u
fi
trap 'systemctl --user start llama-swap.service' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
systemctl --user stop llama-swap.service

for task_label in baseline candidate; do
  task_build="$task_baseline"
  if [ "$task_label" = candidate ]; then task_build="$task_candidate"; fi
  python3 scripts/profile_sycl.py bench --build "$task_build" --label "$task_label" \
    --suite swift --results "$task_results" --raw "$task_raw"
  python3 scripts/profile_sycl.py bench --build "$task_build" --label "$task_label" \
    --suite comparison --cases qwen-f16-p512,agents-f16-p512 \
    --batch 4096 --ubatch 1024 --results "$task_results" --raw "$task_raw"
done
python3 scripts/profile_sycl.py bench --build "$task_candidate" --label candidate \
  --suite swift-tuning --results "$task_results" --raw "$task_raw"

for task_kv in f16 q8_0; do
  task_alias=swift-1.5-27b-think
  if [ "$task_kv" = q8_0 ]; then task_alias=swift-1.5-27b-mtp; fi
  python3 scripts/profile_sycl.py mtp --build "$task_candidate" --label candidate \
    --config "$task_initial_config" \
    --kv "$task_kv" --alias "$task_alias" --horizons 0,1,2,3 \
    --batch 4096 --ubatch 1024 --results "$task_results" --raw "$task_raw"
done
