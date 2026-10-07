#!/usr/bin/env bash
# Compare the chosen greedy horizon with the new probabilistic mode.
# Arguments: f16 horizon, q8 horizon, batch, microbatch.
# Run after the initial sweep and recheck; do not overlap GPU workloads.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_f16_horizon="${1:-2}"
task_q8_horizon="${2:-2}"
task_batch="${3:-4096}"
task_ubatch="${4:-2048}"
task_results=docs/benchmarks/2026-10-07/results.jsonl
task_raw=llama.cpp/build-profile-20261007
task_initial_config=docs/benchmarks/2026-10-07/initial-swift-profiles.yaml
test -f "$task_initial_config" || {
  echo 'The historical Swift benchmark fixture is required for the old-build MTP control.' >&2
  exit 2
}
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  set +u
  source /opt/intel/oneapi/setvars.sh > /dev/null
  set -u
fi
trap 'systemctl --user start llama-swap.service' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
systemctl --user stop llama-swap.service

# Source-only real-text control: identical original batch, sampler and horizon.
python3 scripts/profile_sycl.py mtp --build llama.cpp/build-sycl-20261001 \
  --label baseline --config "$task_initial_config" --kv f16 --alias swift-1.5-27b-think \
  --horizons "$task_f16_horizon" --batch 4096 --ubatch 1024 \
  --results "$task_results" --raw "$task_raw"

for task_kv in f16 q8_0; do
  task_alias=swift-1.5-27b-think
  task_horizon="$task_f16_horizon"
  if [ "$task_kv" = q8_0 ]; then
    task_alias=swift-1.5-27b-mtp
    task_horizon="$task_q8_horizon"
  fi
  for task_sampling in greedy probabilistic; do
    python3 scripts/profile_sycl.py mtp --build llama.cpp/build-sycl-20261007 \
      --label candidate --config "$task_initial_config" --kv "$task_kv" --alias "$task_alias" \
      --horizons "$task_horizon" --draft-sampling "$task_sampling" \
      --batch "$task_batch" --ubatch "$task_ubatch" \
      --results "$task_results" --raw "$task_raw"
  done
done

# A useful new mode can move the optimum; bracket the selected horizon.
for task_kv in f16 q8_0; do
  task_alias=swift-1.5-27b-think
  if [ "$task_kv" = q8_0 ]; then task_alias=swift-1.5-27b-mtp; fi
  python3 scripts/profile_sycl.py mtp --build llama.cpp/build-sycl-20261007 \
    --label candidate --config "$task_initial_config" --kv "$task_kv" --alias "$task_alias" \
    --horizons 1,3 --draft-sampling probabilistic \
    --batch "$task_batch" --ubatch "$task_ubatch" \
    --results "$task_results" --raw "$task_raw"
done
