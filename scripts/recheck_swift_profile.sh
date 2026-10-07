#!/usr/bin/env bash
# Finish the added 64K microbatch controls and recheck the Agents regression.
# Run only after run_swift_profile.sh has completed; GPU work is serialized.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_results=docs/benchmarks/2026-10-07/results.jsonl
task_raw=llama.cpp/build-profile-20261007
if [[ "${SETVARS_COMPLETED:-}" != 1 ]]; then
  set +u
  source /opt/intel/oneapi/setvars.sh > /dev/null
  set -u
fi
trap 'systemctl --user start llama-swap.service' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
systemctl --user stop llama-swap.service

python3 scripts/profile_sycl.py bench --build llama.cpp/build-sycl-20261007 \
  --label candidate --suite swift-tuning --cases p64000-n0-d0-b4096-ub2048 \
  --results "$task_results" --raw "$task_raw"

# ABBA ordering, with three measured pp512/tg128 repetitions per invocation.
for task_pair in 1 2; do
  task_order="baseline candidate"
  if [ "$task_pair" = 2 ]; then task_order="candidate baseline"; fi
  for task_label in $task_order; do
    task_build=llama.cpp/build-sycl-20261001
    if [ "$task_label" = candidate ]; then task_build=llama.cpp/build-sycl-20261007; fi
    python3 scripts/profile_sycl.py bench --build "$task_build" \
      --label "$task_label-repeat$task_pair" --suite comparison --cases agents-f16-p512 \
      --batch 4096 --ubatch 1024 --results "$task_results" --raw "$task_raw"
  done
done
