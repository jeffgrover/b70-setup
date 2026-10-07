# Swift 1.5 integration and SYCL profiling, 2026-10-07

Completed and deployed. Swift has three separate aliases: 256K q8 plain,
256K q8/MTP, and 128K f16/MTP. Select 4096/2048 batching and two probabilistic
drafts for the MTP aliases. Matched sampling-mode gains are 9.5% f16 and 4.7%
q8 generation; direct source-only throughput is essentially flat. At deep
context MTP can improve decoding while slowing an uncached short-output request.
The 256K MTP profile fits but leaves only about 1.6 GiB free in its text check.

All three native targets succeeded. Native output reports `0.6.0-dev`, build
2546, commit `18b5f8b18`, IntelLLVM 2026.1.1. UI download requested `b2546`,
resolved to the upstream `latest` fallback, and verified the downloaded archive;
that is separate from native-binary provenance. Seventeen CPU-only helper tests
pass. The stable `llama.cpp/build` link now targets `build-sycl-20261007`.
The [188 recorded measurements/checks](results.jsonl) contain 50 direct bench
records, 106 bounded real-text MTP/non-MTP samples, 18 complete-answer samples
(12 clarified paired checks plus six original Swift checks), 11 isolated/proxy
validation records, and three old/new role controls. Full responses/logs remain
under ignored `llama.cpp/build-profile-20261007`, not in Git.

Swift paired controls are essentially flat: prompt rates change by -0.6 to -1.1%, decoding by +0.4 to +0.9%. An initial Agents-A1 short-prefill outlier was about 11% slower, but did not reproduce in matched old/new/new/old repeats; it is retained below rather than hidden.

## Scope and provenance

- Saved baseline: `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd`, local build 2415, `llama.cpp/build-sycl-20261001`.
- Deployed candidate: `18b5f8b1862ebfe0f1c33d2355b81a54d2fec867`, `b11477-1-g18b5f8b18`, 131 upstream commits newer, `llama.cpp/build-sycl-20261007`.
- IntelLLVM 2026.1.1, oneMKL 2026.1, oneDNN 2026.0 (library 3.11), and Level Zero driver 1.14.37020 are retained; no compiler/driver package change. Host: Ryzen 9 7940HS (8 physical/16 logical CPUs), kernel 7.0.0-38-generic, B70 device 0xe223 reporting 32656 MiB. The host has 32 GB installed RAM (28 GiB reported usable) and 8 GiB active swap, not the README's former 64 GB swap claim.
- Model: [UkisAI Swift 1.5 Qwen3.8-27B Q4_K_S](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF), 16,363,513,536 bytes; F16 projector 927,606,912 bytes.
- The GGUF has 866 tensors, `qwen35` architecture, native 262144 context, and one MTP block. Its MTP projection is Q4_0 rather than the base Qwen GGUF's Q8_0; the draft horizon needs an independent sweep. Both downloaded files retain the publisher's `Miroslav 1.0` metadata label. Local SHA-256 values match the publisher's Hugging Face LFS metadata: model `46a360c2cdde61b044c0ec188aaca9127689ab0e2418e870de6127e47345ac2c`, projector `10a24dc46eb801ad794886ef27ea1e43634d35388dfaf2e2374b7a8b67faf526`. Provenance follows the verified files, not that stale display label.
- Published evaluation sampling is temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repeat penalty 1. The embedded Swift template supports `low`, `medium`, and `xhigh` reasoning; template compatibility is tested separately.
- The publisher's reasoning-token savings come from other tasks/precisions/runtimes, not this B70 measurement. The publisher uses Swift Open License v1; weights and projector remain outside Git.

## Relevant upstream work since the baseline

- [Avoid slow oneDNN reference implementations](https://github.com/ggml-org/llama.cpp/pull/28985): probe supported implementations at device initialization. The upstream B70 control was flat; do not import the large iGPU speedup claim onto this dGPU.
- [Q8_0 ESIMD DMMV and wide-load MMVQ](https://github.com/ggml-org/llama.cpp/pull/29186): targets Q8_0 weight operators, not a generic speedup for q8 KV. Applicability depends on this mixed-quant GGUF's tensor layout.
- [Probabilistic draft sampling/rejection verification](https://github.com/ggml-org/llama.cpp/pull/27694): new opt-in `--spec-draft-sampling probabilistic`; greedy remains the default. Compare throughput and tool compatibility before enabling it.
- [GLM MLA MKL prefill](https://github.com/ggml-org/llama.cpp/pull/29171): newly merged, architecture/precision-specific, not a Swift/Qwen optimization.
- Additional work includes IQ3_S MMVQ, head-width-512 register tuning, mixed-device attention fixes, and shared NextN tensor/cropping helpers. IQ3_S, width 512, and mixed-device changes do not match this single-B70 Q4_K_S/head-width-256 workload.

## Matched Swift source-only controls

Direct `llama-bench`, full GPU offload, normal oneDNN attention dispatch,
flash attention, eight CPU threads, runtime graphs off. Each case has a
discarded warmup and two measured repetitions (three for pp512/tg128).
These controls do not attach the vision projector or enable MTP. All rates
are tokens/second; `±` is sample standard deviation, not a confidence interval.

| KV | Workload | Batch / microbatch | October 1 baseline | October 7 candidate | Change |
|---|---|---:|---:|---:|---:|
| q8_0 | pp512 | 2048 / 512 | 840.67 ± 2.70 | 833.23 ± 3.04 | -0.9% |
| q8_0 | tg128, empty KV | 2048 / 512 | 26.42 ± 0.04 | 26.62 ± 0.03 | +0.8% |
| q8_0 | pp8192 | 4096 / 1024 | 1077.63 ± 4.06 | 1066.17 ± 2.44 | -1.1% |
| q8_0 | pp64000 | 4096 / 1024 | 866.38 ± 0.51 | 858.79 ± 0.10 | -0.9% |
| q8_0 | tg128 at 64000 KV depth | 4096 / 1024 | 12.43 ± 0.00 | 12.48 ± 0.01 | +0.4% |
| f16 | pp512 | 2048 / 512 | 881.81 ± 1.44 | 874.29 ± 0.35 | -0.9% |
| f16 | tg128, empty KV | 2048 / 512 | 26.61 ± 0.05 | 26.84 ± 0.05 | +0.9% |
| f16 | pp8192 | 4096 / 1024 | 1073.41 ± 0.90 | 1066.76 ± 0.56 | -0.6% |
| f16 | pp64000 | 4096 / 1024 | 866.70 ± 0.13 | 861.83 ± 0.09 | -0.6% |
| f16 | tg128 at 64000 KV depth | 4096 / 1024 | 14.96 ± 0.00 | 15.01 ± 0.02 | +0.4% |

There is no material source-only speedup for Swift in this matrix. The
candidate's f16 decode at 64K is about 20% faster than q8_0, supporting a
128K f16 speed profile alongside 256K q8 capacity profiles. This is a cache
tradeoff, not a claim about Swift's reasoning efficiency or model quality.

### Existing-model controls and the Agents outlier

Qwen f16 at 4096/1024 measured pp512/tg128 of 891.52/27.19 on the baseline
and 882.72/27.10 on the candidate. At 16384 KV depth, decode was 22.40 versus
22.36. These are essentially flat and consistent with the prior October report.

Agents-A1's initial candidate pp512 was 1021.07 ± 18.79 versus baseline
1142.79 ± 6.06. Before deploying, repeated the identical case in ABBA order,
three measured repetitions per invocation:

| Order | Build | pp512 | tg128, empty KV |
|---:|---|---:|---:|
| 1 | baseline-repeat1 | 1145.41 ± 6.72 | 101.20 ± 0.13 |
| 2 | candidate-repeat1 | 1146.54 ± 3.07 | 100.86 ± 0.19 |
| 3 | candidate-repeat2 | 1157.55 ± 11.22 | 100.82 ± 0.14 |
| 4 | baseline-repeat2 | 1146.95 ± 4.54 | 101.14 ± 0.01 |

The short-prefill regression is not repeatable in these controls, and decode
differs by less than 0.5%. The initial outlier's cause was not established;
do not attribute it to a specific upstream change or omit it from the data.
No existing model parameters were changed to obtain these repeated results.

## Batch tuning

Candidate-only prompt measurements, same direct benchmark method as above.
4096/1024 controls are reused rather than counted as fresh repetitions.
The 4096/2048 deep controls were repeated after the initial MTP sweep.

| KV | Batch / microbatch | pp8192 | pp64000 |
|---|---:|---:|---:|
| q8_0 | 2048 / 512 | 859.69 ± 4.61 | not tested |
| q8_0 | 4096 / 1024 | 1066.17 ± 2.44 | 858.79 ± 0.10 |
| q8_0 | 4096 / 2048 | 1176.30 ± 0.66 | 927.14 ± 0.26 |
| q8_0 | 8192 / 2048 | 1175.04 ± 1.28 | 926.80 ± 0.14 |
| f16 | 2048 / 512 | 870.28 ± 0.56 | not tested |
| f16 | 4096 / 1024 | 1066.76 ± 0.56 | 861.83 ± 0.09 |
| f16 | 4096 / 2048 | 1176.17 ± 2.97 | 924.49 ± 0.24 |
| f16 | 8192 / 2048 | 1177.65 ± 1.81 | 924.23 ± 0.86 |

2048-token microbatches improve 8K prefill by about 10% over the existing
1024-token microbatch; 4096/2048 improves 64K prefill by 7–8%. The 8192
logical batch does not improve either measured prompt length over 4096/2048.
Keep the 4096 logical batch and select a Swift microbatch of 2048, supported
by the real-text and production-allocation checks below. This doubles working
buffer demand relative to 1024; q8/MTP's near-limit headroom is tight.

## Client compatibility

Swift's embedded template accepts `low`, `medium`, and `xhigh`, and defaults
to `xhigh`; literal `high` is not listed. Pi 0.85.1 supports a model-level
`thinkingLevelMap`: the generator maps minimal/low to low, medium to medium,
high/xhigh/max to xhigh, and off to the server's `none` template switch.
OpenCode 1.18.30 gets low/medium/xhigh variants and a high variant mapped to
xhigh. Existing model metadata and the chosen client default remain unchanged.
See [OpenCode custom variants](https://opencode.ai/docs/models/#custom-variants)
and [AI SDK compatible-provider options](https://ai-sdk.dev/providers/openai-compatible-providers#chat-model-options).

These are request serialization settings, not replacements for the server's
2048/8192 reasoning budgets or guarantees of task quality/latency. Swift's
embedded template lacks a literal developer branch, so the conservative Pi
metadata uses system messages. Native llama.cpp normalizes developer to system;
rendered prompts preserved the instruction on saved/new Qwen and new Swift.
The original "Confirm the instruction" probe produced generic readiness replies
on both system/developer roles and both builds; that ambiguous obedience check
does not establish a role-handling regression. The clarified harmless greeting
probe returned ROLE-OK through all three Swift aliases and Qwen via llama-swap.
Literal high failed Swift template
application with HTTP 500; low/medium/xhigh were accepted.
The actual v1 request with `reasoning_effort=none` returned the requested OFF-OK
answer but still emitted 115 reasoning characters. Pi's off mapping sends none;
it must not be advertised as a reliable no-thinking switch for this model/runtime.
Use low for the measured low-effort path. Template acceptance alone does not
prove that reasoning is suppressed.

## Real-text draft-horizon sweep

The candidate uses the actual Swift production profile, including projector,
full 128K f16 or 256K q8 target-cache allocation, one slot, and 4096/1024
batching. The drafter's separate cache remains its default f16. The three
coding/reasoning/proxy-investigation workloads contain 5522–5542 input tokens,
deterministic repository-like padding, seed 42, published sampling, and the
embedded template's default xhigh effort. Each workload has one discarded
full warmup and two measured requests. `cache_prompt=false`; recorded requests
have zero reused prompt tokens. Rates below are arithmetic means over six
samples; variability includes different workloads and draft outcomes.

| KV | Maximum greedy drafts | Generation t/s | Warm request seconds | Accepted / generated drafts |
|---|---:|---:|---:|---:|
| f16 | 0 | 24.85 ± 0.01 | 26.25 | n/a |
| f16 | 1 | 31.37 ± 0.85 | 22.64 | 1220 / 1841 (66.3%) |
| f16 | 2 | 32.18 ± 0.93 | 22.21 | 1537 / 3043 (50.5%) |
| f16 | 3 | 30.91 ± 1.92 | 22.92 | 1697 / 4084 (41.6%) |
| q8_0 | 0 | 23.87 ± 0.02 | 27.11 | n/a |
| q8_0 | 1 | 30.24 ± 0.60 | 23.25 | 1214 / 1849 (65.7%) |
| q8_0 | 2 | 31.76 ± 1.34 | 22.45 | 1560 / 3001 (52.0%) |
| q8_0 | 3 | 31.63 ± 1.08 | 22.53 | 1754 / 3914 (44.8%) |

The f16 two-draft setting leads this initial sweep: +29.5% generation and
-15.4% warm request time versus non-MTP at the same batch settings. Three
drafts helps the investigation workload but slows reasoning. Acceptance
percentage alone does not identify the fastest horizon. Q8 two/three greedy
drafts are nearly tied; two leads by only 0.4%, not a universal optimum.
The tuned sampling and deeper-context comparisons follow below.

All completed samples stop at the 512-token output limit without input
truncation; they are **bounded throughput**, not complete answers, verified
code outputs, cold swaps, or whole-agent task scores. Separate bounded
complete-answer checks use low effort and explicitly check final answers.

### Matched old/new MTP control

On the old October 1 build, Swift f16 with two greedy drafts and the same
4096/1024 batching averaged 32.59 ± 0.80 t/s and 21.86 seconds, versus
32.18 ± 0.93 t/s and 22.21 seconds on the candidate. Input prompt hashes
match for all three workloads. Generated output/draft acceptance varies
despite the fixed seed, including across builds; do not read this small
difference as a conclusive source-only regression or speedup. The large
initial MTP gain is already available on the baseline build.

### New sampling mode at the selected microbatch

Matched candidate-only tests at 4096/2048, two drafts, same three prompts and
six measured samples per row. This isolates draft sampling from the
microbatch change.

| KV | Draft sampling | Generation t/s | Warm request seconds | Accepted / generated |
|---|---|---:|---:|---:|
| f16 | greedy | 32.39 ± 0.42 | 21.44 | 1547 / 3021 (51.2%) |
| f16 | probabilistic | 35.46 ± 0.44 | 20.06 | 1679 / 2761 (60.8%) |
| q8_0 | greedy | 31.85 ± 1.53 | 21.75 | 1564 / 2993 (52.3%) |
| q8_0 | probabilistic | 33.35 ± 1.44 | 21.03 | 1634 / 2859 (57.2%) |

Probabilistic f16 generation improves 9.5% over greedy, and warm request time
falls 6.4%, without changing the target sampler. This is a useful new-build
feature gain, distinct from the essentially flat direct SYCL source controls.
Do not extrapolate its magnitude to full tasks or deep contexts; the separate
tool/JSON compatibility and near-limit memory checks are reported below.

Q8 improves 4.7% generation and 3.3% warm request time in the aggregate,
but reasoning is nearly flat: the benefit is workload-dependent and smaller
than f16. One/three-draft probabilistic controls bracket the selected horizon:

| Maximum probabilistic drafts | f16 generation | f16 wall seconds | q8_0 generation | q8_0 wall seconds |
|---:|---:|---:|---:|---:|
| 1 | 32.42 ± 0.60 | 21.42 | 31.63 ± 0.80 | 21.85 |
| 2 | 35.46 ± 0.44 | 20.06 | 33.35 ± 1.44 | 21.03 |
| 3 | 33.58 ± 2.17 | 20.94 | 33.01 ± 1.31 | 21.19 |

Select two on both cache types. The q8 two/three difference is only about 1%,
and three is better on the reasoning workload while two wins coding and
investigation; this choice is workload-scoped, not statistically established
as a universal optimum. The shorter horizon avoids unnecessary extra drafting.

### Deep real-text control: generation gains are not whole-request gains

The same investigation workload with 1536 padding modules has 67,158 input
tokens. On the candidate's full 128K f16 allocation with the projector loaded,
4096/2048 batching, default xhigh effort, and published sampling, compare
non-MTP to the selected two-draft probabilistic mode. Each row has a discarded
warmup and two uncached measured 256-token outputs; identical prompt hashes,
zero cache reuse, and no truncation. These outputs hit the token cap, not
complete task answers.

| Mode | Generation t/s | Prompt seconds | Generation seconds | Whole warm request seconds |
|---|---:|---:|---:|---:|
| non-MTP | 14.59 ± 0.004 | 74.11 | 17.47 | 91.65 |
| two probabilistic drafts | 26.72 ± 0.36 | 86.33 | 9.55 | 95.96 |

MTP raises generation by **83.1%** and cuts decoding time by 7.93 seconds,
but adds 12.23 seconds of prompt ingestion. The total uncached request is
**4.7% slower** at this short output cap. Long fresh inputs with short outputs
can therefore favor non-MTP; output-heavy or cached iterative work can favor
MTP. This experiment does not measure a cached deep request, a longer output,
or q8 at this depth, so it does not establish their crossover. The configured
plain Swift alias is a 256K q8 fallback, not the f16 non-MTP control in this table.

## Correctness and integration checks

Selected native core operations passed **3177/3177** supported cases on SYCL0:
ADD, L2_NORM, RMS_NORM, RMS_NORM_SCALE, MUL_MAT, GLU, SSM_CONV,
SSM_CONV_BIAS_SILU, and TOP_K. Dense unpermuted head-width-256 flash attention
with f16/q8 KV passed **119/119**. These are selected supported cases, not all
backend operations. The expanded upstream matrix differs from October 1's
2037/117 counts.

The specific wider sparse-mask case that failed in the prior pass still fails:
head widths 256, `nr23=[12,2]`, KV 8192, batch 67, f16 K/V,
`kv_view=1`, `n_kv_max=512`; **0/1**, normalized error 0.996637082 against
0.0005 tolerance. It is not the dense single-slot production path. No sparse
attention option is enabled; the failure is neither concealed nor claimed fixed.

The Swift f16 production profile retained its actual 131072-token capacity,
one slot, loaded projector, and full 66/66 GPU layer offload. A 126976-token
synthetic request generated 32 tokens without truncation or reused prompt KV;
no `using host memory fallback` warning appeared. It had 4497 MiB free before
cleanup (27.5 GiB whole-device use). Ordinary SYCL_Host output/staging buffers
are not evidence of device-allocation fallback. Startup fit estimates with
negative unaccounted memory are not used as final VRAM observations.

Swift q8/MTP retained its full 262144-token capacity and 66/66 GPU layer offload.
A 253952-token synthetic request generated 32 tokens without truncation, prompt
reuse, or a host-fallback warning. Plain Swift retained the full 256K allocation
and 65/65 non-MTP GPU layer offload; it received a shallow vision/tool smoke,
not a separate deep fill. All three recognized the llama logo through their
configured Swift projector and completed the forced-tool/result round trip.

| Swift profile | Text-stress input tokens | Whole-device use before cleanup | Free before cleanup | Target self-accounting: model / context / compute MiB |
|---|---:|---:|---:|---:|
| 128K f16/MTP | 126976 | 27.5 GiB | 4497 MiB (4.4 GiB) | 14912 / 8640 / 1080 |
| 256K q8/MTP | 253952 | 30.3 GiB | 1611 MiB (1.6 GiB) | 14912 / 9152 / 2496 |
| 256K q8 plain | shallow only | 27.2 GiB | 4790 MiB (4.7 GiB) | 14685 / 8853 / 2496 |

Whole-device use is `32656 - free` MiB, not just the printed target model/context
buffers. The remaining allocations include drafter/projector and driver/background
usage. These are end-of-run observations, not peak or guaranteed headroom.
Vision was tested before the text fill, not with an image added at near-full KV;
video was not tested. Q8/MTP fits but is a tight capacity profile. Prefer f16 if
128K is enough; keep other GPU work paused, use the plain fallback, or lower
`-ub` to 1024 if extra headroom is needed. Do not infer image-at-256K safety.

### Bounded complete-answer comparison

Three toy tasks (retry time, Unicode normalization/deduplication, LRU state),
low effort, temperature 1, published sampler, JSON-object output, seed 42,
2048-token cap, no prompt reuse. One discarded warmup plus two measured answers
per task on already loaded production profiles. Both scored **6/6** exact final
JSON answers with normal stop; all twelve finished below the cap.

| Tuned production profile | Correct / measured | Mean request seconds | Mean reported completion tokens |
|---|---:|---:|---:|
| Swift 128K f16, two probabilistic drafts, 4096/2048 | 6 / 6 | 7.31 | 267 |
| Base Qwen 128K f16, two greedy drafts, 4096/1024 | 6 / 6 | 7.21 | 279 |

User-instruction hashes match per task in `suite=toy-json-v2` (not full serialized
template prompts). The tuned profiles differ
in weights/template/batching/draft sampling; this is not a pure fine-tuning or
source-only comparison. Swift used 4.3% fewer completion tokens, but was 1.3%
slower in mean wall time: essentially equal latency in this small sample,
not a demonstrated complete-task speedup. Completion usage includes reasoning
and final output; separate reasoning-token counts were not reported.

Six original Swift rows are retained too. Their retry prompt did not explicitly
name the JSON key; both retry answers computed 18 correctly but used
`completion_time`, failing the harness's expected `seconds` key (4/6 strict
matches). The instruction was clarified, not the expected numerical answer,
and the full suite was repeated for Swift and matched Qwen. These original rows
are not mixed into the table above. This is bounded integration/answer checking,
not broad model-quality or complete-agent evaluation.

### Live proxy and client checks

Full forced-tool/result round trips passed through llama-swap for all three Swift
aliases, Qwen thinking, and Agents-A1. A final Swift thinking repeat passed and
left it warm, subject to the normal 600-second idle TTL. The historical
READY-2415 tool sentinel is a fixture, not a version assertion: deployed native
version and response fingerprints identify build 2546 / 18b5f8b18.

The live registry exposes 14 aliases. Generated repository Pi/OpenCode snapshots
match it, retain actual 128K/256K contexts, and preserve all eleven prior client
entries and OpenCode's default. Installed Pi 0.85.1 accepted the generated schema;
nine actual SDK payload checks verified system-role serialization and high→xhigh,
minimal→low, off→none across Swift's three profiles, without network inference.
OpenCode 1.18.30 uses explicit variants. Full interactive agent evaluations were
not run. Home client configs were not changed: run `./bin/llm-swap configure`
to synchronize them while preserving existing default selections.

## Deployment and rollback

The deployed process's resolved executable and every mapped llama/ggml library
come from `build-sycl-20261007`. Live arguments include `-c 131072 -b 4096
-ub 2048 --spec-draft-n-max 2 --spec-draft-sampling probabilistic` on Swift
thinking; actual props confirm one slot and vision enabled. No inherited
`GGML_SYCL_*` overrides were present; runtime graphs remain disabled by default
and normal oneDNN dispatch is retained. llama-swap `/health` returned OK.
All eleven pre-existing YAML profiles remain structurally identical to Git HEAD.

Retained local recovery artifacts:

- `llama.cpp/build-sycl-20261001/`: intact preceding native build.
- `llama.cpp/build-link-before-20261007`: saved old stable symlink.
- `llama.cpp/build-profile-20261007/pre-probabilistic-swift-profiles.yaml`:
  old-build-compatible config retaining all 14 aliases, Swift 4096/2048,
  two greedy drafts, and no unsupported sampling flag.
- The earlier October 1 original/September 12 snapshots remain intact too.

Rollback must restore both the binary link and compatible config. First preserve
any later edits; these explicit commands save the current YAML and move links
instead of deleting binaries. They leave the source checkout at the new commit.

```bash
cd /home/jeff/Code/intel
test -L llama.cpp/build
test "$(readlink llama.cpp/build)" = build-sycl-20261007
test -L llama.cpp/build-link-before-20261007
test "$(readlink llama.cpp/build-link-before-20261007)" = build-sycl-20261001
test -d llama.cpp/build-sycl-20261001
test -f llama.cpp/build-profile-20261007/pre-probabilistic-swift-profiles.yaml
test ! -e llama.cpp/build-updated-link-20261007
test ! -L llama.cpp/build-updated-link-20261007
task_rollback_save=$(mktemp -d /tmp/swift-rollback-20261007.XXXXXX)
cp llama-swap.yaml "$task_rollback_save/llama-swap.yaml"
systemctl --user stop llama-swap.service
mv llama.cpp/build llama.cpp/build-updated-link-20261007
mv llama.cpp/build-link-before-20261007 llama.cpp/build
cp llama.cpp/build-profile-20261007/pre-probabilistic-swift-profiles.yaml llama-swap.yaml
systemctl --user start llama-swap.service
```

The repository was on main at `8f422a9`; GitHub main still matched that commit
at final review. Documentation, summaries, tuning parameters, client snapshots,
and helpers are prepared as a local diff. No commit or push was performed;
weights, binaries, raw logs/responses, and temporary auth files are excluded.

## Reproduction

Prerequisites: the dated old/new native builds, installed oneAPI, local verified
weights/projector, and no competing GPU model workload. Run from the repository
root, sequentially. Each dated runner pauses/restores llama-swap, initializes
oneAPI only if needed, and isolates the selected build's llama/ggml libraries.

```bash
bash scripts/run_swift_profile.sh
bash scripts/recheck_swift_profile.sh
bash scripts/tune_swift_sampling.sh 2 2 4096 2048
bash scripts/validate_swift_profile.sh
```

The committed [historical Swift fixture](initial-swift-profiles.yaml) exactly
matches the old-build/initial greedy MTP inputs. It is not the live registry;
the old build cannot parse `--spec-draft-sampling`. Fresh runs should use a
separate results/raw directory or `--repeat` deliberately: saved measurements
and completed MTP groups otherwise resume/skip, not silently add replicates.
`results.jsonl` records actual commands, sampling modes, prompt hashes, token
caps, cache reuse, and truncation. Raw responses/logs stay in ignored
`llama.cpp/build-profile-20261007`.

The separate deep real-text comparison was:

```bash
source /opt/intel/oneapi/setvars.sh > /dev/null
# Pause other GPU workloads; restore the service even if profiling fails.
trap 'systemctl --user start llama-swap.service' EXIT
systemctl --user stop llama-swap.service
python3 scripts/profile_sycl.py mtp --build llama.cpp/build-sycl-20261007 \
  --label candidate-deep --kv f16 --alias swift-1.5-27b-think \
  --horizons 0,2 --draft-sampling probabilistic --padding-modules 1536 \
  --workloads agent --tokens 256 --batch 4096 --ubatch 2048 \
  --results docs/benchmarks/2026-10-07/results.jsonl \
  --raw llama.cpp/build-profile-20261007
```

CPU-only checks: `python3 -m unittest discover -s scripts -p test_profile_sycl.py`
(17 tests), `bash -n` on the dated runners and `bin/llm-swap`, Python compilation,
and `git diff --check`.

Long complete-agent evaluations and broad model-quality benchmarks are outside this scoped tuning pass. Swift's publisher reports token-efficiency improvements; local per-token throughput must be measured independently, and bounded reasoning throughput does not establish whole-task speedup.
