# Actual A1/A2/B batching and frozen-target preparation

The active configuration/application workflow is now documented in
[batch-settings.md](batch-settings.md). It supersedes the earlier separate sweep
instructions below; historical measurements remain attributable.

31 October 2026. This supersedes the earlier performance guide's statement that
A1/A2 execute one realization at a time. The reported Medium/T4 profile of about
.71/7.95/1.99 seconds per A1/A2/B update is user-provided evidence. This session has
CPU and MPS execution, not a connected T4 runtime. No new T4 timing is claimed.

## Execution and scientific reductions

`training_options.execution="batched"` is the default for new/existing-compatible
configurations on this source. `"accumulated"` retains the one-realization forward/
backward control. The configured `microbatch` is a real neural batch in A1/A2 and
default B. `accumulation=ceil(effective_batch/microbatch)`; a final shorter group is
weighted by its actual number of examples. History records the actual maximum
forward batch AND the complete list of forward batch sizes. Optional decoded B
auxiliaries retain the explicitly reported serial path; no objective is dropped.

A1 batches reference state encoding and decoding. A2 batches frozen encoder cache
misses and learned transitions. Mixed heat/wave batches use the model's existing
absent-state masks. Displacement is decoded for all examples; velocity is decoded
only for wave indices. Query budgets remain the requested domain + fixed region
queries, including zero-weight padded query slots. No query or meaningful region
is discarded. Each scalar/quantity error is reduced within its own realization;
heat field loss is L_u and wave field loss is (L_u+L_v)/2. A1 scalar and regularizer
terms retain their coefficients. A2 alignment, latent composition and decoded
composition each retain their previous per-example denominators and weights.

The optimizer objective remains sum_i L_i / effective_batch. Family sampling is
unchanged, so family probabilities affect the realization counts rather than a
new averaging over family groups. A wave realization does not acquire double
weight by having two quantities. Region-balanced MSE, alpha=.5, detached per-time
reference masks and the existing stratified estimator are unchanged. MSE, weighted
MSE and MAE also have strict loss/gradient comparisons. A2's graph embedding is
computed once per actual forward batch and shared by its transition/composition
calls; all gradients accumulate into the same trainable graph encoder. The state
encoder/coefficient modules remain frozen in A2. B CFM remains a separate loss.

No architecture, grid, point count, time distribution, observation distribution,
objective term, split, frequency coverage or training budget is silently reduced.
Existing weights/data/results are preserved. These changes are throughput work,
not evidence that smoke models have acceptable accuracy or pass the quality gate.

## Frozen preparation and instrumentation

The previous implementation called digest(cache_identity) for every encode.
That identity contains the full latent mean/scale arrays. A2 did so repeatedly
for source, target and initial states. It also scanned the target directory on
misses. The updated implementation hashes the immutable identity once at stage
setup and builds one disk inventory. Per-request keys are tuples of that prefix,
realization identity, exact physical time and autocast state. No model, full config,
or transform-array serialization belongs in a request key.

The prefix includes the dependency checkpoint hash (frozen encoder/coefficient
weights), scientific source, dataset recipe, context-coordinate sampling identity,
normalizations, latent transforms, state schema and resolved precision. Exact
float times distinguish arbitrary continuous times. Duplicate requests inside a
microbatch (for example A2 ta=0 and its initial-state request) are encoded once.
Changed dependency/data/normalization/coordinates/precision cannot reuse the old
cache identity. The dependency checkpoint is loaded and hashed once at stage
startup. A regression verifies those counts and the single large-identity digest.
Resolved config, source hashes and joint schema are also retained once per stage.

A bounded CPU LRU stores complete deterministic frozen encodings. Newly encoded
states remain on the device for the current batch, with one batched CPU copy for
the LRU; they are not copied CPU→GPU again. CPU cache hits are packed for one
batched transfer. Both A2 and B may persist t=0 encodings in the separately bounded
local target cache. It stores at most `frozen_cache_items` new entries (default128),
with a versioned prefix and safe tensor loading. Cached files from earlier accepted
runs are never deleted or rewritten. Noninitial continuous-time fields are NOT
materialized on disk, and times are never quantized. Initial-state cache keys
include physical inputs/coefficients through the immutable dataset realization.

At most one next reference chunk is prefetched by one CPU thread when
`frozen_prefetch=1` (default); `0` disables it for comparisons. The owner thread
alone runs PyTorch accelerator models. Immutable spectral coefficients make CPU
reference preparation safe to overlap. Memory is bounded by current/next reference
chunks, current request latents and the LRU/disk limits. Worker CPU time can overlap
compute and is explicitly excluded from additive host-wall accounting. Pool shutdown
joins/cancels outstanding work at stage exit, including exceptions/stopping.

Detailed per-update exclusive/inclusive host clocks and CUDA stream intervals
include:

- `frozen_key_construction`, `frozen_cache_lookup`;
- `frozen_reference_evaluation` without prefetch, or `frozen_reference_wait` plus
  separately recorded worker reference time with prefetch;
- `encoder_compute`, `frozen_output_transfer`, `frozen_input_transfer` and packing;
- `frozen_serialization`, `frozen_deserialization`, `frozen_disk_read/write`;
- existing reference/region/decoder/evolution/backward/validation/persistence clocks.

`frozen_preparation_step` contains per-step counters/worker seconds;
`frozen_preparation` explicitly labels cumulative stage counters, including cache
hits, misses, deduplications, encoder batch counts and maximum encoder batch.
Setup identity hashing and disk inventory have separate durations. CUDA timings
remain stream intervals, not a sum of kernel durations. No synchronization is
added for every small reference operation. No observations, noise, disclosure
masks or flow times are cached. Drive sync remains outside updates; keep the
local root on runtime SSD. Target caches remain excluded from persistent snapshots.

## Actual verification and local measurements

Evidence: `outputs/batching-20261031/`. No full training was run.

- `regressions.txt`:93 passed,2 CUDA-only checks skipped. This includes actual MPS
  A1 and numerical/reference/loss/resume coverage.
- `preparation-final-tests.txt`:35 passed after the cache-transfer/counter changes,
  including elementwise clipped parameter-gradient and optimizer comparisons.
- `accelerator-final.txt`:actual MPS A1/A2/B and disk-key check passed; CUDA skipped.
- `notebook-tests.txt`:two tests passed, including execution of all six modified
  notebook cells in the isolated local CPU/IPython environment. This is not
  actual Colab/Drive/CUDA execution.
- `gradient-evidence/`:two-update CPU accumulated/batched comparisons. Maximum
  elementwise clipped-gradient difference across all field objectives was4.77e-7.
  For region-balanced MSE, maximum optimizer parameter differences were A1
  2.40e-6, A2 1.79e-7, B 1.59e-6. MAE near-zero-gradient Adam updates differ by
  up to2.05e-5 despite gradient differences <=2.39e-7; exact bitwise equality across
  batch kernels is not promised. Same-mode CPU resume tests remain passing.

The local sweep uses SMOKE capacity, FP32, unchanged point counts across its
variants, effective batch32, two warmup +five measured updates per trial. Both
cache conditions use the same requested samples/RNG and common two-update A1/A2
anchor checkpoints. These anchors establish pipeline dependencies, not accuracy.
Every variant runs fresh in a subprocess to isolate memory peaks. Warm means a
deterministic replay after copying only valid t=0 target cache files; arbitrary
noninitial times still evaluate on demand. A1 has no frozen cache, so its warm
column is a replay, not a frozen-cache benefit. Warmup is separate from cache state.

| Actual forward batch | A1 cold/warm s/update | A2 cold/warm s/update | B cold/warm s/update |
|---|---:|---:|---:|
| Accumulated1 (same effective32) | .1545/.1515 | .5611/.7172 | .1434/.1112 |
| Batched8 | .0569/.0590 | .0922/.1243 | .0434/.0455 |
| Batched16 | .0511/.0490 | .0713/.0709 | .0387/.0409 |
| Batched32 | .0448/.0445 | .0586/.0585 | .0364/.0364 |

These are median **end-to-end update** times including scheduled validation/
checkpoint costs. Throughput below uses the mean end-to-end wall time, so it is
not32 divided by the displayed median. Full distributions/raw rows are retained.

| Actual forward batch | A1 cold/warm examples/s | A2 cold/warm examples/s | B cold/warm examples/s |
|---|---:|---:|---:|
| Accumulated1 |195/195 |55.7/45.1 |205/269 |
| Batched8 |480/459 |302/215 |597/589 |
| Batched16 |526/534 |382/381 |661/614 |
| Batched32 |589/576 |447/449 |692/700 |

Cold process peak host RSS was about249–263 MiB. This is a process high-water
mark; CUDA peak allocated/reserved is null locally. No T4 memory-fit claim follows.
Stage setup was about.42–.47 s. With batch32, seven-update exclusive host
preprocessing totals were about.085/.093/.018 s (A1/A2/B); overlapped frozen
reference worker time was separately.0110/.0007 s (A2/B). A2/B key construction
was about.00034/.00016 s TOTAL, not per encode. Local A2/B cache sizes were about
.022/.026 MiB because the smoke dataset has only12 realizations. Its cache-hit
rates must not be extrapolated to thousands of Medium training solutions.

Use `cpu-sweep/sweep.json` for all-in stage durations, separate setup/export costs,
cache/preprocessing costs, actual batch sizes, variability and raw rows. This sweep
preceded final per-step counter/device-transfer bookkeeping; focused tests cover
those subsequent changes. It is not a benchmark of the Medium architecture and
cannot replace the user-provided T4 numbers. No winning training batch or accuracy
claim is selected from it. Larger optimizer batches change training dynamics.

## Exact Colab reload and benchmark

1. Preserve existing runs/checkpoints. Open/upload the updated `colab.ipynb` and
   **Runtime → Restart session** before importing the updated source. Upload the
   new `pdeflow-batched-training-source.zip` in cell1; check its SHA256 sidecar and
   the manifest-validation output. New archive hashes create a distinct source
   directory; do not overwrite imported modules or existing source caches.
2. Cell3: reproduce the exact requested Medium model and advanced configuration
   (or load the original resolved JSON). Keep architecture, points, precision,
   families, all loss terms, seeds, splits and distributions. Set RUN_TRAINING=False.
   Do not resume a scientific run merely to benchmark; strict old-source resume
   still rejects a changed source fingerprint.
3. For only the batching sweep, cell4: RUN_PREFLIGHT=False,
   RUN_SOURCE_COMPARISON=False, RUN_MICROBATCH_SWEEP=True,
   SWEEP_EFFECTIVE_BATCH=32, SWEEP_MICROBATCHES="8 16 32". The original `config`
   and Session remain unchanged; the effective-batch override is recorded only
   in the probe. All variants use32 and the same resolved precision (FP16 on T4).
4. Inspect `sweep.json`, per-variant/per-cache `benchmark.json`, and job/log files.
   They record exact requested/executed configs, hardware/native precision,
   immutable dependencies, all-in wall time, examples/s, CUDA peaks, preprocessing,
   skipped scaler updates and cache identities/counts. An OOM is recorded for
   that variant; no architecture/points/objective shrink occurs. Non-OOM errors
   stop with the exact log path. Full requested calibration is preserved, so
   anchors/baseline can take time; this is a bounded profiling experiment.

For an EXACT effective-batch8 comparison with the user's current profile, use
SWEEP_EFFECTIVE_BATCH=8 and SWEEP_MICROBATCHES="1 2 4 8". Microbatch16/32 cannot
fit an optimizer batch of8; silently changing that batch would invalidate the
comparison. Do not compare32-example seconds/update directly with8-example
seconds/update. Use identical effective batch within a comparison and report
throughput separately. No preset/training configuration is automatically changed.

Equivalent code in the updated notebook, retaining the exact `config` object:

```python
import sys, subprocess, uuid
from pathlib import Path
from pdeflow.runtime import write_json
probe = Path(LOCAL_ROOT) / ('batch-sweep-' + uuid.uuid4().hex[:10])
requested_json = probe.with_suffix('.requested.json')
write_json(requested_json, config.to_dict())
subprocess.run([sys.executable,
    str(PACKAGE_ROOT / 'scripts/benchmark_microbatches.py'),
    '--config', str(requested_json), '--output', str(probe),
    '--effective-batch', '32', '--microbatches', '8', '16', '32',
    '--updates', '7', '--warmup', '2'], check=True)
print(probe / 'sweep.json')
```

Or from a terminal inside the extracted updated source:

```bash
python scripts/benchmark_microbatches.py --config /content/requested.json \
  --output /content/batch-sweep-01 --effective-batch 32 \
  --microbatches 8 16 32 --updates 7 --warmup 2
```

Output must be new. The parent creates common2-update random A1/A2 anchors, then
runs an accumulated control and each requested microbatch with cold/warm replays
(seven updates each stage/trial, not the configured full training budgets).
There is no production gate override. CPU tests verify full gradient vectors;
the sweep additionally reports loss/gradient-norm differences and family-count
agreement. Norm agreement alone is not full vector equivalence. Actual CUDA FP16
finite-gradient behavior and Medium8/16/32 memory/speed remain to be measured on T4.

The new source package, baseline source, manifest/hash verification, copied notebook,
preservation record and completion report are under `outputs/batching-20261031/`.
No verification server was started and no existing scientific artifact was removed.

Delivery verification is recorded in `package-verification.json`,
`preservation-after.json` and `completion.json` beside the source ZIP. The updated
archive includes both batching/cache modules, the new tests/sweep script, frontend
source/lockfile, project metadata, configs, notebook and documentation. It excludes
all output trees, environments, dependencies, caches, datasets, checkpoints and
user scientific reference files. A separate starting-source ZIP is retained for
reproducibility; neither accepted earlier archives nor scientific runs are replaced.
