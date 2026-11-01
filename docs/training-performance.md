# Heat/wave training performance recovery — 31 October 2026

Later update: [actual A1/A2 batching and frozen preparation](batched-training.md)
supersedes the serial A1/A2 limitation described below. The measurements and
archive paths on this page remain the historical reference-optimization record.

The reported Colab T4 medians were A1 39.335 s, A2 88.183 s and B 23.687 s.
These are user-observed measurements, not measurements reproduced on this Mac.
The 175-day projection should be replaced only after profiling the updated source
on the allocated GPU with the original requested configuration. No CUDA device
is available in this session. No substantial training was launched.

## What changed without changing the scientific workload

- Generated fields were explicitly band-limited already, but each reference query
  evaluated the full reference-grid Fourier spectrum, including numerical tails.
  Generated examples now retain the declared modes. The constructor checks that
  the sum of discarded absolute u/v coefficients is at most 1e-12 times the larger
  of one and initial u/v amplitude; otherwise it rejects the declared bandlimit.
  Imported/unrestricted references retain their complete spectrum. This is not a
  lower reference grid, smaller requested frequency band, or new interpolation.
- Nyquist splitting is vectorized. Wave u/v share phases; batched heat/wave
  references evaluate independent fields/times in bounded chunks. Mean wave
  displacement remains u_mean(0) + t v_mean(0), and mean velocity stays constant.
  No full space–time arrays or fixed replacement frame set are created.
- Each solver owns immutable input/coefficient arrays and at most eight cached
  coefficient pairs and eight read-only FFT frames, keyed by exact time, parameter,
  quantity and shape as appropriate. Continuous times remain continuous. A
  reference LRU defaults to 32 examples. Chunking bounds total batch-query-mode
  temporary arrays; current batch and immutable spectra are additional memory.
- Normalization calibration traverses each realization once, sharing evaluation
  across its four existing calibration times. The calibration population is not
  reduced. Final frozen latent calibration uses the configured microbatch.
- Region analysis remains reference-only, periodic, signed and per supervised
  physical time/quantity. The analysis cache includes realization/quantity,
  parameter, exact time, geometry, loss/detection settings and normalization scale.
  FFT frames/coefficients are reused; arbitrary target coordinates are still
  evaluated spectrally rather than approximated by analysis-grid interpolation.
  Zero-weight padded slots reuse their identical first target instead of solving
  it again. Decoded query tensors, budgets and the stratified estimator stay intact.
- A2 shares one decoded-time evolved state across u/v. Frozen encoder outputs have
  a bounded CPU LRU (128 entries by default), keyed by dependency/encoder hash,
  dataset/coordinate sampling identity, normalization, exact physical time,
  state schema, source, resolved precision and autocast state. No trainable A1
  encoder outputs are cached. Random continuously sampled A2 times are never
  quantized; cache hit rates can be low and are reported.
- B caches deterministic t=0 targets and frozen family graph vectors. Its local
  disk target cache stops growing at 128 entries by default; existing cache files
  are not deleted. AMP and FP32 targets have separate filenames. Conditioning
  locations, physical times, masks/disclosure, noise and flow time remain fresh.
  Optional decoded B auxiliaries retain their original serial path.
- Default B CFM now uses actual configured forward microbatches and accumulates
  their summed losses divided by the effective batch. A1/A2 still accumulate
  one realization at a time; their microbatch setting is an upper bound, not
  a claim of vectorized neural execution. No automatic microbatch increase or
  architecture shrink is applied. The profiler returns an explicit OOM proposal.
- Family rejection examines deterministic family identity before constructing
  references. Optional 0–4 workers prepare the selected initial references in a
  bounded batch; worker results preserve index order. This is preparation
  parallelism, not a claim of complete CPU-target/GPU-compute overlap. No
  unbounded prefetch queue is used. Default workers remain zero until measured.
- Observation padding happens on CPU before transfer. CUDA uses pinned state/
  observation buffers and asynchronous transfers; CPU/MPS use blocking transfers.
  Per-quantity CUDA scalar checks were replaced by CPU weight validation and
  per-update finite loss/gradient checks. Detached logging scalars are collected
  at the optimizer boundary. No plots are generated per update.
- Disposable latent targets are excluded from future persistent snapshots.
  Existing scientific files are untouched. Keep LOCAL_ROOT on runtime-local SSD;
  Drive belongs only in PERSISTENT_ROOT. Session sync remains checkpoint-boundary
  and interval-controlled, not per update. Manual selection of a Drive-backed
  local root defeats this policy and should be avoided.

The architecture, spatial/physical-time support, model capacity, region-balanced
MSE alpha=.5, supervision/observation distributions, point counts and B velocity
objective are unchanged. Region-balanced loss still updates A1 encoder/decoder;
A2 uses that field objective through learned evolution with A1 frozen. These
optimizations do not improve the scientific accuracy of existing smoke weights.

## Precision and timing contract

PyTorch's default BF16 capability query includes emulation. We explicitly query
native support (`including_emulation=False`, or capability >=8 on older APIs).
T4 capability7.5 therefore resolves auto to FP16 + GradScaler. Explicit emulated
BF16 is rejected rather than silently run slowly. Hardware metadata distinguishes
native support from support including emulation. See the
[PyTorch capability API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html).
Graph message aggregation casts messages to the accumulator dtype; field/CFM
reductions retain FP32. Nonfinite FP32/BF16 gradients fail. FP16 overflow skips
that optimizer update and reduces the scaler; skipped updates, scales and gradient
finiteness are reported. They must not be hidden inside a successful throughput
claim. Actual T4 FP16 execution remains unverified locally.

Every history row has synchronized `step_seconds`, excluding validation and
persistence; `end_to_end_seconds` includes scheduled validation/persistence.
Setup (including dataset preparation), final calibration/export, validation and
persistence have separate clocks. Preflight uses two warmup + eight measured
updates per stage by default (at least five measured); it reports median, mean,
standard deviation, min/max and p10/p90. It preserves requested calibration,
architecture, point counts and scientific objective. Only the disposable probe's
update budget and checkpoint/validation frequency change. Additional post-profile
physical/reconstruction probes are explicitly small checks, not accuracy studies.

`component_timings` contains **per-update** host inclusive/exclusive durations,
CUDA stream intervals and separately named cumulative host totals. Nested scopes
such as region_supervision → reference_evaluation / region_analysis and
frozen_target → encoder_reference → reference_evaluation cannot be added together.
Exclusive host scopes partition instrumented host time. Their sum plus remaining
uninstrumented work and boundary GPU wait accounts for update wall time; it is not
an estimate of GPU kernel work. CUDA events surround compute scopes, with
synchronization only at profiling/update boundaries. Stream intervals include
stream scheduling gaps and must not be called sums of kernel durations. Transfer
clocks are host enqueue/copy costs, not pure DMA kernel durations. Repeated report
reads do not add cumulative counters twice. Cache hit/miss counters explicitly
separate per-step and cumulative counts. Validation events are outside training
component deltas.

Preflight reports CUDA peak allocated/reserved memory reset for each stage;
process-lifetime host peak RSS is separately labelled. It reports cache bytes,
population/I/O cost, setup and final export costs. No full-dataset warm cache is
claimed after a few random updates. Random large datasets can have few cache hits.
The rough duration remains compute-update-only; use all-in totals to assess setup,
validation, exports and persistent sync as well.

## Recorded local controls

Evidence: `outputs/performance-20261031/`. All are bounded execution checks.

Reference-only CPU control uses the large preset's 64² reference, 1024 encoder
points, 512 domain +256 region queries per supervised quantity, alpha=.5 and batch8.
Unit normalization is explicitly identical in both paths. Eight repeats per cache
condition, same fields/times/queries. This does **not** allocate the large network.

| Reference batch median | Full-spectrum control | Optimized |
|---|---:|---:|
| Cold | 4.08343 s | 0.07005 s |
| Warm | 4.02347 s | 0.06209 s |

All 44 compared FP32 arrays matched exactly. See
`requested-reference-benchmark.json`; distributions are retained. A separate
batched-reference test checks mixed heat/wave arbitrary times against serial
queries, including zero mode. This preprocessing result must not be extrapolated
to T4 training speed or a new days-to-completion estimate.

A full original-source versus optimized-source CPU control uses the same **smoke**
architecture and points in both runs: seven updates/stage, two warmup/five measured.
It does not claim to benchmark the user's large architecture on CPU.

| Stage | Before median/update | After median/update | Before / after examples/s (mean wall) |
|---|---:|---:|---:|
| A1 | .01279 s | .01254 s | 151.9 /158.5 |
| A2 | .04968 s | .04137 s | 36.7 /48.4 |
| B | .01284 s | .00946 s | 143.4 /213.5 |

Total stage wall times before/after: A1 .780/.780 s, A2 .526/.448 s,
B .239/.212 s. Original setup/export subdivisions were not instrumented and are
not invented. Updated setup times .522/.0417/.0442 s and final
calibration/export .0466/.0373/.0382 s are recorded. Updated B local target cache
was34,972 bytes. CPU has no CUDA peak memory; benchmark JSON records that as null.
These millisecond CPU results have limited statistical power. Maximum trained
parameter differences against the preserved original source: A1 zero, A2
1.19e-7, B1.59e-6; maximum loss difference2.38e-7 and gradient-norm difference2.38e-7.
See `before-after-final-numerics.json`. Focused tests separately compare the
region estimator/loss/prediction gradients and B microbatch1 versus2 updates.

The pre-optimization source is `source-before.zip`; it is the repository state at
this task's start, not a claim that its hash equals the archive in the user's
Colab runtime. Compare recorded source hashes if using another deployed baseline.
MPS non-blocking transfers initially produced nonfinite gradients; retaining
blocking CPU/MPS transfers fixed the real regression. Failures and repaired runs
are saved. CUDA and actual Colab/Drive execution remain untested here.

## Reload Colab and run the bounded same-workload comparison

1. Preserve the current run/checkpoints in their existing persistent location.
   Download the updated source ZIP and its SHA256 sidecar, and the updated
   `notebooks/colab.ipynb`. Open/upload that notebook in Colab (updating only the ZIP
   does not replace old notebook cells).
2. Restart the notebook kernel with **Runtime → Restart session** before installing
   another source. Do not try to hot-reload imported model modules. Keep existing
   source caches/run directories; the setup extracts the new ZIP under its hash.
3. In cell1 choose/upload the updated ZIP, leave repository URL empty for archive
   setup, and check the printed archive SHA256. The setup validates the complete
   source manifest and rejects modified cached files. Inspect `pdeflow.__file__` if
   uncertain about imports. An already-running process keeps its old imported code.
4. Set **RUN_TRAINING=False**. Keep the exact original requested architecture,
   family weights, point counts, loss, seed, optimizer, distributions and advanced
   settings in cell3. Use a separate benchmark name/local output. Do not resume
   an existing scientific run as a throughput experiment. New source hashes are
   incompatible with strict old-source resume by design; do not auto-enable a
   source-change override or change an old run's precision silently.
5. Cell4's updated `session.preflight(updates=10, warmup=2)` profiles the unchanged
   requested capacity. Inspect resolved native/emulated precision, skipped updates,
   peak allocated/reserved VRAM, setup/calibration time, variability and components.
   Cold/warm counters are actual, not inferred from warmup. An OOM does not shrink
   the model; explicitly revise microbatch/accumulation and profile again if needed.
6. For a before/after comparison, upload `source-before.zip`, extract it into a
   distinct baseline directory (e.g. `/content/pdeflow-before`), and set
   `BASELINE_SOURCE_ROOT` to its extracted `pdeflow/` folder in cell4. Set
   `RUN_SOURCE_COMPARISON=True`. The harness runs fresh subprocesses with the
   **same** resolved config and explicit FP16 on T4. It changes only ten probe
   updates and checkpoint/validation frequency. It does not run Session's training
   action or claim a passed representation gate. Full requested calibration is
   retained; the slow baseline/setup may take appreciable time. Compare both
   `benchmark.json` files rather than just the fastest update. Baseline AMP support
   is not assumed verified; any failure is a failure to record, not a speedup.

Equivalent commands after extraction, with `requested.json` written from the exact
notebook config (explicit `precision='fp16'` for BOTH T4 benchmark paths):

```python
from dataclasses import replace
from pdeflow.runtime import write_json
benchmark_config = replace(config, training_options={**config.training_options,
                                                    'precision': 'fp16'})
write_json('/content/requested.json', benchmark_config.to_dict())
```

```bash
python /content/UPDATED/pdeflow/scripts/benchmark_training.py \
  --source /content/BEFORE/pdeflow --config /content/requested.json \
  --output /content/benchmark-before --updates 10 --warmup 2
python /content/UPDATED/pdeflow/scripts/benchmark_training.py \
  --source /content/UPDATED/pdeflow --config /content/requested.json \
  --output /content/benchmark-after --updates 10 --warmup 2
```

Replace UPDATED/BEFORE with the actual extraction paths. Output directories must
be new. Configuration JSON is recorded unchanged alongside the explicit probe
configuration, source hashes, all per-step rows, effective throughput, peak memory,
cache sizes and setup/export costs. The original source lacks per-update all-in
and setup subdivisions; the harness records total stage wall time for both and
marks unavailable subdivisions null. Use these totals for the fair all-in check.
Before/after FP16 comparisons do not isolate the old BF16-emulation penalty;
precision-policy improvement is a separate expected benefit requiring T4 evidence.

For the independent cold/warm reference control in the updated source:

```bash
python scripts/benchmark_references.py --config /content/requested.json \
  --output /content/reference-benchmark.json --repeats 8
```

Next: run this bounded T4 comparison, inspect any scaler overflows/VRAM limits and
remaining compute time, then decide whether further A1/A2 batching or a bounded
CPU-supervision prefetcher is justified. Do not begin the 100k-stage budgets until
those measurements support the requested run. Poor smoke-model accuracy and the
representation gate remain unchanged.

## Verification and delivery record

Saved successful execution before the interrupted handoff:

| Check | Result | Evidence |
|---|---|---|
| Affected numerical/reference/region/precision/cache/three-stage/resume tests |85 passed,2 CUDA-only tests skipped | `verified-tests.txt` |
| Final focused performance checks after logging changes |26 passed | `final-focused.txt` |
| Actual six notebook cells, isolated local CPU/IPython |2 tests passed | `notebook-tests.txt`, `notebook-execution/notebook-run-all.log` |
| Real MPS A1 execution | Passed after blocking-transfer correction; included in affected tests | `mps-transfer-fix.txt`, `verified-tests.txt` |
| Original/optimized CPU controls | Completed; reused, not repeated during handoff recovery | `cpu-before/benchmark.json`, `cpu-final/benchmark.json` |
| CUDA/T4/actual Colab and Drive |Not executed in this session | Run the supplied bounded comparison |

The notebook execution predates the final logging-only counter additions, which
are covered by the final focused suite. No completed suite or training benchmark
was repeated during handoff recovery. Python compile, documentation/command checks,
archive hashes/contents and preservation verification finish the delivery.

Delivery files are `outputs/performance-20261031/pdeflow-performance-source.zip`,
its `.sha256` sidecar, a copy of the updated `colab.ipynb`, and the preserved
`source-before.zip` baseline with its sidecar. `package-verification.json` records
all manifest/member hashes and exclusions. `preservation-after.json` checks every
pre-existing checkpoint/array hash against the starting baseline; `completion.json`
records final source/Git/evidence status. No verification server was started.
The attempted native Chrome inspection produced no usable state and was interrupted;
it is not CUDA/Colab evidence and was not retried.

To extract the baseline safely in a Colab code cell (this does not install it into
the live kernel or change current model imports):

```python
from google.colab import files
from pathlib import Path
import hashlib, zipfile, json
uploaded = files.upload()  # choose source-before.zip only
assert len(uploaded) == 1
baseline_zip = Path(next(iter(uploaded)))
baseline_hash = hashlib.sha256(baseline_zip.read_bytes()).hexdigest()
baseline_dir = Path(LOCAL_ROOT) / ('baseline-source-' + baseline_hash[:16])
with zipfile.ZipFile(baseline_zip) as z:
    assert all(not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts
               for i in z.infolist())
    assert sum(i.file_size for i in z.infolist()) < 512 * 1024**2
    manifest = json.loads(z.read('pdeflow/SOURCE_MANIFEST.json'))
    for name, entry in manifest['files'].items():
        data = z.read('pdeflow/' + name)
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    if not baseline_dir.exists():
        z.extractall(baseline_dir)
BASELINE_SOURCE_ROOT = str(baseline_dir / 'pdeflow')
print('Baseline source:', BASELINE_SOURCE_ROOT, 'ZIP SHA256:', baseline_hash)
```

Then copy that printed root into cell4's BASELINE_SOURCE_ROOT field. Keep
RUN_TRAINING=False. To run only the source comparison, set RUN_PREFLIGHT=False
and RUN_SOURCE_COMPARISON=True; this avoids an additional standalone preflight.
For comparison metadata use the same original requested configuration, with both
paths explicitly resolving to FP16 on T4. Neither snapshot is installed over the
other; fresh subprocess imports select the requested source root.
