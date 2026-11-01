# Colab FP16 anchor failure — 31 October 2026

## Confirmed cause and evidence

The supplied T4 log is copied unchanged to
`outputs/anchor-fp16-recovery-20261031/provided-colab-log.txt`; its eight FILE
sections are also saved under `provided/`. The experiment JSON excerpt starts
partway through the file and does not include the parent's error. The process
log is empty. An original Python traceback was not provided and cannot be
recovered from those excerpts. The failure condition is confirmed by the source
snapshot and reproduced locally; newly generated tracebacks are retained.

The notebook cell4 calls `Session.benchmark_batch_sizes`, which calls
`batch_experiments.start` → parent `run` → a subprocess `worker` → shared
`coordinate.training.train` → `coordinate.engine.train`. A2 needs a disposable
A1 dependency; B also needs A2. These anchors have two **successful optimizer
updates** as their goal. The old parent supplied `attempts=2`, passed by the
worker as `stop_after=2`. That argument deliberately counts attempts, for
fixed-example efficiency comparisons. The engine already advances progress and
learning rate only on successful updates.

Both supplied A1 attempts had finite losses (1.7789891 and 2.3169098) but invalid
gradients, so GradScaler skipped them and backed off 65536 → 32768 → 16384.
Successful progress stayed zero. The trainer saved an incomplete `resume.pt`,
but correctly did not export `autoencoder.pt`. The parent's exact failing branch
was `if not path.exists(): raise ValueError('Disposable anchor failed: '+str(result))`.
The worker called the two rows `insufficient_measurements` because it applied a
five-row throughput rule to an anchor. The parent failed before any candidate
launch; its exception handler did not update pending rows or save a traceback.
Cell4 did not print its top-level `error`.

The old `numerically_finite` predicate was finite loss AND
`(gradients_finite OR optimizer_update_skipped)`. This reported handled invalid
gradients as finite. It did **not** establish successful training.

## Numerical audit and correction

The supplied train-fitted normalization scales are finite, nonzero: heat u
0.31696025; wave u 0.42403877; wave v 3.24093127. The recorded losses and components
are finite. This gives no evidence for changing normalization or the objective.
The two attempts alone cannot distinguish loss-scaler startup overflow from
persistent FP16 backpropagation instability. CUDA is unavailable on this Mac;
we do not claim to have identified the originating T4 operation or demonstrated
T4 recovery.

The shared path retains FP32 field subtraction, detached reference weights and
FP32 weighted reductions; A1 retains equal per-example/per-quantity reductions,
scalar reconstruction and latent regularization. A2 keeps its decoded field
objective, alignment/composition and graph terms. B keeps latent velocity flow
matching distinct. Region balancing (alpha=.5), arbitrary continuous times,
architecture, point counts, family mixture and precision policy are unchanged.
Forward autocast surrounds neural work; backward is outside autocast. Each
microbatch loss is divided by the effective batch, accumulated, unscaled once,
then clipped and stepped once. T4 auto remains FP16 + GradScaler, not emulated
BF16. No scaler or optimizer is recreated between attempted updates.

Changes:

- An anchor now declares its role, required successes=2 and maximum attempts
  `2 * max_consecutive_overflows` (16 with the existing default8). This bound is
  derived: before N successes, a run with fewer than L consecutive skips cannot
  consume more than N*L attempts. The existing eight-consecutive-skip failure is
  unchanged. This is not an unlimited retry or a higher numerical-failure limit.
- Anchor completion requires completed training and two successes, not five
  timing samples. Cancellation and the parent's wall cap still bound the job.
  Ordinary throughput trials retain fixed attempted-example budgets and at least
  five measured attempts. Skipped work is never a successful update; scheduler,
  scaler/RNG/optimizer checkpoint and resume counters retain their semantics.
- Nonfinite loss fails before optimizer mutation. Norm clipping raises before
  modifying invalid gradients. FP16 element overflow is safely skipped by
  GradScaler. A norm overflow with finite elements fails before stepping, since
  GradScaler would not detect that overflow. Other clipping/kernel errors propagate.
- Overflow history records the unscaled, unclipped invalid parameter names/counts
  (first12 tensors), scale before/after and loss finiteness. Diagnostics incur
  extra device inspection only on a skipped attempt. No NaNs are replaced or
  concealed. Persistent skips save `resume.pt`, `history.json`, `run.json` and
  diagnostic details before raising.
- `numerically_finite` now requires finite losses **and gradients** on all recorded
  attempts. Separate fields report handled skips and nonfinite-gradient attempts.
  An anchor may complete after safe startup skips while this flag remains false;
  its two successful updates must still be finite. Recommendations continue to
  exclude skipped/unstable candidates, including skips during warmup.
- Worker setup and training exceptions retain type, traceback and a sidecar
  `error-traceback.txt`, also printed to process.log. The parent stores anchor
  results and its traceback; unlaunched trials become `blocked` with the cause.
  Cell4 prints errors, anchor attempts/successes/scales, last gradient diagnostics,
  worker traceback and process-log path directly through shared `failure_details`.

## Reload the exact source in Colab

1. Download the new `pdeflow-fp16-anchor-source.zip` and `colab.ipynb` from
   `outputs/anchor-fp16-recovery-20261031/`. Open the **new notebook** in Colab.
2. Stop any old experiment through its notebook cancellation control before
   restarting the runtime. Keep its files, including failed experiment
   `b8c73f80fd214f37a8578e49a5d743da`; do not retry into its directory.
3. Choose Runtime → Restart session. In cell1 leave `REPOSITORY_URL=""` and
   `REVISION=""`, set `SOURCE_ARCHIVE="/content/pdeflow-fp16-anchor-source.zip"`
   after uploading it, or leave empty to use the uploader. Run cell1. It hashes
   the ZIP, extracts into a new hash-named source cache, validates cached bytes
   and installs without replacing the runtime's PyTorch/CUDA stack. Compare its
   printed SHA256 with the delivered sidecar. No old cache needs deletion.
4. Keep `RUN_TRAINING=False`, `RUN_PREFLIGHT=False`,
   `RUN_SOURCE_COMPARISON=False`, `RUN_BATCH_EXPERIMENT=False` and
   `APPLY_SELECTED_BATCH_SETTINGS=False` while verifying. Preserve your existing
   training settings; do not enable the failed-representation-gate override.

If the disposable Colab filesystem was lost, upload the saved original
`anchors/autoencoder/job.json` along with the source ZIP. The next check uses its
actual requested configuration, not a smaller preset. It needs no old model.

## Short T4 verification (not executed locally)

After cell1, run this additional cell. It copies the original Medium anchor job
into a **fresh** directory. The two-update anchor is diagnostic only. The parent
experiment's separate fixed-effective256 candidates are not launched here.

```python
import json, sys, subprocess, uuid
from pathlib import Path
import torch
from pdeflow.coordinate.contracts import CoordinateConfig
from pdeflow.coordinate.optimization import options
from pdeflow import batch_experiments

original = Path('/content/pdeflow-training/combined-medium-01/batch-experiments/b8c73f80fd214f37a8578e49a5d743da/anchors/autoencoder/job.json')
job = json.loads(original.read_text())  # adjust only this path if uploaded elsewhere
c = CoordinateConfig(**job['config']).validate()
assert torch.cuda.is_available() and c.runtime['device'] == 'cuda'
assert c.ae_steps == 2 and c.training_options['precision'] == 'fp16'
assert c.field_loss['type'] == 'region_balanced_mse' and c.field_loss['alpha'] == .5
root = original.parent.parent.parent / ('fp16-verification-' + uuid.uuid4().hex)
root.mkdir(exist_ok=False)
job.update(root=str(root/'stage'), role='anchor', successful_updates=2,
           attempts=2*options(c)['max_consecutive_overflows'], warmup=0,
           stop_file=str(root/'stop-request.json'), wall_seconds=180)
(root/'job.json').write_text(json.dumps(job, indent=2))
print('GPU:', torch.cuda.get_device_name(0), 'artifacts:', root)
subprocess.run([sys.executable, '-m', 'pdeflow.batch_experiments',
                'worker', str(root/'job.json')], check=True)
r = json.loads((root/'stage/result.json').read_text())
print(batch_experiments.failure_details({'anchors': {'autoencoder': r}}))
print({k:r.get(k) for k in ['status','successful_updates','attempted_updates',
      'skipped_updates','numerically_finite','nonfinite_gradient_attempts','scales']})
assert r['status'] == 'completed', 'Inspect printed traceback and saved artifacts; do not start training'
assert r['successful_updates'] == 2
successful = [x for x in r['history'] if not x['optimizer_update_skipped']]
assert len(successful) == 2 and all(x['gradients_finite'] for x in successful)
```

`wall_seconds` is checked between updates; setup/final export can add time. For
external cancellation, write `root/stop-request.json` from a second runtime
client or interrupt the cell. No retries reset the scaler.

If it completes after the first two skips, this establishes startup recovery
**for this measured T4 workload**. If it fails after eight skips, save the whole
new verification directory, especially result.json, resume.pt, run.json,
history.json and error-traceback.txt. Parameter names identify the affected
backward branches; a same-state FP32 diagnostic/anomaly trace is the next bounded
investigation. Do not classify persistent invalid gradients as startup or lower
precision/objective/query budgets to make the check pass.

Only after this succeeds, configure the intended Medium joint settings in the
new notebook cell3 (match the original advanced settings), and use cell4 for
one bounded integration experiment: A2/evolution only, candidate8, fixed
effective8, cold only via the shared call below. This exercises parent→A1 anchor
→A2 candidate; it is not the original effective256 sweep or a speed comparison.

```python
r = session.benchmark_batch_sizes(background=False, mode='efficiency',
    stages=['evolution'], candidates=[8], effective_batch=8,
    cache_conditions=['cold'], warmup=2, measured=5, max_seconds=180)
print('Experiment:', r['id'], 'state:', r['state'])
print(batch_experiments.failure_details(r))
for row in r['rows']:
    print({k:row.get(k) for k in ['stage','status','attempted_updates',
          'successful_updates','skipped_updates','numerically_finite','error']})
```

A candidate with startup skips stays ineligible for automatic recommendation;
no settings are applied by these calls. Do not launch the full36-row sweep until
examining these bounded results. Existing checkpoints and scientific outputs
remain untouched.

## Local verification and its limits

See `outputs/anchor-fp16-recovery-20261031/` for copied input, original source
snapshot, test logs, exact reproduction/recovery histories, persistent-failure
traceback and real subprocess A1/A2 anchor + A2/B trial records. CPU GradScaler
checks inject invalid gradients into otherwise real CPU FP32 training; they are
not CUDA FP16 evidence. Native MPS FP32 three-stage execution and CPU batched vs
accumulated loss/gradient equivalence are checked separately. Guarded native
CUDA tests are included and skipped here. No substantial training, new accuracy
claim, gate bypass or architecture/scientific-objective change was made.
