> For active M1–M5 mesh training use [mesh training controls](mesh-training-controls.md). This guide records the retained factorized A1/A2/B workflow.

# Configure, benchmark, apply, then train

The training form and active Colab notebook now share batch resolution and an
explicit application API. Benchmark results do not train or reconfigure the actual
run. **Apply selected batch settings / Apply recommendation** changes the
configuration used by the next training action. Applying does not start training.
Architecture, dataset distribution, field loss (including region-balanced MSE
alpha=.5), observation/time sampling, point counts and learning rate are preserved.

## Three separate quantities

- **GPU microbatch:** independent examples in one neural forward/backward group.
- **Gradient accumulation:** microbatches contributing to one optimizer update.
- **Effective batch:** examples contributing to that update.

For complete groups, effective = microbatch × accumulation. For a remainder,
accumulation = ceil(effective/microbatch). Example: effective7, microbatch3 uses
[3,3,1], accumulation3. Each individual example loss is summed and divided by7;
the last group is not overweighted. This is accumulation, not a claim of seven
examples processed together on the GPU. Mixed heat/wave batches use the existing
per-example/per-quantity reductions. Wave velocity decodes use only wave indices;
actual quantity subbatch sizes and the full forward-group sizes are logged.

The common effective batch may have different A1/A2/B microbatches. Custom positive
values, including effective1024 and microbatch1024, are validated. There is no
memory-fit promise for1024. A large effective batch can use small neural batches
through accumulation. The authoritative per-stage summary shows all three values,
planned successful updates, final remainder size and approximate examples processed.

Auto is a documented conservative fallback: min(effective,8). It is **unqualified**
until measured, not an estimate of hardware throughput or memory capacity. Applying
a recommendation records explicit per-stage sizes and the benchmark provenance.

## Normal training UI

Open **Train model → Batch settings** below the ordinary scientific settings.

1. Choose Effective batch size and GPU microbatch Auto/Manual. Advanced per-stage
   overrides accept a positive number or auto; empty inherits the global choice.
2. Choose the budget policy, then **Apply selected batch settings**. Old → new
   values and the authoritative table update. Pending batch edits disable Train.
   This targets the next NEW training submission; it does not mutate a listed run.
3. Optionally open **Benchmark batch sizes**, choose stages (default A1 only), mode,
   automatic or editable candidates, and a runtime cap. **Estimate experiment**
   precedes launch. Then **Run batch benchmark**. Inspect progress, cancel, or
   choose a persisted result from Saved batch experiment after a page refresh.
4. **Apply recommendation** or select completed results per stage and **Apply
   selected result**. Unselected stage overrides are retained. Stale science/source/
   hardware configurations reject application. The result state distinguishes
   benchmark completion from application and actual training.
5. Review the table and press **Train model**. The saved request/configuration is
   passed to the worker. Training is blocked while the selected benchmark runs;
   backend/Session guards also prevent conflicting execution.

The normal screen stays compact; experiment controls and stage overrides are
collapsed. The editable candidate list can include8,16,32,64,128,256,512,1024 or
other compatible positive values. Explicit incompatible candidates fail validation.
No architecture, point count, region supervision or effective batch is silently
shrunk to make a candidate fit.

## Colab: exact sequence and reload

Download the new notebook AND source ZIP. Open/upload the notebook in Colab,
**Runtime → Restart session**, then upload the new ZIP in cell1. Check the printed
archive hash and adjacent SHA256 sidecar. The setup uses a hash-named extraction
and verifies its members; never overwrite an imported old source tree. Keep existing
scientific run directories and persistent snapshots. CUDA/actual Colab execution
of this change is not verified on this Mac.

After setup/storage, the sequence is:

1. **Cell3 Configure.** EFFECTIVE_BATCH is a positive integer; GPU_MICROBATCH is
   Auto/Manual, with MANUAL_GPU_MICROBATCH for the explicit size. A1/A2/B overrides
   are ordinary form fields: inherit, auto, or a positive number. No JSON or internal
   Session edits are required. Select BATCH_BUDGET_POLICY.
2. **Cell4 optional Benchmark.** Set RUN_BATCH_EXPERIMENT=True, selected stages,
   experiment mode, candidate mode/list and budget/cap. Auto tests at most three
   progressively increasing compatible sizes by default. RUN_IN_BACKGROUND=True
   returns promptly; rerun this cell with RUN_BATCH_EXPERIMENT=False to inspect
   persisted progress. CANCEL_BATCH_EXPERIMENT stops at an optimizer boundary
   where possible and retains partial results. Set RUN_PREFLIGHT independently;
   benchmarking a selection is not a mandatory whole-stage sweep.
3. **Cell5 Select/apply.** Choose Recommended result, Selected result, or Manual
   settings. For manual selection choose effective batch and each stage's size.
   Press **Apply selected batch settings**, or set the identically named form
   checkbox and run this cell when widgets are unavailable. It rebinds the real
   Session, saves settings, prints old → new, and refreshes the authoritative plan.
4. Review the run name, precision, per-stage batch values, successful-update budget
   and examples. If required, run cell4 with RUN_PREFLIGHT=True and
   RUN_BATCH_EXPERIMENT=False to replace the invalidated resource profile.
5. **Cell6 Train.** RUN_TRAINING remains False until you explicitly enable it.
   This action prints the authoritative Session summary immediately before launch.

After a result selection, rerun only Apply, the required preflight if applicable,
and Train. There is no need to rerun source/storage/data setup. Rerunning Configure
is an explicit new configuration choice. For an existing run, recorded settings
are retained unless APPLY_BATCH_CHANGES_ON_RESUME is selected and explicit
continuation is authorized. Larger model preflight guards and representation quality
gates remain in force; no failed scientific gate is automatically bypassed.

## Experiment modes

**GPU batching efficiency:** keep effective batch, ordered realizations, physical
and flow-time/observation/noise draws, model, precision, point counts, all objective
terms, seed and optimizer settings fixed. Vary only neural microbatch/accumulation.
Effective1024 can compare micro32/64/128/256/512/1024 with accumulation32/16/8/4/2/1.
Choose a smaller effective batch for a cheaper experiment. Microbatch cannot exceed
it. This tests execution efficiency, not whether a large optimizer batch learns
better. Any skipped steps disqualify a result from recommendation.

**Effective-batch learning comparison:** deliberately vary effective batch, keeping
architecture/distribution/scientific objective unchanged. Choose fixed examples,
fixed wall seconds, or fixed successful optimizer updates. Fixed examples uses
ceil(examples/effective) attempted batches, reporting actual exposure/rounding;
overflow skips do not hide extra example consumption. Fixed wall starts at worker
training setup and stops at an optimizer boundary; final persistence and held-out
validation are reported separately and can extend elapsed duration. Fixed updates
means equal successful updates, with unequal examples/attempt counts explicitly
reported. No automatic learning-rate scaling. Held-out validation, not the test
split, is evaluated consistently after the bounded probe. No automatic scientific
winner/recommendation is selected in learning mode.

Each candidate owns a subprocess, fresh model/optimizer/scheduler/scaler/RNG/sampler
state and disposable outputs. Two-update common frozen A1/A2 anchors are generated
only when needed by the selected stages. They are pipeline fixtures, not accuracy
models. A1-only experiments need no A2/B anchor. Actual run model/progress/RNG are
not advanced. Candidate initial scaler settings are identical. OOM is caught as
OOM, not as a generic numerical exception; other errors retain their type/message.
Process exit releases candidate memory before another candidate. Automatic search
stops larger candidates after a confirmed OOM/headroom boundary. Explicit candidate
lists report each result without changing the workload.

A pre-launch duration cannot be inferred from VRAM. A matching Session preflight can supply a rough measured-baseline extrapolation
(without assuming a batching speedup). Without one, the estimate explicitly says
unknown until a measured pilot, lists the experiment's runtime cap and trial count,
and becomes a measured remaining-time estimate after a completed candidate. Setup,
requested calibration, anchors, validation/export and boundary latency can be
substantial. Default candidate/stage subset and runtime cap bound exploration;
not every requested candidate is guaranteed to complete before the cap. Dead-worker
recovery retains completed rows and marks unfinished rows interrupted. Cancellation
is not permission to delete scientific state.

## Results and selection

The compact table records stage, microbatch, accumulation, effective batch,
examples/sec, median update seconds, peak CUDA memory, skips, cache condition and
status. Saved detailed records include mean/std/p10/p90, successful updates/sec,
actual full/quantity forward sizes, FP16 scale history, attempts/successes/examples,
preparation/setup, validation/checkpoint overhead, cache counts and identities,
source/hardware/configuration, errors and partial progress.

Cold means fresh candidate target cache. Warm means replay with valid immutable
initial-target cache files; arbitrary continuous times are not replaced with cached
frames. Warmup and steady-state rows are separate. A1 has no frozen-target benefit.
Recommendations use conservative throughput across all requested cache conditions,
no skipped/numerically invalid rows, bounded variability, and CUDA memory headroom
when measurable. Within5% throughput, lower measured memory/smaller microbatch is
preferred. These are explicit heuristics, not significance tests. CPU/MPS memory
headroom is unqualified when CUDA allocation/total memory is unavailable. A result
must complete all requested conditions before application; manual selection does
not turn an OOM or skipped-step candidate into an eligible configuration.

## Authoritative shared API and budgets

```python
from pdeflow.colab import apply_training_settings
session = apply_training_settings(session, {
    'effective_batch': 1024, 'microbatch': 'auto',
    'stages': {'A1': 32, 'A2': 64, 'B': 128},
}, budget_policy='keep_updates')
print(session.training_summary())
# No training has started. The next session.run(...) uses these settings.
```

Or after a shared experiment:

```python
result = session.benchmark_batch_sizes(
    stages=['autoencoder'], candidates=[8,16,32], effective_batch=32,
    warmup=2, measured=5, max_seconds=120, background=False)
session = session.apply_batch_result(result['id'], recommended=True,
                                     budget_policy='keep_examples')
```

The replacement preserves name, local/persistent roots, runtime/science settings,
sync policy and snapshot state. Application validates, archives old/new provenance,
invalidates old profile/resource estimates, writes `applied-settings.json` and
`resolved-config.json`, and returns the replacement Session. A stale old Session
binding is rejected by the next run. The UI uses the same shared config-application
logic and sends its returned request to the training worker.

Budget policies are explicit. Keep updates:10,000×8=80,000 examples becomes
10,000×1024=10,240,000 examples. Keep approximate examples:ceil(80,000/1024)=79
updates,80,896 examples,896 extra from rounding. The existing B minimum of two
updates is enforced and any additional exposure is recorded. Neither policy
promises equivalent optimization. LR is preserved; no batch-based LR scaling.
For new runs, implicit schedule horizons follow the resolved update budget;
explicit horizons are retained. For continuation, the recorded optimizer/scaler/
scheduler and horizon are preserved, with remaining-example conversion based on
saved progress. The application report and history retain old/new configurations.

An active worker rejects application. Existing runs require `continuation=True`
(or the notebook's explicit continuation control). Batch authorization and history
are recorded; original configuration provenance remains. Architecture/data/loss
compatibility and source checks still reject incompatible changes. Completed frozen
stages are retained when compatible; their original provenance is not rewritten.
Changing decomposition/effective batch does not promise bitwise-identical training.
Old checkpoints with ambiguous skipped-step progress require explicit migration or
their original source, not silent reinterpretation.

## Update and skip semantics

`progress` and `scheduler.completed_updates` now mean **successful optimizer
updates** (`counter_semantics=successful_updates_v1`). `attempted_updates`,
`successful_optimizer_updates`, `skipped_updates` and examples consumed are separate.
FP16 overflow skips the optimizer step, lowers the scaler, advances neither the
scheduler nor successful progress, and still counts consumed examples/RNG draws.
Resume restores all counters, scaler/optimizer/scheduler/RNG and overflow streak.
Eight consecutive overflows fail clearly by default; an explicit initial scaler
scale/limit may be configured, without a hidden FP32 fallback. Compute duration/ETA
uses the measured successful-update rate, rather than treating skipped attempts
as successful work. Actual CUDA FP16 overflow behavior remains unverified here;
the counter/state-machine tests explicitly simulate overflow on CPU.

## Evidence and limitations

`outputs/batch-settings-20261031/` contains saved logs, real request/results and
preservation/package records. Checks overlap and are not summed as unique tests.

- `affected-tests.txt`:65 passed,2 CUDA-only skipped, including gradient/remainder,
  workflow/resume and actual MPS paths.
- `settings-verified.txt`:settings-specific suite, covering1024/custom sizes,
  stage overrides, budgets, real Session replacement, active rejection, stale
  preflight, actual worker settings, RNG isolation, skips, provenance and errors.
- `budget-and-recovery-tests.txt`:actual bounded wall-budget learning and dead-worker
  recovery passed; `learning-test.txt`:equal28-example comparison passed.
- `notebook-final-tests.txt`:two tests passed, including actual seven-cell local
  CPU/IPython configure → benchmark → apply → train → metadata verification.
- `frontend-final.txt`:65 frontend tests passed. Production TypeScript/build and
  lint passed; three existing Fast Refresh warnings remain.
- `browser-final.txt`:real rendered API benchmark → application → actual MPS
  pipeline run passed. This is numerical integration, not mocked browser data.
- `browser-current.txt`:persisted-result inspection and final manual application
  UI passed without repeating training/benchmarks. Screenshot:
  `batch-settings-current.png`. Browser test selector failures are retained and
  were corrected; no successful expensive suite was rerun without a change.

The actual browser worker used effective4. A1/B logged [2,2] and A2 [1,1,1,1],
with2 successful updates per stage. The local notebook applied effective4 to its
real Session after a short disposable benchmark and trained with it. These models
only check the pipeline; they are not trained enough to assess scientific accuracy.
CUDA/actual Colab/Drive and large1024 memory fit remain untested. Mocked OOM and
CPU overflow tests establish handling, not CUDA execution. Existing scientific
outputs and checkpoints remain unchanged; no substantial training was launched.

Final focused delivery checks are saved in `final-application-checks.txt` (six
passed), `frontend-application-final.txt` (eight passed), and the final build/lint
logs. These verify later application/provenance/estimate refinements without
repeating completed broad suites. `final-evidence/applied-real-run.json` records
actual effective5 training with [2,2,1] forwards; `final-evidence/benchmark-apply-run.json`
links a real disposable benchmark, applied Session configuration and optimizer
history. Counters from CPU overflow/OOM simulations remain labelled separately.

Source delivery: `pdeflow-batch-settings-source.zip`, SHA256 sidecar, copied
`colab.ipynb`, and starting `source-before.zip` under the evidence root. The ZIP
manifest verifies all included source bytes and exclusions. Preservation and final
status live in `preservation-after.json`, `package-verification.json` and
`completion.json`. Only this task's isolated API8002/frontend5175 were stopped;
existing services and accepted outputs were not reset, staged, committed or removed.

FP16 anchor correction (31 October2026): see [anchor recovery](anchor-fp16-recovery.md).
Anchors require two successful updates within the existing overflow bound;
throughput candidates still count attempted examples. `numerically_finite` now
requires finite gradients, independently of safe scaler skips. Cell4 displays
saved worker and parent failure details. Old failed records are not rewritten.
