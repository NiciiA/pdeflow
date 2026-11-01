# Physical reference data providers

Implementation uses one trainer and a synchronous CPU reference producer.
`provider.mode` is offline, online or hybrid. Omitting provider preserves legacy
NPZ/sharded offline commands and their fitted-training-set transform. The new
provider offline mode uses `prepare-training`; online/hybrid do not require
`--data`. Low-level commands require explicit fixed `--evaluation` manifests; named
`pdeflow train --preset ...` prepares/reuses them automatically.

Sampling distributions/specifications are independent of mode and storage.
SHA256-derived IDs use disjoint train/calibration/validation/test namespaces,
root seed, physical distribution/solver version, logical index and bounded
attempt number. Separate seeds identify initial fields, known forcing,
reference stochasticity (currently deterministic), observations and solver
metadata. Neural source/time/batch streams retain their own torch generators.
One producer/rank is supported: workers other than0 and initialized distributed
execution are rejected before generation. No accelerator model runs in workers.

Online solves a genuinely new physical problem at each presentation. Hybrid
is a bounded in-memory round-robin replay pool, not asynchronous prefetch.
It fills capacity slots, presents each at most reuse_limit times, then replaces
that slot with the next independently identified reference. It stores no more
than capacity clean targets. Queue length is zero. Resume stores cursor, pool
specifications/usage counts and consumed IDs; it regenerates cached targets
from their physical IDs and charges these extra solve calls separately.
Generated-ahead is distinct from consumed. Logical resume has the same sequence;
no cross-device bitwise claim is made.

A separate finite training-only calibration namespace fits the transform once.
Its IDs and exact transform are checkpointed and reused on resume. There are no
validation/test statistics or adaptive normalization updates. Existing channel
scale floor1e-8 handles near-constant channels explicitly; nonlinear field
transforms remain owned by each specialist. Calibration is not an ordinary
training presentation. Data event journals retain attempted/accepted physical
specifications, solver diagnostics, rejection reasons and times. At most
max_attempts distinct attempt IDs are tried per logical problem, then failure
propagates. Rejection can bias the effective distribution; report failure rate.

Fixed evaluation manifests retain physical IDs/specifications, checksummed
cached clean references and fixed observation locations/noise realizations.
Changing mode does not change their identity. Cached target/record hashes and
physical distribution/solver identity are checked before use. Evaluation truth
remains excluded from inference requests and training normalization.

SIGINT is deferred to the end of an optimizer update. The trainer then attempts
an atomic checkpoint at that safe boundary, reports its step and closes the
provider. An initial step0 checkpoint is saved after calibration. Atomic writes
preserve the preceding valid checkpoint on save failure. Event journals can
contain generated-but-unconsumed attempts; the checkpoint cursor/consumed IDs
are authoritative. No background producer can outlive training.

Accounting distinguishes sampling, reference solve, cache/load, data wait,
transform/transfer, neural host, validation/checkpoint and total time. Data-wait
includes synchronous reference generation; nested costs must not be added.
Accelerator host phase timings can overlap device execution; explicit timing synchronization
occurs at checkpoint/total boundaries; existing scalar loss and finite checks
can also synchronize. No extra per-step synchronization was added for timing. Offline setup
and calibration must be charged when comparing total cost. `data_benchmark`
uses matched priors/models/update budgets and reports shared fixed-evaluation
setup separately. It is a bounded execution measurement, not a speed claim.

### Executed matched cost probe (22 October 2026)

CPU, four updates, batch two, same small model/prior and eight presentations:

| Specialist | Offline seconds | Online seconds | Hybrid seconds |
|---|---:|---:|---:|
| Heat | .2549 | .0681 | .0732 |
| Darcy | .1153 | .0706 | .0597 |
| Chemistry | .4664 | .4249 | .2777 |

These warm totals include offline preparation or online reference solves,
calibration, training, validation and checkpoints. Common optimizer initialization
was measured separately (heat1.1139s, Darcy.4678s, chemistry.6200s); shared fixed
evaluation preparation was .0303s/.0250s/.1008s. Add these common costs when
estimating a fresh invocation. This is one tiny trial per mode, not a throughput
ranking for larger training. Offline solved eight training problems, consumed
six unique ones and reused two presentations; online solved/consumed eight with
zero reuse; hybrid solved/consumed four and reused four. Each mode additionally
used two independent calibration problems. There were zero reference failures.
Thus replay has a different effective unique-example budget even at matched
optimizer/presentation budgets. Raw phase/call accounting is in
outputs/online-chemistry-20261021/timing-summary.json and throughput-final-*/.

## Stable identities and audit details

New manifest schema2 computes its logical identity from physical distribution,
IDs and indices, excluding measured runtime. Recreating held-outs in a different
data mode therefore yields the same identity and problems. Separate record and
target SHA256 checks still detect mutation. Schema1 manifests from the earlier
online implementation remain readable without rewriting them.

The journal records attempt-start and completion separately. Its streaming audit
counts actual completed/failing/regenerated calls and unique accepted IDs,
including discarded work after a checkpoint; incomplete attempts are labelled
with unknown duration. Logical checkpoint counters govern reproducible consumption.
When moving a checkpoint, also copy the journal for a complete historical cost
audit; the checkpoint alone suffices for sequence/transform restoration.
Calibration cancellation before any optimizer setup may leave no training
checkpoint; rerunning regenerates its deterministic calibration stream. During
ordinary training, SIGINT is deferred to an optimizer boundary and reports the
saved step. Reference generation uses single-threaded FFT/reduction kernels;
workers=0 and solver_threads=1 are the only supported settings. Torch CPU thread
count is separately runtime.threads; no background models/queues are created.

## Exact local commands

Run from the repository root using Python3.12. No saved training set is needed
for online/hybrid; clean numerical reference solves are still performed.
Choose a fresh output root and keep its evaluation manifest fixed.

```bash
PDEFLOW_CONFIG=configs/experiments/online_mps_heat_2d.json
PDEFLOW_RUN=outputs/my-online-heat
.venv/bin/python -m pdeflow.cli prepare-evaluation \
  --config "$PDEFLOW_CONFIG" --output "$PDEFLOW_RUN/evaluation"
.venv/bin/python -m pdeflow.cli train --config "$PDEFLOW_CONFIG" --device mps \
  --evaluation "$PDEFLOW_RUN/evaluation" --output "$PDEFLOW_RUN/training" --stop-after 100
.venv/bin/python -m pdeflow.cli train --config "$PDEFLOW_CONFIG" --device mps \
  --evaluation "$PDEFLOW_RUN/evaluation" --output "$PDEFLOW_RUN/training" \
  --resume "$PDEFLOW_RUN/training/checkpoint.pt"
```

Use these real profiles and a distinct run root for each family/dimension:

| Family | Small CPU online | Larger MPS | Colab CUDA config (untested runtime) |
|---|---|---|---|
| Heat2D / true3D | online_heat_2d.json / online_heat_3d.json | online_mps_heat_2d.json / online_mps_heat_3d.json | online_colab_heat_2d.json / online_colab_heat_3d.json |
| Darcy2D / true3D | online_darcy_2d.json / online_darcy_3d.json | online_mps_darcy_2d.json / online_mps_darcy_3d.json | online_colab_darcy_2d.json / online_colab_darcy_3d.json |
| Chemistry2D / true3D | online_chemistry_2d.json / online_chemistry_3d.json | online_mps_chemistry_2d.json / online_mps_chemistry_3d.json | online_colab_chemistry_2d.json / online_colab_chemistry_3d.json |

All filenames are under configs/experiments. Larger heat/Darcy grids are32²/16³;
chemistry pilots16²/8³ have three species/four physical frames. Larger profiles
use3000/1500 updates, not automatically executed here. Set --device cpu for a
CPU run. Configured MPS/CUDA absence raises an error; no hidden fallback.

A bounded real preflight prepares its own fixed evaluation set, calibrates,
performs two updates with resume, evaluates physical guidance and reloads the
export on CPU. It respects configured cache/retry/worker limits:

```bash
.venv/bin/python -m pdeflow.cli preflight \
  --config configs/experiments/online_mps_chemistry_2d.json --device mps \
  --output outputs/my-chemistry-preflight
```

Compare offline and hybrid with identical8² heat held-outs:

```bash
.venv/bin/python -m pdeflow.cli prepare-evaluation \
  --config configs/experiments/online_heat_2d.json --output outputs/modes/evaluation
.venv/bin/python -m pdeflow.cli prepare-training \
  --config configs/experiments/offline_heat_2d.json --output outputs/modes/offline-data
.venv/bin/python -m pdeflow.cli train --config configs/experiments/offline_heat_2d.json \
  --data outputs/modes/offline-data --evaluation outputs/modes/evaluation \
  --output outputs/modes/offline-training
.venv/bin/python -m pdeflow.cli train --config configs/experiments/hybrid_heat_2d.json \
  --evaluation outputs/modes/evaluation --output outputs/modes/hybrid-training
```

Equivalent offline_/hybrid_darcy_2d.json and chemistry_2d.json profiles exist.
For a recorded matched-cost comparison use the real module:
`python -m pdeflow.data_benchmark --config configs/experiments/online_darcy_2d.json --output outputs/my-throughput`.
Do not compare modes by excluding offline preparation or online solver cost.

## Reconstruction, discovery and transfer

After a new run, use the same config (CPU remapping is supported), checkpoint
and fixed test inputs. Evaluation truth is optional and opened only after all
reconstructions and conventional baselines have completed.

```bash
.venv/bin/python -m pdeflow.cli reconstruct --config "$PDEFLOW_CONFIG" --device cpu \
  --checkpoint "$PDEFLOW_RUN/training/inference.pt" \
  --problem "$PDEFLOW_RUN/evaluation/test/0/problem.json" \
  --observations "$PDEFLOW_RUN/evaluation/test/0/observations.json" \
  --reference "$PDEFLOW_RUN/evaluation/test/0/reference.npz" \
  --output "$PDEFLOW_RUN/reconstruction"
.venv/bin/python -m pdeflow.cli evaluate --result "$PDEFLOW_RUN/reconstruction/observation_pde"
```

Register exports through PDEFLOW_MODEL_ROOTS (colon-separated on macOS), then
start the API. Each inference.pt has a content ID; the current equation/full
request selects compatible checkpoints. Unknown grids/roles/boundaries do not
silently fall back to heat. Explicitly review any layout suggestion. Example
using existing saved artifacts:

```bash
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training:outputs/multispecialist-20261021/cpu-2d/training:outputs/multispecialist-20261021/cpu-3d/training:outputs/online-chemistry-20261021/chemistry-2d/training:outputs/online-chemistry-20261021/chemistry-3d/training"
export PDEFLOW_JOB_ROOT="outputs/my-online-workbench/jobs"
export PDEFLOW_DATA_ROOT="outputs/my-online-workbench/data"
.venv/bin/python -m pdeflow.workbench.api
# In a second terminal, from frontend:
node --version
npm run dev -- --strictPort
```

Open http://127.0.0.1:5173 (API8000). Use the supported Node on PATH; recovery
used Homebrew Node22.23.3. Add the preserved 1D heat training root when running
the entire browser suite. Chemical preset insertion plus a compatible grid
routes automatically; point observations and reference attachment work as for
other families. UI data-mode controls export config/commands only.

notebooks/colab.ipynb uses the same APIs. Choose an online_colab_* profile and
DEVICE="cuda" on suitable hardware, with an explicit persistent RUN_ROOT.
It persists fixed evaluation, checkpoints and optional offline data; online/
hybrid skip full training-data generation. Re-running resumes. Transfer
portable-export/inference.pt, config.json, test_problem.json and
test_observations.json to the Mac. Training arrays are not needed for inference.
For training resume retain checkpoint.pt, identical source/config, evaluation
manifest/cache, and optionally journals for complete cost history. CUDA and
actual Colab execution remain untested; local notebook cells were exercised on
CPU without a saved training directory.

## FK inference after training

FK uses the same exported independent-Gaussian linear-CFM model; no extra FK
training objective or checkpoint is required. Current exact-grid8²/8³ inference
profiles are configs/experiments/fk_*.json, with explicit MPSfloat32 variants.
Validate the full checkpoint/layout first; larger trained layouts need their own
matching public config plus inference.fk block. FK/importance select real
Euler–Maruyama stepping and weighted inference; the existing gradient/refinement
paths retain separate semantics. See fk-verification.md for exact local/MPS
commands, Colab transfer and current negative/limited scientific evidence.
CUDA/actual Colab FK execution is untested; training resume is not FK resume.

## Local run wrapper

`pdeflow train --preset heat-2d|darcy-2d|chemistry-2d --device mps` allocates a
managed run and calls these same provider/trainer functions. --data-mode
online/offline/hybrid is a fresh-run override; offline preparation is automatic,
and online still requires no saved training set. Provider identity/normalization
and reference isolation remain unchanged. Recoverable preparation reuses valid
completed physical records; an optional completed calibration cache avoids
refitting before the first checkpoint if startup was interrupted. Normal resume
uses the checkpoint's exact frozen transform/provider cursor and never calibrates
again. Partially completed calibration can repeat deterministic solves.
The notebook's preferred section now uses these same train/resume/evaluate
commands with an explicit persistent workspace. See everyday-usage.md.
