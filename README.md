# PDEFlow

A research framework for reconstructing PDE fields from incomplete measurements,
with a broader1D–3D chemistry and steering goal. **New training defaults to M1, an
observation-informed physical mesh model.** M2 adds learned continuous evolution;
M3 is an FGN-style stochastic model; M4/M5 are distinct physical-state flow and
diffusion models. All share periodic2D heat/wave conditioning and explicit u0/v0
state. They require no mandatory separately trained autoencoder latent target.

The original factorized A1/A2/B models, implementation and checkpoints are retained
as explicit legacy baselines. The imported2500-update model remains poor; no new
accuracy improvement is claimed. The new20-update trainability exports are execution
diagnostics, not useful pretrained defaults. Finer coordinate queries do not prove
finer detail. CUDA/live Colab/Drive have not been executed for the new family.

See [the complete mesh guide](docs/mesh-family.md) for exact M1–M5 objectives,
parameter counts, local/Colab suite commands, verified restore and comparisons.
The [verification record](docs/verification.md) separates CPU/MPS/browser evidence
from unexecuted scientific experiments.

## Training stability and live Colab progress

The stability release adds explicit FP32 boundaries for sensitive mesh attention/
kernel products, checks losses before backward, preserves the last healthy checkpoint
on failure, and captures targeted replay diagnostics. The primary Colab training
cell now displays live progress/ETA for individual models and suites. The reported
2269-update remote failure has **not** been reproduced locally; its artifacts are
still needed to verify the first failing operation and healthy recovery point.
Read [the recovery guide](docs/mesh-training-recovery.md) for the exact read-only
collection cell, source-pinned corrected continuation and short T4 acceptance.
Never update scientific source beneath active training or manually edit hashes.

## Install and start

Use Python3.11–3.13 (tested3.12) and supported Node22.13+/24/26 (see .nvmrc):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[workbench,test]'
npm --prefix frontend ci
python -m pdeflow.workbench.api --workspace outputs/local
```

In a second terminal, from the repository root:

```bash
npm --prefix frontend run dev
```

Open **http://127.0.0.1:5173**. API: http://127.0.0.1:8000.
Alternatively, `PDEFLOW_WORKSPACE=outputs/local python scripts/dev.py` launches both;
Ctrl-C stops its own children. Use your supported Node installation on PATH.

## Shortest wave journey

Choose **Wave** under Equation → choose **Standing mode**, **Travelling wave**, **Two-pulse interference**, **Velocity-only pulse** or **Multimode waves**. Matched standing/travelling presets supply consistent velocity. Or **Draw your own** and
choose **Draw quantity** u₀/v₀ → Play or scrub physical time. Travelling wave supplies
its analytically matched velocity; velocity-only pulse explicitly sets u₀=0.
Simulation works without a model. The drawing target is explicit; editing one field
preserves the other. Display quantity selects displacement u or velocity v=u_t.

Choose Measurement quantity u/v and Random, Evenly spaced, Feature-focused or Click
points → **Take measurements** → import/select a compatible mesh or legacy v2 model → **Run model**.
**Infer wave speed c jointly** hides true c from conditioning. Heat has the analogous
known/unknown D option. For smoke models, explicitly check **Barely trained**.
Compare reference/model/error; change Error metric or Error map independently.
Physical diagnostics expands residuals and measured invariant drift. Profiles remain
physical spatial integrals of the selected quantity. Time/quantity changes reuse the
same generated initial state. Metric changes do not change observations or weights.

## New training and comparison

The mesh training screen and Colab Step 3 expose learning rate, effective batch,
GPU microbatch and derived accumulation before training. Advanced settings allow
per-stage overrides. See [mesh training controls and bounded benchmark](docs/mesh-training-controls.md)
for explicit `keep_updates`/`keep_examples`, `1e-4` experiments, resume rules and
Colab reload instructions. Existing runs keep their recorded settings.

Train model → choose M1–M5, capacity, budget and Training equations → optional
field loss (MSE default) → Preflight selected model → Train model. Stop/resume and
current-stage validation previews are on the same screen. Legacy baseline and
comparison opens the retained A1/A2/B controls. No training starts on page open.

```bash
pdeflow mesh plan --candidate M1 --capacity xsmall --updates 500 --device mps --output outputs/m1-config.json
pdeflow mesh preflight --config outputs/m1-config.json --output outputs/m1-run
pdeflow mesh train --config outputs/m1-config.json --output outputs/m1-run
pdeflow mesh resume --run outputs/m1-run --device mps
pdeflow mesh import --checkpoint outputs/m1-run/export/model.pt --workspace outputs/local
```

For Colab open notebooks/colab.ipynb in a **fresh runtime**, upload the current
source ZIP, configure Drive, choose candidate or SUITE=True, explicitly preflight,
then explicitly enable RUN_TRAINING. The [mesh guide](docs/mesh-family.md) has exact
restore/export and immutable-case comparison instructions. Never replace scientific
source beneath active training. The original notebook is retained as
notebooks/legacy-factorized-colab.ipynb for recorded A1/A2/B experiments.

## Retained legacy workflow and evidence

The older A1/A2/B commands and measurements below describe the retained legacy
architecture. They do not define the new default or establish new-family accuracy.

The following legacy model examples retain their original loader. New mesh checkpoints have separate architecture IDs; no weight conversion or
automatic retraining occurs. A smoke export produced during this implementation is:
`outputs/operator-wave-20261030/final-cpu/flow/coordinate.pt`.
Import via the UI, or register it explicitly:

```bash
pdeflow coordinate-import --checkpoint outputs/operator-wave-20261030/final-cpu/flow/coordinate.pt --workspace outputs/local
```

## Train locally

Click **Train model** in the top bar. **Training equations** offers Heat only, Wave only
or Heat + Wave (the new-run default), independently of the playground equation.
The resolved summary shows the actual family probabilities, state schemas and active
parameter ranges; conflicting advanced family selections reject before execution.
Pipeline check/Smoke exercises execution; Development/XSmall is a bounded first
learning experiment, not an accuracy-qualified model. Budget and capacity remain
separate. New runs select **Standard MSE**. Choose **Region-balanced MSE — experimental**
explicitly to use the existing blend (starting α=.5). Only that choice shows
region controls; Standard MSE adds zero region-training queries.
Monitor stage/updates/validation; **Stop** saves at an optimizer boundary; **Resume**
uses recorded objectives/dependencies. **Use model** makes a completed export
available for explicit reconstruction selection. The representation gate compares
matching whole-field metrics with a training-fitted constant baseline. A failed gate
requires inspection and a recorded override; tiny pipeline checks have an explicit
execution exemption.

```bash
pdeflow training start --problem heat-wave-2d --model resolution-flexible --budget development --device mps
pdeflow training status latest
pdeflow training stop latest
pdeflow training resume latest
pdeflow training use latest
```

Low-level shared stages (these launch training when you run them):

The managed command above enforces the representation gate. With low-level commands,
evaluate and inspect A1 before starting A2 or B. Stop if it fails the gate; these
commands do not establish quality or authorize overriding a failed gate.

```bash
python scripts/joint_model.py config --preset xsmall --family joint --device mps --output outputs/my-operator/config.json
pdeflow coordinate-train --config outputs/my-operator/config.json --stage autoencoder --output outputs/my-operator/autoencoder
pdeflow coordinate-train --config outputs/my-operator/config.json --stage evolution --dependency outputs/my-operator/autoencoder/autoencoder.pt --output outputs/my-operator/evolution
pdeflow coordinate-train --config outputs/my-operator/config.json --stage flow --dependency outputs/my-operator/evolution/evolution.pt --output outputs/my-operator/flow
pdeflow coordinate-import --checkpoint outputs/my-operator/flow/coordinate.pt --workspace outputs/local
```

Resume the corresponding stage with its same command plus `--resume`. Increase its
recorded ae_steps/evolution_steps/flow_steps and use `--resume --extend` for deliberate
budget extension. Architecture, objective and frozen dependency changes require a
new run. Choose `--family heat`, `wave` or `joint`. CPU/MPS use fp32; CUDA accepts
`--device cuda --precision auto` (or validated bf16/fp16). There is no hidden fallback.
Large/XLarge remain substantial, unprofiled configurations. No long training ran here.

## Reconstruction and evaluation

The bounded demo writes explicit problem/observation files for both families and
known/unknown scalars; truth is attached only after generation:

```bash
python scripts/joint_model.py demo --checkpoint outputs/my-operator/flow/coordinate.pt --output outputs/my-operator/demo --device cpu
pdeflow coordinate-reconstruct --checkpoint outputs/my-operator/flow/coordinate.pt --problem outputs/my-operator/demo/wave-unknown-problem.json --observations outputs/my-operator/demo/wave-observations.json --output outputs/my-operator/reconstruction --steps 12 --particles 2 --device cpu
pdeflow coordinate-evaluate-ae --checkpoint outputs/my-operator/autoencoder/autoencoder.pt --output outputs/my-operator/state-validation.json --device cpu
python -m pdeflow.coordinate.temporal_validation --checkpoint outputs/my-operator/flow/coordinate.pt --output outputs/my-operator/reconstruction-validation.json --device cpu --subset 2 --count 12
```

Use fresh output paths. Held-out evaluation needs the original dataset manifest;
portable inference needs neither training data, external dependencies nor the state
encoder. A2 evaluation uses `pdeflow.coordinate.validation.evolution` with the same
checkpoint/output/device arguments. Scientific evaluation is distinct from completion.

## Colab and source transfer

Use [mesh optimizer/batch controls](docs/mesh-training-controls.md) for the active mesh training UI and Colab notebook. The [earlier batch configure → benchmark → apply guide](docs/batch-settings.md) applies to the retained A1/A2/B workflow. It includes per-stage sizes, explicit budget policies, and authoritative Session application. The [actual A1/A2 batching guide](docs/batched-training.md) records the earlier throughput work. The earlier [T4 performance recovery](docs/training-performance.md) records the reference/precision changes. The retained factorized preflight uses two warmup/eight measured updates. Active mesh preflight uses eight total successful updates (two warmup/six measured) and preserves the original schedule horizon. Native precision detection and separate setup/reference/compute/validation/persistence reporting remain.

Open `notebooks/colab.ipynb` from the latest source ZIP. Its eight steps cover source
verification, Drive mount/probe, science, optional benchmarks, explicit application,
training/resume, **Save and verify complete run**, and optional diagnostics.
[Exact settings, equations, save/restore and bounded ablations](docs/observation-aligned-training.md).
New runs use unique IDs separate from display names; keep the printed RUN_ID.
First-checkpoint, successful-update cadence and stage/final snapshots retain the
full encoder, dependencies, optimizers/scalers/RNG and exact source. Three manifest
views distinguish inference, diagnostic and resumable artifacts without duplicate
weight bundles. Restore uses recorded science and verified manifests; it never
substitutes today's preset or an unrelated encoder.

A1 learns representation, A2 freezes it and learns supervised evolution, B freezes
both and learns conditional generation. New runs default to Standard MSE; region-balanced alpha=.5 is explicit and
experimental; baseline FM remains the default B objective. The separate explicit
`joint.observation_auxiliary` adds batched decoded clean-endpoint Gaussian/tolerance
supervision when enabled. Optional A2 physics and structured phase-sensitive
spectral field loss remain off unless selected. No change demonstrates accuracy.
The failed representation gate remains a stop. Interrupt once or `session.stop()`
for orderly saving, then `session.finalize()` for complete verification. Download
`export/coordinate.pt` for inference on the Mac; retain the diagnostic/resumable
store for recovery. **Actual Colab/Drive/CUDA execution of this revision is
unverified**; local execution of the actual notebook cells is separate evidence.

```bash
python scripts/package_source.py --output outputs/pdeflow-operator-wave-source.zip
python scripts/joint_model.py counts --output outputs/operator-parameter-counts.json
```

The ZIP excludes weights, datasets, accepted outputs, environments, caches and
secrets. See SOURCE_PACKAGE.md. Contracts, formulas, capacities and limitations:
[operator-wave-model](docs/operator-wave-model.md). Actual evidence and preservation:
[operator-wave-verification](docs/operator-wave-verification.md).

Two trained families do not establish universal PDE generalization. The graph encoder
is structured conditioning, not symbolic solving. Learned evolution is not guaranteed
physically exact; sparse unknown-state/parameter inference may be ambiguous. Residuals
and invariants are measured. Static/weak branches, forcing, new boundaries and other
neural dimensions/geometries are unsupported by this v2 specialist; numerical research
modules and historical outputs remain available.

Colab FP16 anchor failure/reload: [bounded recovery and T4 verification](docs/anchor-fp16-recovery.md).
The two-attempt anchor bug is repaired; skipped gradients are no longer labelled
finite, and cell4 displays failures and tracebacks. CPU scaler-control and MPS
checks are separate from native CUDA/T4 verification, which remains outstanding.

## Diagnose a saved heat/wave reconstruction

The shared CLI captures immutable measurements/reference/noise and evaluates A1,
true-initial-state A2 and production B separately. Matching A2 is required for an
omitted encoder; a portable export alone never substitutes a random encoder.

```bash
pdeflow coordinate-diagnostic-case --checkpoint MODEL.pt --job JOB_DIRECTORY --output NEW_CASE
pdeflow coordinate-diagnose --checkpoint MODEL.pt --case NEW_CASE --output NEW_REPORT --device mps
```

See [exact case commands, measured results and limits](docs/reconstruction-diagnosis.md).
The collapsed **Reconstruction diagnostics** view shows ensemble errors at exact
sensors, separately from whole-field error. Experimental latent correction is a
separate explicit command, not a retrained model or calibrated posterior. New
training may opt into `joint.noiseless_probability`; its default0 preserves old
sampling, and resume changes are rejected. The existing region-balanced alpha=.5 option is preserved, while new runs default to Standard MSE.

Current field-loss workflow: [Standard versus experimental region-balanced MSE](docs/standard-mse-workflow.md).
The selection affects A1 and decoded A2 only. Training loss controls learning;
display metrics only change evaluation. CLI new-run selection, without launching:

```bash
pdeflow training start --plan --device cpu --field-loss mse
pdeflow training start --plan --device cpu --field-loss region_balanced_mse --field-alpha 0.5
```

## Known measurement times and A2 monitoring (November 2026)

New training runs use **continuous, stratified and shuffled whole-interval sensor
times**. Each sensor's time is known conditioning, independent of reconstruction
and flow time. In the playground, **Take measurements → Measurement times** offers
Random across the time interval or arbitrary Choose measurement times. The count
is TOTAL, with optional explicit counts per entered time. Click points stores its
visible Active measurement time per click. Display-time changes preserve readings.
Inspect the timestamp timeline/list and temporal marker tolerance. JSON export/
import and diagnostic replay preserve timestamps. Imported static/undeclared
trajectory references cannot silently supply arbitrary-time readings.

[Measurement policies, configuration, compatibility and evidence](docs/measurement-times.md)
explain the new B default and independently tested continuous A2 target-time policy.
[A2 investigation](docs/a2-investigation.md) documents corrected current nonzero-time
previews, full-history validation markers, reproducible validation and bounded
learning results. The current remote Colab run's actual artifacts remain needed;
the document includes a read-only export cell and safe Drive-verification procedure.
Do not reload new scientific source inside an active training process. Preserve its
verified snapshot/source; use a NEW experiment for the revised observation policy.

## Actual trained-model investigation

The downloaded run `20261101T153417-b088674d2622` has matching completed A1/A2/B
checkpoints at2,500 successful updates each. [The scientific diagnosis](docs/trained-model-analysis.md)
reports actual encoder/evolution oracles, fixed-noise conditional inference,
conditioning ablations, physics checks and observation-only correction. A1 is
imperfect, A2 is useful for heat but weak for wave velocity/consistency, and B adds
a large initial-state/conditioning error. A full-batch B component-logging defect
was fixed; the training objective and stored weights were unchanged. Original
artifact hashes and preserved source are verified. CPU/MPS execution is distinct
from historical T4 training; no new CUDA/Colab execution was performed.

Targeted conditioning experiments (1 November2026): **Train model → Conditional
experiment (B)** and notebook step3 expose B0 baseline, B1 direct spatial sensor
attention, B2 endpoint observation supervision, and B3 both. New-run A1 remains
Standard MSE; region-balanced/spectral options remain available. Wave-only A2
`evolution.kinematic` and A1 `representation_weights` are independent advanced
experiments, off/unchanged by default. Strict legacy loading, explicit new-run
warm starts, exact frozen-source audits and full artifact persistence are described
in [spatial-conditioning.md](docs/spatial-conditioning.md), with runnable Colab
configuration and reload/acceptance instructions. Eight-update comparisons and a
20-update deterministic control were executed; **no accuracy improvement is
established**. Preserve the original completed2500-update model for comparison.
