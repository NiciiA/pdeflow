# PDEFlow

A research framework for reconstructing PDE fields from incomplete measurements,
with a broader 1D–3D chemistry and steering goal. The active playground uses an
**Operator-Aware Generative Latent Neural Operator v2**, shared by periodic 2D
heat and wave equations. It generates initial physical states and unknown typed
scalars, then predicts dynamics with a learned graph-conditioned evolution operator.
There is no unrestricted trajectory latent or reference-solver inference shortcut.

The repository contains two-update pipeline fixtures and a user-imported Medium
model with **2,500 successful flow updates**. Its saved reconstructions are poor;
[the reconstruction diagnosis](docs/reconstruction-diagnosis.md) separates actual
sensor fit, initial-state error, oracle-stage availability and bounded correction.
Resolution flexibility does not demonstrate fine-detail recovery.
Historical Fourier-inverse superiority and non-monotonic test-budget results remain
unchanged; no matched accuracy comparison of this architecture has been run.

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

Choose **Wave** under Equation → choose **Standing mode**, **Travelling wave** or
another preset → select an independent Initial velocity, or **Draw your own** and
choose **Draw quantity** u₀/v₀ → Play or scrub physical time. Travelling wave supplies
its analytically matched velocity; velocity-only pulse explicitly sets u₀=0.
Simulation works without a model. The drawing target is explicit; editing one field
preserves the other. Display quantity selects displacement u or velocity v=u_t.

Choose Measurement quantity u/v and Random, Evenly spaced, Feature-focused or Click
points → **Take measurements** → import/select a compatible v2 model → **Run model**.
**Infer wave speed c jointly** hides true c from conditioning. Heat has the analogous
known/unknown D option. For smoke models, explicitly check **Barely trained**.
Compare reference/model/error; change Error metric or Error map independently.
Physical diagnostics expands residuals and measured invariant drift. Profiles remain
physical spatial integrals of the selected quantity. Time/quantity changes reuse the
same generated initial state. Metric changes do not change observations or weights.

Existing v1/older weights are preserved and incompatible; no weight conversion or
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

Use the [batch configure → benchmark → apply → train guide](docs/batch-settings.md) for the active training UI and Colab notebook. It includes per-stage sizes, explicit budget policies, and authoritative Session application. The [actual A1/A2 batching guide](docs/batched-training.md) records the earlier throughput work. The earlier [T4 performance recovery](docs/training-performance.md) records the reference/precision changes. The updated preflight uses two warmup/eight measured updates, native precision detection and separate setup, reference, compute, validation and persistence clocks.

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
