# PDEFlow

A research framework for reconstructing PDE fields from incomplete measurements,
with a broader 1D–3D chemistry and steering goal. The active playground uses an
**Operator-Aware Generative Latent Neural Operator v2**, shared by periodic 2D
heat and wave equations. It generates initial physical states and unknown typed
scalars, then predicts dynamics with a learned graph-conditioned evolution operator.
There is no unrestricted trajectory latent or reference-solver inference shortcut.

Only tiny execution checks have been run. The available new checkpoints have
**two updates per stage and poor accuracy**, including worse-than-constant state
representation. Resolution flexibility does not demonstrate fine-detail recovery.
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
separate. New runs select experimental region-balanced field loss, α=.5.
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

Before substantial training, follow the [T4 performance recovery and source-reload guide](docs/training-performance.md). The updated preflight uses two warmup/eight measured updates, native precision detection and separate setup, reference, compute, validation and persistence clocks.

Open `notebooks/colab.ipynb`. Upload a current source ZIP or configure your own
repository URL/revision. Set fast local and persistent roots. Choose FAMILIES
heat/wave/joint, capacity, device/precision, loss and stage budgets. Inspect the
training-only gallery and real three-stage preflight; then enable RUN_TRAINING.
A1 learns state representation; A2 freezes it and learns evolution; B freezes both
and learns conditional generation. Optional decoded auxiliary terms are off by default.
Interrupt once or `session.stop()` for orderly saving/sync. Restore the same source,
run name and recorded configuration to resume. Download `export/coordinate.pt` and
its provenance JSON, then import on the Mac. No Drive path or external encoder is
required for inference. **Actual Colab/CUDA execution remains unverified**; local
execution of the actual notebook cells on CPU is separate evidence.

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
