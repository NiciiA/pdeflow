# Advanced PDEFlow workflows

PDEFlow is a modular research package for reconstructing PDE fields from sparse
measurements. Implemented specialists execute periodic heat in **one, two and three spatial
dimensions**, variable-permeability Darcy and A+B ↔ C chemical reaction–diffusion
in **two and three spatial dimensions**, with real shared conditional flow-matching training and heuristic
guided reconstruction, plus mathematically specified FK/terminal-importance
steering with weighted populations and real systematic resampling. Standalone Poisson numerics are also verified. Heat is the
verification anchor for a broader chemistry/steering programme, not the final
scientific scope. See [the capability matrix](capabilities.md),
[mathematics](mathematics.md), [decisions](decisions.md), and
[verification record](verification.md).

The [heat recovery handoff](handoff.md) records the historical existing-artifact
workbench/CLI workflow, backend evidence, pilot results and final source archive.
For an extracted archive, see [SOURCE_PACKAGE.md](../SOURCE_PACKAGE.md).

The current workbench starts with an empty **MathLive symbolic equation editor**.
Insert Heat, Darcy or chemistry, edit the full 𝒞=(𝓔,λ,Ω,ℬ,ℐ,y,M), and let the current parsed
equation route to a compatible checkpoint. Other presets remain editable
systems without execution adapters. [The multi-specialist handoff](multispecialist.md)
contains exact launch, larger Darcy MPS generation/training/resume, CLI and Colab
commands, checkpoint registration and current limitations.
See [the exact PDE grammar and request schema](symbolic-editor.md),
[larger dataset profiles and MacBook commands](local-training.md), and
[the specialist extension map](equation-specialists.md).

The CPU smoke models are deliberately undertrained. They establish working
numerics, optimizer updates, persistence and reconstruction, **not useful
scientific accuracy, exact posterior sampling or calibrated uncertainty**.
The conventional Fourier inverse is substantially better on the heat cases;
test quality was non-monotonic across the heat training budgets. The Darcy
conventional inverse is implemented and also outperformed the bounded online
generative study. See [Darcy study](darcy-study.md).

The current implementation adds common offline/online/hybrid physical-reference
providers and executable chemistry. See [the current handoff](online-chemistry-handoff.md),
[online training and resume commands](online-training.md), and
[chemical numerics and assumptions](chemistry.md). The earlier handoffs
retain their historical evidence. The latest FK verification is dated 25 October.

## Feynman–Kac steering — current continuation

The [FK handoff](fk-verification.md) contains exact target assumptions,
analytic Gaussian/mixture results, heat comparisons, real browser/MPS evidence,
launch/CLI commands and the source archive. The [derivation](fk-mathematics.md)
specifies score-corrected vanishing-noise Euler–Maruyama dynamics, telescoping
potentials and finite-step/particle approximations. FK plus refinement is rejected.

```bash
source .venv/bin/activate
python -m pdeflow.fk.toys --kind gaussian --output outputs/my-fk-gaussian
python -m pdeflow.fk.toys --kind mixture --output outputs/my-fk-mixture
pdeflow reconstruct --config configs/experiments/fk_heat_2d.json --mode fk \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --problem outputs/accepted_cpu_2d/data/test_problem.json \
  --observations outputs/accepted_cpu_2d/data/test_observations.json \
  --output outputs/my-fk-heat
pdeflow evaluate --result outputs/my-fk-heat/fk \
  --reference outputs/accepted_cpu_2d/data/test_reference.npz
```

These commands use existing local exports. A source-only archive omits them;
[SOURCE_PACKAGE.md](../SOURCE_PACKAGE.md) explains generating compatible demos.
Actual CPU/MPS execution and analytic target checks passed; CUDA/actual Colab
remain untested. The heat FK probe improves observation fit but worsens
unobserved error relative to unguided generation; undertrained models do not
establish useful reconstruction or calibrated uncertainty. Full latest regression:
143 Python tests,20 frontend tests,16 real browser workflows; FK includes1D–3D,
reopen, reference isolation, rejection and cancellation. No large training ran.

## Install

Tested: Python **3.12.3**, macOS 15.2 arm64, PyTorch 2.5.1, NumPy 2.1.3,
Matplotlib 3.9.2 and pytest 8.3.4. `requirements-tested.txt` pins the complete
observed macOS Python environment. Python 3.11 is permitted by package metadata
but untested. CUDA dependencies are platform-dependent; Colab is not assumed to
have the same preinstalled packages. No mandatory CUDA extensions or services.

From the repository/source directory, with uv installed:

```bash
UV_CACHE_DIR="$PWD/cache/uv" uv venv .venv --python python3
UV_CACHE_DIR="$PWD/cache/uv" uv pip install --python .venv/bin/python \
  -c requirements-tested.txt -e '.[test,workbench]'
source .venv/bin/activate
pdeflow hardware
python -m pytest -q
```

Standard pip alternative: `python3 -m venv .venv`, activate it, then
`python -m pip install -c requirements-tested.txt -e '.[test,workbench]'`.
Use Python 3.12 for the tested environment. All commands below run from this
source directory; all input and output paths are configurable CLI arguments.
Existing completed data/training/reconstruction runs are protected from
accidental fresh-run overwrite. Choose a new output directory for a new run.

## Execute the CPU acceptance profiles

```bash
pdeflow smoke --config configs/experiments/cpu_2d.json --output outputs/run-2d
pdeflow smoke --config configs/experiments/cpu_3d.json --output outputs/run-3d
```

Each command generates complete-trajectory splits and noisy sensors, trains
for half the budget, resumes to the full budget, reloads the inference export,
runs unguided/observation/observation-plus-PDE reconstruction and the Fourier
inverse, then evaluates and plots. The 2D profile uses 30 trajectories, 30
optimizer updates and an 8x8 grid. The true 3D profile uses 18 trajectories,
20 updates and an 8x8x8 grid. Both save four physical-time frames. CPU was
actually tested; MPS was subsequently verified with normal host access; CUDA remains unavailable.

## Generate, train, reconstruct and evaluate separately

```bash
pdeflow validate-config --config configs/experiments/cpu_2d.json
pdeflow generate --config configs/experiments/cpu_2d.json --output outputs/demo/data
pdeflow train --config configs/experiments/cpu_2d.json \
  --data outputs/demo/data --output outputs/demo/training --stop-after 15
pdeflow train --config configs/experiments/cpu_2d.json \
  --data outputs/demo/data --output outputs/demo/training \
  --resume outputs/demo/training/checkpoint.pt
pdeflow inspect-checkpoint --checkpoint outputs/demo/training/inference.pt
pdeflow reconstruct --config configs/experiments/cpu_2d.json --device cpu \
  --checkpoint outputs/demo/training/inference.pt \
  --problem outputs/demo/data/test_problem.json \
  --observations outputs/demo/data/test_observations.json \
  --reference outputs/demo/data/test_reference.npz --output outputs/demo/reconstruction
pdeflow evaluate --result outputs/demo/reconstruction/observation_pde
```

Repeat with `cpu_3d.json` for 3D. `--stop-after` simulates a bounded interruption
without changing the configuration. Resume requires the same config, dataset,
training split, transform and implementation; restores Adam/progress/RNG; and
continues to the configured total steps. Checkpoints are written every configured
interval and at bounded stops. No scheduler or precision scaler is used, and
their saved states are explicitly null. A real interruption can lose updates
since the last completed checkpoint. Writes use temporary files and atomic
replace. A failed generation before its final manifest requires a new dataset
folder; incomplete files are not silently treated as a complete dataset.

`test_reference.npz` is optional evaluation truth. **Reconstruction needs only
`inference.pt`, a config, `test_problem.json` and `test_observations.json`.** It
never loads the training data. Omit `--reference` for standalone inference.
These JSON files contain known diffusivity/grid/field metadata and measured
values; hidden initial fields never enter the model or guidance. Checkpoints
use PyTorch's restricted `weights_only=True` loader and CPU device remapping.

To use a portable checkpoint downloaded from Colab on a Mac:

```bash
pdeflow reconstruct --config config.json --device cpu --checkpoint inference.pt \
  --problem test_problem.json --observations test_observations.json \
  --output outputs/from-colab
```

The checkpoint validates the exact spatial grid, field/time layout, backbone,
formulation and conditioning schema. `--device cpu` explicitly overrides a saved
CUDA runtime. MPS is optional via `--device mps` where available; MPS float32 training/guidance and CPU reload were verified in the continuation. Float32/full precision is the default. CPU/CUDA float64 is
configurable; MPS float64, mixed precision and unavailable devices are rejected.
There is no hidden fallback. Cross-device bitwise equality is not promised.

## Configuration and outputs

Edit a copy of a JSON profile to switch `inference.stepper` between `euler` and
`heun`, or `inference.schedule` between `uniform` and `quadratic`. No training or
physics code changes are required. Reconstruction always exports the three
required guidance comparisons and the conventional inverse baseline, using the
same sampling seed across generative comparisons. Guidance strengths, clipping,
noise floor, candidate count, basis degree and ridge are explicit. No test-truth
tuning is performed. `gpu_2d.json` (64x64) and `gpu_3d.json` (16 cubed) are larger
unexecuted CUDA pilots, not claims of hardware sizing or model quality.

Each run writes:

- `data/`: safe NPZ arrays, hash/split/seed manifest, standalone test metadata,
  noisy measurements and separate evaluation reference.
- `training/`: atomic `checkpoint.pt` for resume, portable `inference.pt`,
  loss history, resolved config, environment, source identity and runtime costs.
- `reconstruction/`: comparison JSON; each method has sample arrays, physical
  metrics, known problem/observation JSON and a reference/measurement/mean/error
  PNG. 3D plots use the labelled central z slice at the final physical time.

Reconstruction error, observation fit, finite-difference and spectral-space
residuals, mass drift, candidate spread, descriptive interval coverage, network
calls, gradient evaluations and timing are reported separately. A low residual
alone does not establish a correct solution. Memory is a measured process
lifetime RSS high-water mark, not isolated per-phase GPU memory. Generated
outputs, caches, weights and datasets are ignored by Git. The supplied folder
initially had no Git metadata. Source and the npm lockfile are staged in a local
repository, but no commit exists: the final commit action was rejected by
automatic approval review. Recovery leaves the index intact; no remote or
publication is configured.

## Colab

Open [notebooks/colab.ipynb](notebooks/colab.ipynb). Choose an exact Git commit
and your own repository URL, or upload a source archive. To create the archive:

```bash
python scripts/package_source.py --output outputs/pdeflow-source.zip
```

The notebook creates an isolated environment under a configurable disposable
cache, installs this package with the tested constraints, inspects hardware,
and calls the same CLI. Select an explicit persistent output path after setting
up your preferred storage. Data/checkpoints/results live there; no Drive mount,
account or repository URL is assumed. Re-run the training cell with unchanged
configuration to resume. Download `inference.pt` and the public input/config
files for local inference. Notebook cells were compiled and archive setup/cache/path checks executed locally. Colab and CUDA execution remain untested; no full GPU pilot was launched.

## Architecture and next slices

`core` defines semantic contracts and capabilities. `equations` evaluates safe
scalar trees. `numerics` provides CPU references and differentiable operators;
`data`, `observations`, `conditioning` and `models` have separate roles.
`generative` defines the probability path/objective, `training` orchestrates
optimization/persistence, `sampling` composes ODE dynamics/steppers/schedules,
`steering` applies endpoint gradients, and `evaluation` owns metrics, figures
and the inverse baseline. Fixed physical-time frames are reversibly packed into
spatial CNN channels; changing physical times requires a compatible checkpoint.
Independent candidates have an explicit particle axis and no interactions.

Only periodic scalar heat reconstruction is trained here. Poisson has numerical
verification but no neural checkpoint. No generic PDE solver, unseen-equation
understanding, resolution generalization, arbitrary time-window inference,
unknown-parameter inference or off-grid interpolation is claimed.

The next complete slice is variable-coefficient Darcy in 2D/3D: a verified
conservative reference solver, positive permeability transforms, explicit
coefficient/pressure unknowns, flux/gauge checks, sparse inverse baseline and
trained reconstruction evaluation. Then add substantive A+B reversible C
chemistry with positivity and the correct a+c and b+c invariants. Waves, fluids,
Fokker–Planck, a specified diffusion formulation, refinement and FK remain
planned. FK first needs a derived stochastic formulation and a tractable toy
validation; the current split guidance is not FK or exact posterior sampling.

## Scientific workbench and bounded pilot

The functional local React/Vite/TypeScript workbench uses real shadcn components
and a FastAPI compute service. It supports problem/operator editing, strict
physical-coordinate CSV/JSON measurements, server-discovered checkpoints,
compatibility validation, real background jobs, progress/cancellation,
1D curves, 2D heatmaps, true 3D slices, sample/mean/spread views, post-inference
evaluation references, artifact export and persistent history. No training
starts on UI load. See [workbench.md](workbench.md) for API/root/resource
configuration and browser verification.

```bash
# Use Node >=22.12 (tested 24.19.0), with npm on PATH.
cd frontend && npm ci && cd ..
source .venv/bin/activate
python scripts/dev.py
# Open http://127.0.0.1:5173; Ctrl-C shuts down both services.
```

Start with **Heat / diffusion**, choose a registered checkpoint and review
**Use checkpoint layout** if needed. Edit known D and validate, then generate
explicit synthetic observations or import measured rows. Configure/validate the
complete reconstruction before running. Adding an unsupported term or leaving
an incomplete equation blocks execution. Full-request export/import retains
equations, declarations, boundaries, initial information, noise and observations.

Larger heat profiles use restartable float32 shards and bounded-memory training.
For the 2000/200/200 independent-instance MPS pilot:

```bash
CFG2="configs/experiments/mps_2d_pilot.json"
RUN2="$PWD/outputs/local-mps-2d-pilot"
pdeflow estimate-data --config "$CFG2" --samples 8 --output "$RUN2/estimate.json"
pdeflow preflight --config "$CFG2" --output "$RUN2/preflight"
pdeflow generate --config "$CFG2" --output "$RUN2/data"
pdeflow train --config "$CFG2" --data "$RUN2/data" --output "$RUN2/training"
# After interruption, with unchanged source/configuration:
pdeflow train --config "$CFG2" --data "$RUN2/data" --output "$RUN2/training" \
  --resume "$RUN2/training/checkpoint.pt"
```

Every checkpoint also writes portable inference.pt; no separate export command
is required. The full datasets and long budgets were not run during this task.
Actual intended-layout 32² and true 16³ MPS preflights passed, including guidance
and CPU reload. The 12000-instance standard and 1200-instance 3D profiles,
measured resource estimates, discovery/reconstruction commands and explicit
limitations are in [local-training.md](local-training.md). CUDA and actual
Colab remain untested.

The new local training-budget profile and accelerator configurations are
separate from accepted smoke outputs. Launch a fresh bounded study:

```bash
python -m pdeflow.experiments --config configs/experiments/local_budget.json \
  --output outputs/my-budget-study
python scripts/verify_backends.py --device mps --output outputs/my-mps-regression
```

The study evaluates exports at 0/100/500 updates along one training trajectory,
selecting guidance and baseline regularization only on validation cases.
[Actual pilot results](pilot.md) include the negative result: validation
CFM loss improves while reconstruction error is non-monotonic. The Fourier
inverse remains better. CPU and actual MPS float32 paths, guidance and CPU
reload were exercised; CUDA and Colab remain untested.

[Reference audit and verification](verification.md) records regression
results for absent, correct and unrelated references. Samples are identical;
only evaluation changes. [Darcy conventions](darcy.md) and the
[multi-specialist handoff](multispecialist.md) document actual 2D/3D execution,
current routing, larger MPS commands and the unfinished inverse benchmark.

## Online reference training and chemistry

The same trainer supports legacy offline datasets plus deterministic offline,
online and bounded hybrid physical-data providers. Online means sampling and
solving new clean physical problems, not training on model predictions. A saved
full training dataset is optional; numerical references, frozen training-only
calibration and stable evaluation manifests remain necessary.

[Online training/resume commands](online-training.md) cover all three
specialists, fixed held-outs, workers=0, bounded replay, cost accounting,
checkpoint registration, local MPS and Colab. [Chemistry conventions](chemistry.md)
describe the verified positive periodic reference and species/invariant handling.
The workbench exports data-mode configs/CLI commands without launching training.
The current task's evidence is in outputs/online-chemistry-20261021 and the
latest section of verification.md. Actual CUDA/Colab execution remains untested.

## Bounded process feedback

The diagram audit and executable feedback controller are documented in
[process-audit.md](process-audit.md) and
[process-verification.md](process-verification.md). Component-level checks
control subsequent configured gradients; actual final fields gate criteria-met
status. Missing thresholds or unresolved constraints produce budget-exhausted
status. The workbench shows decisions, reasons, final grid and budgets.
A real nested heat8²→16² reconstruction reselects a compatible fine checkpoint
and preserves actual observations. The subsequent FK continuation implements
real systematic resampling with a separately specified target. FK plus refinement
remains rejected. See [FK verification](fk-verification.md) for the supported
method and remaining diagram branches.

```bash
pdeflow smoke --config configs/experiments/process_heat_2d.json --output outputs/my-process-heat
pdeflow smoke --config configs/experiments/process_heat_fine_smoke.json --output outputs/my-fine-heat
```

Both are bounded real training/inference, not actions performed on opening the
UI. Process defaults have no acceptance thresholds and cannot report criteria
met. Exact existing-artifact CLI/refinement/workbench commands and scientific
limits are in process-verification.md.
