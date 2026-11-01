Latest continuation: [Coordinate specialist and source workflow](coordinate-specialist.md), with [actual verification](coordinate-verification.md),27 October2026. Earlier handoffs below are historical.

Latest continuation: [Training catalogue, native512 and monitoring handoff](catalogue-handoff.md) (26 October 2026). Earlier records below remain historical.

Latest continuation: [Flow workspace, model selection and capacity handoff](flow-workspace-handoff.md) (26 October 2026). Earlier dated handoffs remain historical.

Latest continuation: [Reference experiments and training suites handoff](reference-handoff.md) (26 October 2026). Earlier dated material below remains historical.

# Recovery handoff — 2026-10-21

This is the earlier heat recovery record. The current executable heat/Darcy
handoff and exact commands are in [multispecialist.md](multispecialist.md).
Earlier scientific measurements below are historical and unchanged.

This is the earlier recovery record. The subsequent symbolic editor now starts
with empty equations: insert Heat before applying a checkpoint layout. Current
grammar/workflow is in symbolic-editor.md and expanded MacBook training commands
are in local-training.md. Earlier scientific measurements below are unchanged.

Implementation was complete before this recovery. No scientific or frontend
implementation changed here and no training, pilot or browser run was repeated.
Recovery corrected source packaging and documentation. Existing accepted runs,
pilot outputs, screenshots and earlier archives were preserved.

## Working scope

The package executes real conditional flow-matching training, save/resume,
export and sparse reconstruction of periodic scalar heat in 1D–3D. Euler/Heun,
uniform/quadratic schedules and unguided/observation/observation+PDE modes are
implemented. Independent candidates are separate from physical time and space.
Fourier inverse reconstruction is the conventional baseline. Poisson reference
numerics are verified; neural Poisson inference is unavailable.

The local workbench edits canonical problem/operator specifications, imports
strict physical-coordinate CSV/JSON measurements, discovers compatible local
checkpoints and runs a managed numerical subprocess. It supports progress,
cancellation, durable history, artifact export, 1D curves, 2D maps and true 3D
orthogonal slices. References are attached only for evaluation; they do not
enter inference. Spread is descriptive, not calibrated uncertainty.

## Start on the existing MacBook checkout

From the repository root, with Python 3.12 and Node >=22.12 on PATH:

```bash
source .venv/bin/activate
npm --prefix frontend ci
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training:outputs/workbench-20261021/cpu-1d-regression/training"
python scripts/dev.py
```

The existing environment and dependencies are installed. `npm ci` is needed
only when installing/restoring frontend dependencies. The recovery shell's
default Node was too old; use a supported Node installation. Tested versions:
Python 3.12.3, Node 24.19.0, npm 10.2.4. The launcher prints its URLs and Ctrl-C
stops both child process groups. Open **http://127.0.0.1:5173**. The API is at
**http://127.0.0.1:8000**, with interactive schema at `/docs`.

For separate terminals (same working directory and environment):

```bash
.venv/bin/python -m pdeflow.workbench.api
```

```bash
npm --prefix frontend run dev
```

Default API discovery already finds the accepted 2D/3D training folders.
The explicit roots above also expose the 1D run. All three exports and their
public observations were safely loaded and compatibility-checked on CPU during
recovery. API/job roots and resource limits are documented in workbench.md.

## Shortest workbench reconstruction

1. Select the accepted 2D heat checkpoint (8×8, four saved physical times,
   30 optimizer updates), apply its layout, and validate the problem.
2. Continue to Observations and use the explicit synthetic-observations action.
   It generates compatible reference data and noisy measurements. Detach the
   reference to demonstrate reconstruction using measurements alone. Manual
   rows or CSV/JSON imports are alternatives; see workbench.md for schemas.
3. In Reconstruction choose CPU, two samples, four generation steps and
   observation+PDE guidance. Validate the complete request, then run.
4. Inspect Results, change physical time/sample/mean/spread, and download NPZ,
   metrics and configuration. Reopen the result from Runs after a refresh.
   Reattach reference data only if reference-based error is wanted.

This uses existing exports and requires no training. For a fresh source ZIP,
generate `outputs/demo-2d` with the command in SOURCE_PACKAGE.md, register
`outputs/demo-2d/training`, then follow the same flow. The ZIP omits checkpoints
and accepted outputs deliberately.

## CLI reconstruction and evaluation

Run from the repository root after activating `.venv`. Use a new output path:

```bash
RUN="outputs/handoff-$(date +%Y%m%d-%H%M%S)"
pdeflow reconstruct --config configs/experiments/cpu_2d.json --device cpu \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --problem outputs/accepted_cpu_2d/data/test_problem.json \
  --observations outputs/accepted_cpu_2d/data/test_observations.json \
  --reference outputs/accepted_cpu_2d/data/test_reference.npz --output "$RUN"
pdeflow evaluate --result "$RUN/observation_pde"
```

Reconstruction produces all three sampling modes and the Fourier baseline.
Omit `--reference` for inference without truth. Evaluation truth is read after
every reconstruction has completed. Evaluation updates metrics in the selected
result directory. For fresh ZIP demos replace `outputs/accepted_cpu_2d` with
`outputs/demo-2d`. Training data are unnecessary for inference.

## Colab and transfer

Open notebooks/colab.ipynb in Colab. Upload the final source ZIP and set
`SOURCE_ARCHIVE` to its uploaded path, leaving `REPOSITORY_URL` blank. The
alternative is your own repository URL plus an exact 40-character commit SHA;
this checkout has no commit or remote to supply. Choose Python 3.11/3.12
(3.12.3 tested locally), a disposable `CACHE_ROOT`, and an explicitly configured
persistent output directory. No storage account or mount is assumed.

Run the cells in order: isolated install and hardware inspection; select
`PROFILE`, `DEVICE` and `RUN_NAME`; validate; restore/generate data; train/resume;
export; reconstruct/evaluate. Start with `PROFILE="cpu_2d"`, `DEVICE="cpu"`.
Use `gpu_2d`/`gpu_3d` with `DEVICE="cuda"` only on suitable hardware. The larger
profiles are unexecuted configurations. Unchanged source/config/data and the
same run name resume the last completed persistent checkpoint. Use a new name
for a different configuration; interrupted uncheckpointed updates are lost.

Copy the entire `<persistent output>/<RUN_NAME>/portable-export/` directory
back to the MacBook. It contains `inference.pt`, `config.json`,
`test_problem.json`, `test_observations.json`. From that directory with the
local package environment activated:

```bash
pdeflow reconstruct --config config.json --device cpu --checkpoint inference.pt \
  --problem test_problem.json --observations test_observations.json \
  --output reconstruction-cpu
pdeflow evaluate --result reconstruction-cpu/observation_pde
```

Choose another output name on repetition. No training array is required.
Notebook compilation and archive/cache/path setup logic were tested locally;
**actual Colab and CUDA execution remain UNTESTED**.

## Verification: previous execution versus recovery

| Check | Recorded previous execution | Recovery session |
|---|---|---|
| Python suite | 45 passed, 30.91 s; 15 upstream warnings | 45 collected; cached failure map empty; suite not repeated |
| Frontend unit | 8 passed | 8 passed again |
| TypeScript/production build | Passed | Passed again (`npm run build` includes `tsc -b`) |
| Lint | 0 errors, 3 generated-component warnings | Same result again |
| Chromium workflows | 4 passed, 50.5 s | Passed last-run record, four test definitions, screenshots and actual jobs inspected; not rerun |
| Launcher | Both HTTP 200; exit 0, clean shutdown | Saved JSON/log inspected; no new service launch |
| CPU | 1D/2D/3D training/resume/reconstruction and isolation tests | Existing exports loaded and observation compatibility checked |
| MPS float32 | 6 updates, resume, endpoint/guidance, reconstruction, export/CPU reload | backend_verification.json inspected; no accelerator rerun |
| CUDA / Colab | UNTESTED | UNTESTED |
| Source package | Earlier ZIP/wheel/sdist existed | Final ZIP extracted; CRC, per-file hashes, completeness/exclusions and launch-script syntax checked |

The full Python test stdout was not separately retained; verification.md and
the previous execution record report 45 passes, supported by 45 cached test
IDs and an empty failure cache. This is recorded evidence, not a new full run.
Playwright retains a passed last-run status without per-test timing details;
the four-workflow count/time is from the previous execution record. Persisted
API history contains ten successful runs (1D: one; 2D: five; 3D: four) and three
cancellations. These counts include manual QA as well as browser tests.
Screenshots and jobs corroborate those paths, without proving browser behavior
from screenshots alone. The recovery socket probe was permission-restricted,
so current port closure is not independently claimed; clean shutdown is the
recorded launcher result. No expensive scientific verification was repeated.

Recovery logs: outputs/workbench-20261021/recovery/{command-checks.json,
artifact-checks.json,frontend-unit.log,frontend-build.log,frontend-lint.log,
python-collection.log,archive-checks.json,recovery-checks.json}. Scientific,
frontend, test/config and notebook source matched the earlier final source ZIP
before recovery. Only documentation, source-package metadata and the packaging
script changed. Whitespace checks cover both existing staged and new changes.

## Training-budget result

One training trajectory supplies exports at 0/100/500 updates, with 120 split
trajectories, 16×16 nodes, six physical times, one training seed and two fixed
test cases. Two validation cases at the final budget select guidance weights
1/.01 and ridge .1; these settings are frozen across every test budget.
There is no test-selected winning training budget.

| Updates | Training CFM objective | Validation CFM objective | Unguided error | Observation error | Observation+PDE error | Fourier error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.262254 | 2.332563 | 1.120581 | 1.125430 | 1.109725 | .236383 |
| 100 | .775055 | .879483 | .949002 | .756363 | .751510 | .236383 |
| 500 | .453315 | .574266 | 1.632448 | 1.190294 | 1.189886 | .236383 |

Errors are averages over the two test cases of the relative L2 error of the
four-sample ensemble mean on unobserved values, using spatial-cell volume and
physical-time trapezoidal weights, with squared reference norm floored at
1e-12 before taking the square root of the ratio. The CFM
objective is mean squared velocity prediction error using fixed objective
noise/times. Observation RMSE and PDE residual RMSE are distinct diagnostics,
in physical fields; detailed values, individual-sample errors, spread, mass,
measured times/calls/RSS and definitions are in pilot.md and mathematics.md.
Each generative comparison uses 24 Heun steps (48 unguided / 71 guided batched
model calls); guided runs also use 24 gradient evaluations. The complete pilot
took 31.6107 s, including comparisons and plots.

**The conventional Fourier inverse remains better. Test quality was
non-monotonic across budgets despite falling validation objective.** Lower
residual at 500 updates did not imply better reconstruction. These results do
not isolate the cause or establish scientific convergence or calibration.

Machine-readable evidence is in outputs/workbench-20261021/budget-study/:
study.json, objectives.json, selection.json, metrics.json, data/manifest.json,
fixed_inputs/, budget_{0,100,500}/, and learning_curves.png, quality_budget.png,
quality_cost.png, ensemble_spread.png. The pilot-source.zip snapshot is retained.

## Locations, packaging and Git

Existing accepted outputs: outputs/accepted_cpu_2d/, accepted_cpu_3d/ and
accepted_portable_euler_quadratic/. New study/dimension/MPS outputs and browser
screenshots: outputs/workbench-20261021/. API job history: outputs/workbench/.
The complete final source delivery is:

`outputs/workbench-20261021/recovery/pdeflow-source-final.zip`

It includes frontend source/lockfile, Python, configs, tests, scripts, metadata,
notebook and documentation; SOURCE_PACKAGE.md explains regeneration of omitted
data/weights, and SOURCE_MANIFEST.json contains payload checksums. Existing
ZIPs and wheel/sdist in outputs/workbench-20261021/dist/ remain historical
build outputs. Use the final ZIP for the complete current source/docs.

Git is on branch `main` with **no commits and no remote**. Ninety files were
already staged as initial additions. Recovery leaves that index intact:
documentation/package-script updates are unstaged and new documentation files
are untracked. Nothing was reset, discarded, pushed or published. The prior
commit action was rejected by automatic approval review after its usage limit;
it was not retried or bypassed. Delivery does not require a commit.

## Limits and next complete slice

Known constant positive diffusivity, periodic scalar heat, exact saved
grid/time layout and on-grid observations are the learned scope. No off-grid
interpolation, unknown-parameter reconstruction, resolution generalization,
Darcy/chemistry execution, neural Poisson, diffusion/CFG, FK or refinement is
claimed. The API is a localhost research service. MPS is float32; CPU remains
the portable reference/inference backend. CUDA and Colab need real hardware
verification. This one-seed/two-test-case pilot supports engineering checks,
not a model-quality conclusion across populations.

The next complete implementation task is the conservative 2D/3D
variable-coefficient Darcy slice in darcy.md: positive permeability, fluxes,
gauge/boundary compatibility, independent convergence checks, inverse baseline
and genuine trained reconstruction evaluation. Follow with the planned
A+B↔C chemistry slice and its correct invariants. More seeds and larger held-out
sets are needed before interpreting heat quality trends.
