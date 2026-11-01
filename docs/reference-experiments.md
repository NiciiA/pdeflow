# Reference experiments and physical comparison

The workbench has four explicit actions: **Solve reference**, **Sample
measurements**, **Reconstruct from measurements**, **Compare**. Reference solving
requires a recognized current equation and complete physical inputs, but no
neural checkpoint. A compatible checkpoint is required for reconstruction.
Manual measurement entry/CSV/JSON import remains available without a reference.

## Local workflow

Activate the installed package from the repository root:

```sh
source .venv/bin/activate
pdeflow train --preset quickstart-heat-2d --device cpu --steps 2
pdeflow serve
```

In another terminal, from the repository root:

```sh
cd frontend
npm run dev -- --port 5173 --strictPort
```

Open http://127.0.0.1:5173 (API http://127.0.0.1:8000). Use the supported Node
version in .nvmrc, not a tool's private runtime. A combined launcher is
`python scripts/dev.py` with the venv active and Node/npm on PATH. It starts no
training. `PDEFLOW_WORKSPACE` or CLI `--workspace` selects a separate workspace.

1. Insert Heat/diffusion, inspect parameters, domain, periodic conditions and
   saved physical times. Default heat example is 8×8, four frames, D=0.07;
   the `quickstart-heat-2d` checkpoint above matches it. Suites use larger grids.
2. In Reference experiment click Load visible reference defaults. The JSON is
   editable and contains **all** reference-only initial fields. Preview supplied
   field before solving. Auto with explicit finite Fourier modes selects the
   supported analytic evolution. No inputs are inferred invisibly.
3. Solve reference; inspect fields/times/slices and solver diagnostics. Every
   successful solve gets a new immutable ID. Changed equation/domain/parameters/
   forcing/initial arrays/solver/accuracy mark the previous reference stale.
4. Select inverse disclosure and measured fields/times, pattern/count/seed/noise.
   Sample measurements. The existing observation table shows noisy values and
   coverage. Show reference truth changes display only.
5. Reconstruct from measurements opens the existing numerical configuration.
   Validate complete request, then Run reconstruction. Two particles/four steps
   suffice for an execution check; they do not establish model quality.
6. Associated synthetic truth is compared automatically **after** inference. Compare remains available to change evaluation policy or reattach explicitly. The retained reference and mean
   share plot ranges; field/time/slice controls coordinate reconstruction,
   reference, error and observed locations. Physical diagnostics, controller/FK
   traces, sample/spread views and conventional baselines remain available.
   The comparison table separates observed, unobserved and full-domain errors.
   Select MSE/RMSE/relative-L2 for per-time charts; undefined values stay absent.
7. Export arrays, comparison JSON, measured CSV, inverse request or complete
   experiment ZIP. The ZIP contains the request, result, baseline artifacts,
   attached complete reference, matching sampling and clean evaluation-only
   samples, plus a manifest stating that truth was excluded from the worker.

Darcy: use the forcing editor and periodic cell centres. Full permeability is a
reference input; choose `a: unknown`, `u: partially_measured`, select only u for
pressure-only observations. Static fields have no physical-time control. Separate
a/u/log-a metrics retain the zero-mean pressure gauge without truth alignment.
Chemistry: supply all three positive initial species fields; measure only chosen
species and mark the others unknown. Rates/diffusivities remain known.

## Supported solver inputs

| Family/layout | Reference input | Algorithm and classification |
|---|---|---|
| Heat 1D/2D/3D periodic nodal | initial_u | Explicit finite Fourier modes: analytic spectral evolution of that exact finite-mode continuum initial condition. Array/expression input: numerical evolution of its grid spectral interpolant. Neither is a general symbolic PDE solver. |
| Darcy 2D/3D periodic cell centres | permeability; f fully specified in request | Conservative harmonic face fluxes, projected CG; a>0, mean-zero forcing, zero-mean pressure. Numerical reference; no analytical option. Default rtol=1e-10, max_iterations=4000. |
| A+B⇌C chemistry 2D/3D periodic nodal | initial_a, initial_b, initial_c | Positive FD uniformization diffusion / exact local reversible reaction / Strang splitting. Numerical reference; no analytical option. max_step=0.0025, max_iterations=4000; saved positivity/stoichiometric diagnostics. |

Inputs are scalars, spatial arrays, `{ "expression": "exp(0.3*cos(2*pi*x))" }`
or `{ "fourier": { "offset": 0.2, "modes": [ { "wave": [1,1],
"amplitude": 0.6, "phase": 0 } ] } }`. Fourier wave numbers are integers
strictly below Nyquist per axis. Expressions use physical coordinates x/y/z,
pi, sin/cos/exp/sqrt/tanh/abs, arithmetic and bounded constant powers. No eval,
attribute access, comprehensions, imports or arbitrary calls. JSON files can be
imported into the reference input editor; large arrays are exported numerically.
Boundary conditions other than supported periodic Cartesian ones are rejected.
Accuracy options are family-specific; unused settings are rejected.

## Low-level CLI and complete experiments

Use fresh output directories; existing artifacts are never overwritten:

```sh
pdeflow reference-example --preset heat-2d --output heat-problem.json
pdeflow reference --problem heat-problem.json --output outputs/my-reference
pdeflow observe --reference outputs/my-reference --sampling configs/reference/sampling.json --output outputs/my-observations
pdeflow experiment --reference outputs/my-reference --sampling configs/reference/sampling.json --model auto --output outputs/my-experiment --device cpu
pdeflow compare --result outputs/my-experiment/reconstruction --reference outputs/my-reference --output outputs/my-comparison.json
```

`--reference` also accepts an artifact ID from the same `--workspace`.
`--model` accepts an inference.pt path, discovered content ID or auto. Auto
selects a compatible available export deterministically, without quality ranking.
`--preset heat-2d` (also heat-1d/3d, darcy-2d/3d, chemistry-2d/3d) replaces
`--reference` for a one-command experiment. Examples in configs/reference have
editable full requests. `--stop-after reference` or `--stop-after observations`
requires no checkpoint. `--inference configs/reference/inference.json` sets a
bounded sample/step budget. Existing low-level reconstruct/evaluate commands
remain available; `compare` uses full coordinate-aware comparison metadata.

For Darcy pressure-only synthetic inference:

```sh
pdeflow experiment --problem configs/reference/darcy-2d.json --sampling configs/reference/darcy-pressure.json --model outputs/multispecialist-20261021/cpu-2d/training/inference.pt --output outputs/my-darcy-experiment --device cpu
```

The historical path is optional; on a fresh checkout train the corresponding
layout or edit the explicit reference grid to match the selected export. A
32² pilot cannot silently execute an 8² request. `pdeflow models --import PATH`
links a CPU-remappable inference export (also useful after Colab transfer).

## Contracts and provenance

ReferenceProblem reuses the current mathematical request and separates full
reference-only fields in `inputs`. ReferenceSolution is an immutable directory:
reference-problem.json, problem.json, reference.npz `[1,C,T,*space]`, and
reference.json. Its manifest records canonical equations, physical coordinates,
actual times, units, solver/accuracy/diagnostics, source/specification and array
hashes, seed and timestamp. Load validates numerical/specification checksums and
physical metadata. Browser previews return a bounded plane (≤65536 nodes), never
a complete 3D array. Public summaries redact large known spatial arrays.

ObservationSamplingSpec produces noisy ObservationSet plus the existing
ProblemSpec/mathematical request as the inverse problem. Only noisy measurements
are adopted. Full initial states and hidden permeability never enter known_fields,
normalization, sampler, controller or stopping. Scalar parameters and forcing must
remain fully known for current checkpoints. Fully known target-field conditioning
is unsupported and rejected with a capability explanation. Reference visibility,
metric choice and reference attachment are absent from ReconstructionRequest.
EvaluationReport is generated after the existing ReconstructionResult.

Sampling supports fixed spatial sensors over selected saved times, independently
random spatial sensors per field/time, regular flattened-grid sensors, uniformly
random unique field-space-time points, explicit positions and exact reuse of prior
indices. Spatial count is per selected field/time; space_time count is total;
positions/reuse length is authoritative. Fields and time indices are explicit.
The shared differentiable on-grid PointOperator performs H; no coordinate snapping
or temporal interpolation. Independent SeedSequence streams drive sensor selection
and Gaussian noise. Clean H(Uref) is a separate evaluation-only NPZ.

## Error and transfer conventions

MSE=Σw e²/Σw; RMSE=√MSE; MAE=Σw|e|/Σw; relative L2=√(Σw e²/Σw U²).
Weights are Cartesian cell volumes and trapezoidal saved-physical-time weights.
Static fields use one spatial integral. Masks exclude **exact measured indices**
per field/time, with no invented observed area. Empty regions yield null. Relative
L2 is null if reference RMS norm ≤1e-12; absolute errors remain available.
Error of the weighted population mean and weighted mean of individual errors are
separate. No cross-species aggregate score or truth-selected best particle.
Weighted empirical spread is not calibrated posterior uncertainty.

Native comparison checks field names/units, equation, physical parameters,
known fields, domain origins/extents/boundary, grid convention and saved times.
`--policy restrict_reference` supports integer nested grids: coincident periodic
nodes or conservative cell-volume averages. Known forcing/coefficient fields must
also agree after the declared restriction; otherwise comparison is rejected.
`--shared-times-only` evaluates the exact intersection of saved times, no temporal
interpolation. Report/export retains all three grids and transfer rules. A shared
subset can produce metrics without embedding a full-grid truth plot. Coarse/fine
discretization error is not solely learned-model error. Unsupported interpolation,
arbitrary meshes and target layouts remain explicit limitations.
