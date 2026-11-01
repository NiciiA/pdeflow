# Local workflow recovery and scientific workbench — October 25, 2026

The interrupted local-workflow recovery was completed **before** UI redesign.
Its restart/TIME_WAIT/live-port/reload/worker-ownership/clean-shutdown evidence is
in [everyday-verification.md](everyday-verification.md) and
`outputs/local-workflow-20261025/services.json`. Its 151-test run, six bounded MPS
layout probes, real SIGINT journey and external-wheel training are recorded
earlier execution, not new UI-task training claims.

New execution/evidence lives in `outputs/workbench-redesign-20261025/`.
No accepted experiment tree was overwritten. No long training budget was run.
The algorithms, Fourier/Darcy baselines, checkpoint semantics and scientific
targets remain as documented previously.

## Executed checks

| Check | Actual result / evidence |
|---|---|
| Python package | **155 passed**, 16 warnings, 101.28s; `python-final.log` |
| New visual API checks | **3 passed**, 15 warnings; `visual-api-second.log`: true 3D forcing/unknown fields, row errors, actual managed history, real weighted inference and reference isolation |
| Frontend unit tests | **31 passed**; `unit-final-31.log`, including navigation during pending validation, manual-validation/debounce races, reset/view persistence, unavailable reference metrics, distinct baseline tables, request status/defaults, specialist symbol roles, input marker selection and actual FK event coordinates |
| TypeScript/Vite | Build succeeds; bundle-size advisory remains |
| ESLint | Zero errors; three existing shadcn component export warnings |
| Real redesigned browser journeys | **12 distinct cases passed**, across `browser-fifth.json` (10 passed, chemistry failed before fix), `browser-affected.json` (A/K passed), `browser-chemistry-recovered.json` (chemistry passed) |
| Expert browser regression | **17 distinct cases passed**, using affected reruns after disclosure/selector fixes; per-case evidence in `browser-completion.json` |
| Final-source simple CLI | One-update stop, same-run resume to20 and one-case validation evaluation; `final-source-{train,resume}.log`, `final-evaluate.log` |
| Whitespace / cleanup | Working tree and original staged index checks pass; API8000/frontend5173 released, zero active jobs; `cleanup.json` |
| Packaging | Source ZIP CRC/payload hashes and required files checked; current wheel payload matches every Python source file and includes all eight presets |
| CPU | Real training, inference/guidance/FK, API/browser, reference-isolation and cleanup |
| MPS | Prior local-workflow execution: two updates at all six Mac layouts; actual device continuation. No new MPS quality claim from UI tests |
| CUDA / actual Colab | **Untested** |

Exact commands (repository root, existing environment):

```bash
source .venv/bin/activate
python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run lint
npm --prefix frontend run e2e -- design.spec.ts --timeout=120000
```

Real browser acceptance used the live API, preserved compatible heat1D/2D/3D,
Darcy2D/3D and chemistry2D/3D exports, plus automatically discovered managed runs.
The isolated QA workspace stores links to these exports; it does not move them.
For an ordinary clean install, generate the short quickstart checkpoint, then
use `pdeflow serve`; QA artifacts are not a startup dependency.

## Browser coverage and visual inspection

A: real guided heat equation, unknown-input preview, selectable observations,
numerical reconstruction, optional truth and desktop/narrow result panels.
B: supported rearrangement, diffusivity edit, real CSV export/import and run.
C: unsupported source and invalid JSON survive refresh and cannot execute stale
mathematics. D: Darcy2D/3D forcing arrays and permeability/pressure roles; static
slices have no physical-time control. E: chemistry species meanings, parameter
edit, observations/species/time and real run. F: true heat3D input/output planes
and explicitly attached reference/error. G: real eight-particle FK, weighted
observation prediction, recorded ESS, ancestry and resampling. H: actual managed
training/validation history and discovered checkpoints. I: durable history,
refresh and edited-input banner while prior results remain available. J: row
import failure, wrong-family override and real cancellation. K: identical-task
quality/cost comparison; mismatched heat/Darcy comparison rejected.

`browser-completion.json` maps all29 unique passing cases to their actual reports. Later state-management checks are in `browser-state-verified.json` (4 passed); final A/MathLive/FK checks are in `browser-final-affected.json` (5 passed). The passing expert cases are distributed across `browser-regression-verified.json`, `browser-regression-remaining.json`, `browser-regression-affected.json` and the final affected report. Failed/interrupted attempts remain separately recorded; they are not counted as passes.

Actual screenshots are under `browser/`: `define-desktop.png`,
`observations-desktop.png`, `results-{desktop,narrow}.png`,
`results-focus.png`, `results-narrow-focus.png`, `darcy-{2d,3d}.png`,
`chemistry.png`, `heat-3d.png`, `fk.png`, `training-history.png`,
`comparison.png`, `unsupported-preserved.png`.
Inspected the real app with the in-app browser and rendered desktop/narrow PNGs.
Narrow stage-label clipping was corrected; horizontal page overflow checks pass.
Known/unknown input data, weighted output plots and raw histories were inspected.

## Failures retained and corrected

The initial browser launch required installation of the locked Playwright
Chromium build. Early new-test selectors used names/strict exact text that did
not match actual accessible controls and paragraphs. Running two browser suites
against the same output directory caused trace-file contention; simultaneous
full tests/numerical exports also exceeded a30s assertion. Those attempts remain
recorded. Final suites run sequentially, with bounded numerical waits; no
numerical tolerances were relaxed. A reusable generated Matplotlib font cache
was used for later browser exports; default independent startup was exercised
in the prior recovery gate.

A full Python run found optional job-display metadata breaking low-level
worker-only requests. Display metadata is now conditional; the focused lifecycle
regression and final full suite pass. A real chemistry browser journey exposed
cancellation of pending equation validation during stage navigation. The
canonical editor now remains mounted while hidden; only its current visible
preview mounts in the definition stage. A frontend regression and real chemistry
rerun verify this fix. Manual incompatible model selections no longer receive
a misleading compatible-model summary. Reset returns to definition, workspace stage persists separately from saved-result selection, and superseded debounce callbacks exit before touching report state. Real MathLive focus is asserted without unrelated test-helper clicks stealing it. Baseline error tables have distinct accessible labels.

## Final-source quickstart and scientific meaning

The final managed run is
`outputs/local/runs/20261025T194431-workbench-final-adfe16b1/`.
It stopped after update1 and resumed to20; strict training source identity
matches the final Python package. Older exports remain inference-compatible,
but their training resume still requires the source recorded in each checkpoint.

One **validation** case, two generated samples, default20-update quickstart:

| Method | Observation RMSE | Finite-difference PDE RMSE | Relative unobserved L2 of mean |
|---|---:|---:|---:|
| Unguided |1.46636|30.7865|1.58449|
| Observation guidance |0.602541|28.6562|1.50588|
| Observation + PDE guidance |0.601130|28.3169|1.50201|
| Conventional Fourier inverse |0.125418|0.210187|0.522088|

Observation RMSE is the square root of weighted mean squared residual over candidates and measured points. PDE RMSE applies the same population aggregation to finite-difference physical residuals. Unobserved relative L2 compares the population mean to optional truth, using grid-cell and physical-time quadrature, excluding measured nodes and a1e-12 denominator floor. Mean-prediction scatter plots deliberately differ from the population RMSE, which also includes candidate variability.

These are execution/diagnostic checks, not scientific validation or a test-based
selection. Full metrics/costs are in the run's
`evaluations/20261025T194706-af4d1acf/`. The conventional inverse remains better.
The retained0/100/500 study remains non-monotonic across test budgets; no winning
training budget is selected by test performance. See experiments.md/pilot.md.

## Limitations and next slice

Fixed periodic Cartesian specialist layouts remain required. General neural
equation understanding, arbitrary geometry/boundaries, generic resolution
generalization, calibrated uncertainty and exact learned posterior guarantees
are absent. FK plus refinement/retry remains rejected. Live polling can miss
very rapid intermediate events; saved final histories remain authoritative.
Linked external exports do not acquire fabricated histories or training-range
metadata. Browser drafts depend on local storage; export larger drafts for
transfer. MathLive keeps the initial bundle large, and figure-cache warmup can
make a first numerical export slower. Full Mac pilots, CUDA and actual Colab
execution remain unverified.

Next complete scientific slice: a validation-planned larger Darcy training and
inverse-baseline comparison with fixed held-outs, observation budgets, multiple
training/sampling seeds and measured cost; preserve conventional baseline
advantages and report failure cases. Do not promote a smoke checkpoint based
on its attractive plots or low residual.
