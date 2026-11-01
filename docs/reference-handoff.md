Current full-catalogue/native512 workflow: [training-catalogue.md](training-catalogue.md). The commands below retain their original six-target historical scope.

# Reference experiments and suites: runnable handoff

Run these commands from the repository root using the supported Python3.12 venv
and Node22.23.3 on PATH. Existing accepted experiments/checkpoints are preserved.
For installation on another machine: `python -m pip install -e '.[workbench,test]'`
then `pdeflow setup` (or --headless without the frontend).

## Start and exercise the workbench

```sh
source .venv/bin/activate
pdeflow train --preset quickstart-heat-2d --device cpu --steps 2
pdeflow serve
```

The training command is needed only when no compatible8² heat export exists.
In another terminal, from the repository root:

```sh
cd frontend
npm run dev -- --port 5173 --strictPort
```

Open http://127.0.0.1:5173. API: http://127.0.0.1:8000.
Combined alternative: `python scripts/dev.py` with the venv active. No service
starts training automatically. Use the same --workspace/PDEFLOW_WORKSPACE for
training/discovery and serving. Imported external checkpoints remain supported.

Shortest workflow: insert Heat/diffusion → inspect the explicit physical layout →
Load visible reference defaults → Preview supplied field → Solve reference →
choose disclosure/fields/times/count/noise → Sample measurements → Reconstruct
from measurements → Validate complete request → Run reconstruction → Compare.
Reference visibility is display-only. Reference and clean samples never enter
the inference request. Darcy pressure-only: mark a unknown, uncheck measured a,
keep u partially measured. Chemistry: measure chosen species, leave others unknown.
Current fixed-grid checkpoints must match the exact resolved request layout.

## CLI physical reference, observations, inverse and metrics

```sh
pdeflow reference --problem configs/reference/heat-2d.json --output outputs/my-reference
pdeflow observe --reference outputs/my-reference --sampling configs/reference/sampling.json --output outputs/my-measurements
pdeflow experiment --reference outputs/my-reference --sampling configs/reference/sampling.json --model auto --output outputs/my-experiment --device cpu
pdeflow compare --result outputs/my-experiment/reconstruction --reference outputs/my-reference --output outputs/my-comparison.json
```

One command: `pdeflow experiment --preset heat-2d --model auto --output
outputs/another-experiment --device cpu`. Put it on one shell line. Or use an
explicit `--model PATH/TO/inference.pt`. Stop before inference with
`--stop-after reference` or `--stop-after observations`; these need no weights.
Use fresh output paths. `reference-example --preset heat-2d --output problem.json`
exports editable full inputs. Native comparison is default; explicit fine-grid
restriction uses `compare --policy restrict_reference` and optional
`--shared-times-only`, subject to coordinate/known-input checks.

Numerical arrays: reference/reference.npz, observations/observations.json,
reconstruction/result.npz; report: reconstruction/comparison.json; complete
manifest: experiment.json. UI Export complete experiment creates a ZIP of request,
results, baseline, attached reference and matched sampling provenance. Reference
NPZ/inputs, noisy CSV/inverse request, metrics and samples also export separately.

## Plan, train, resume and inspect a finite collection

```sh
pdeflow train-all --plan
pdeflow train-all --suite quickstart --device cpu
pdeflow train-all --suite local-pilot --device mps
pdeflow resume-suite latest
pdeflow suite-status latest
```

Optional: `--workspace outputs/my-models`, `--include heat-2d,darcy-2d`,
`--exclude chemistry-3d`, `--seed 37`, repeated `--budget heat-2d=1000`,
`--reuse` (identical completed config/source only), `--force` (new children),
`--continue-on-error`, `resume-suite --retry-failed`. No confirmations per child.
Default pilot: heat/Darcy/chemistry2D then3D;2D3000 updates/batch2,3D1500/batch2.
Quickstart: same explicit grids,2 updates/batch1. Grids:32²,32²,16²,16³,16³,8³.
Only bounded quickstarts were executed here. Ctrl-C saves the active child safely;
resume preserves completed children, optimizer/provider/RNG/calibration/held-outs.
Strict source/config identity rejects incompatible continuation. Models & Training
shows discovered exports, durable suites and linked per-child history.

## Short verification

```sh
python -m pytest tests/unit/test_reference_comparison.py tests/integration/test_reference_suite.py -q
cd frontend
npm test
npm run build
npm run lint
npx playwright test e2e/reference.spec.ts
```

Browser tests need the running API/frontend with compatible heat1D16-node,
heat2D/3D8-node, Darcy2D/3D8-node and chemistry2D/3D8-node exports. Suite-discovery
journey additionally needs a completed managed quickstart suite in that workspace.
The historical accepted exports linked during this task remain available on this
machine; on a fresh machine generate matching smoke configurations or choose the
matching layout explicitly. Component mocks do not count as numerical workflows.

## Status and next slice

Heat has a supported analytic finite-Fourier case and numerical grid spectral
interpolant; Darcy and chemistry references are numerical approximations with
saved diagnostics. Implemented specialists are heat1D–3D, Darcy2D/3D and
chemistry2D/3D. Available and smoke-trained exports are distinct from scientifically
evaluated models. In the saved CLI probe Fourier inverse remains much better
(unobserved RMSE0.01514 versus0.71129). Prior0/100/500 test quality remains
non-monotonic; no test result chose a training budget.

Actual tests/hardware, output paths, Git and packaging status are in
[reference-verification.md](reference-verification.md) and
outputs/reference-suite-20261026/completion.json. CPU and bounded MacBook MPS
were exercised. CUDA/actual Colab remain untested. Existing Colab notebook installs
this package, prepares persisted paths, launches/resumes/exports; transfer
inference.pt back to the Mac and `pdeflow models --import PATH` before serving.
No cloud execution was performed for this task.

Remaining limits: periodic Cartesian grids, exact-grid/saved-time observations,
finite declared solver capabilities, compatible fixed layouts, no fully-known
unknown-target/unknown-parameter conditional specialist, no general symbolic
solver or arbitrary mesh/temporal interpolation. Empirical spread is uncalibrated.
The next complete slice is validation-designed repeated-seed scientific evaluation
of trained specialists versus conventional baselines, including numerical-reference
resolution/tolerance studies and held-out testing without test-driven selection.
