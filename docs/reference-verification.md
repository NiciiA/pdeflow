# Reference experiments and sequential suites: verification, 26 October 2026

Evidence root: `outputs/reference-suite-20261026/`. Existing accepted output roots,
weights, datasets and earlier screenshots/archives were preserved. This is a new
implementation slice, not a repeat of the earlier training-budget studies.

## Actual checks

| Check | Actual execution / saved evidence |
|---|---|
| Reference/unit checks | Initial20 passed in1.80s; final added numerical heat convergence and gauge rejection. Included in35 affected final passes. Analytic finite Fourier 1D/2D/3D, solver classification, true3D references, safe AST rejection, exact/noisy reproducible H mapping, all patterns/reuse/positions, disclosure, quadrature/masks, weighted means, null norms and coordinate transfer. |
| Complete Python regression | Initial180-case run:178 passed,2 failed in100.13s (`python-all-first.log`). The two failures were legacy isolation fixtures with all-ones Darcy pressure, rejected by the new zero-mean gauge validation. Fixtures corrected to positive a/zero-mean pressure, without weakening validation. |
| Final clean Python regression | **182 passed in103.61s**, zero failures/skips (`python-delivery-all.log`), on the delivered source including the early input-allocation guard. |
| Final affected Python | `pytest tests/unit/test_reference_comparison.py tests/integration/test_fk_pipeline.py tests/integration/test_process_pipeline.py tests/integration/test_reference_suite.py -q`:35 passed in50.48s (`python-affected-final.log`). Includes both formerly failing cases, new convergence/gauge test and all reference/suite integrations. The final early-allocation guard adds another regression; python-reference-last.log records27 further affected passes. A clean final full pass is recorded below. |
| Real API integration | Three families, two actual identical inverse reconstructions each, changed evaluation reference/metrics/attachment, identical particles/weights/process/cost/stopping, strict worker schema, matched sampling, numerical ZIP/CSV export. `api-permitted.log`:3 passed; final affected run rechecks them. |
| Frontend |33 tests, six files, actual TypeScript/Vite build, ESLint zero errors/three existing component-export warnings. `frontend-verified.log`, `build-verified.log`, `lint-verified.log`. Production bundle size warning remains. Unit mocks establish component behavior only, not numerical integration. |
| Real Chromium | **9 passed in48.1s** on delivered source (`browser-delivery.log`): heat1D/2D, Darcy2D/3D, chemistry2D/3D full reference→sampling→actual worker→comparison; unsupported analytical/disclosure rejection; suite discovery/child history; reference-free manual CSV→actual reconstruction. Desktop/narrow screenshots in browser/. |
| Real CPU suite | Final six-entry suite,2 actual updates each/batch1, original explicit pilot grids. `suite-delivery-cpu-status.json`, `suite-delivery-cpu.log`. Completed, OS lock released. |
| Real SIGINT/resume | Actual SIGINT after third actual optimizer update: first heat child completed at2, Darcy child interrupted at1. Resume to2; completed checkpoint/history and active calibration/held-out manifest hashes unchanged. `suite-delivery-interruption.json`, script and log. Completed suite CLI resume also succeeds without retraining children. |
| MPS | Actual final-source sequential heat32² and Darcy16³,2 updates each, batch1: `suite-delivery-mps-status.json`, log and managed child artifacts. Explicit MPS, no model fallback. Both new exports reload on CPU (mps-cpu-reload.json). Earlier six-layout/MPS↔CPU/FK evidence remains historical; no fresh claim for those cases here. |
| CLI | Real reference/observe/experiment/compare/plan/status/resume commands and a fresh small heat training→automatic model discovery→full experiment: corresponding cli-*.log, demo-training-verified.log, demo-experiment-verified.log. |
| CUDA / actual Colab | Untested. No long local-pilot suite launched. |

Initial sandbox API workers and the first isolated-wheel probe failed because OpenMP could not open local shared
memory. The permitted local-process run passed; no mocked worker counted as a
numerical success. Initial API rejected accuracy:null exported from the dataclass;
nullable accuracy is now accepted consistently. Initial1D browser setup requested
8 nodes while the existing1D checkpoint required16; test now explicitly uses16.
Manual CSV browser validation initially raced the pending asynchronous parser;
the test waits for parsing before validating. An exploratory train command used
unsupported --json, then was corrected and executed; documented train commands
omit it. These development failures remain in their logs. No automatic approval
review rejection remains blocking this task.

## Actual final-source managed artifacts

Source identity: `7d1f6e98646a23e382f3f815900585c7840e41dedf3b759a22763a40f4e163de`.

- CPU suite: `suite-delivery-cpu/suites/20261026T133907-quickstart-94a64137/suite.json`.
- MPS suite: `suite-delivery-mps/suites/20261026T133910-quickstart-d7fccb73/suite.json`.
- SIGINT suite: `suite-delivery-interrupt/suites/20261026T133927-quickstart-35321992/suite.json`.
- Each child: sibling workspace `runs/<suite-id>-<preset>/`; training/checkpoint.pt,
  inference.pt, best-inference.pt, actual history/calibration/provider state and
  fixed evaluation manifest. These are smoke-trained available exports, not
  scientifically evaluated model quality.
- Fresh 8² demo: `demo/runs/` plus `demo-experiment-verified/experiment.json`.
- Browser runs/references: `suite-delivery-cpu/workbench/{jobs,data}/` (final browser
  verification), earlier development journeys in `workspace/workbench/`.
- Independent full CLI probe: `cli-heat/experiment.json`, reference/, observations/,
  reconstruction/, baseline_fourier_inverse/. `cli-comparison.json` verifies the
  separate coordinate-aware comparison command.

Earlier development suites precede final comparison-source changes and retain
those recorded source hashes. They are preserved as development evidence; strict
resume appropriately rejects a source change. Use the final-source suites above
for continuation. Old inference exports remain valid for inference.

## Negative numerical result

The independently saved `cli-heat` uses the explicit8² finite-Fourier reference,
four actual times,10 fixed spatial sensors across all times (40 measured values,
noise σ=0.03, seed54), two particles/four generation steps, existing
smoke heat export, CPU. For field u over exact unobserved grid points:

| Method | MSE | RMSE | MAE | Relative L2 |
|---|---:|---:|---:|---:|
| Learned observation+PDE reconstruction mean |0.5059387103|0.7112936878|0.5782441069|1.9992301884|
| Conventional Fourier inverse |0.0002292810|0.0151420286|0.0121837715|0.0425596364|

Metrics use cell-volume/trapezoidal-time quadrature, exact per-field/time observed
indices excluded,216 unobserved nodes, reference RMS0.3557837872. The conventional
inverse is much better. This is one bounded execution probe, not a new scientific
quality claim. No checkpoint, budget or particle was selected using test/reference
error. The earlier0/100/500 training study remains non-monotonic on test quality;
its original outputs/docs were not replaced or rerun.

## Git, packaging and limitations

No Git commit exists (main has no HEAD), and no commit was attempted. The original
90-entry staged index remains unchanged (SHA256
`9c5f4385c42492042ad13d7a58854995f74e30e47ed9ba06ae7f4b65daead9c8`);
new work is unstaged/untracked alongside existing user changes. .idea was preserved.
Both staged and unstaged whitespace checks pass. Source/wheel packaging and archive
integrity are recorded in completion.json in the evidence root. The portable source
archive excludes outputs, weights, data, .venv, node_modules, caches, private files
and scientific reference attachments; SOURCE_PACKAGE.md explains regeneration.

Only periodic uniform Cartesian families/layouts are supported. No general symbolic
PDE solver, off-grid/temporal interpolation, arbitrary boundary/mesh or unknown
parameter/fully-known-target conditional checkpoint is claimed. Refinement remains
subject to its existing compatible-grid route; comparing a restricted reference
does not establish resolution generalization. Numerical references are approximations;
empirical ensemble spread is uncalibrated. Large-suite duration, model quality and
CUDA/Colab execution remain unestablished. Next complete implementation slice:
validation-designed repeated-seed scientific evaluation across trained specialists,
reference resolution/tolerance studies and conventional baselines, preserving an
untouched test set and separating training budget from inverse execution cost.
