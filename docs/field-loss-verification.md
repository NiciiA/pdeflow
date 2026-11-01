# Field loss verification — 29 October2026

This session implements real gradient-updating representation objectives, not only
new metrics. No substantial training or controlled scientific comparison was run.
All earlier scientific outputs remain in place. The new tiny models are pipeline
checks with poor accuracy, not evidence that region balancing improves results.

Evidence root: `outputs/field-losses-20261029/`.

| Check | Actual execution / evidence |
|---|---|
| Loss mathematics, estimator and API | `math-api-final.log`:23 passes, including mathematical quadrature/weights/gradients, per-example/time reductions, periodic edges/corners, noise floors, normalization, explicit inadequate region-budget rejection, nonuniform WMSE and region estimators, exact training-preview calibration, scalar/contribution-map consistency. `math-final.log`21 math-only passes overlaps this set. |
| All four representation objectives | `training-tests.log`:18 passes in the initial focused matrix, including four real 2-update CPU AEs, exact same-objective resume and incompatible-change rejection, shared Colab Session AE/flow/export, actual MPS training. Subsequent `smokes-saved.log`:5 passes rerun only four tiny AEs plus MPS to retain artifacts under `training-smokes/` after pytest temporary cleanup. |
| α changes learning | `gradient-effects-final.log`:actual α0/α1 training changed both encoder and decoder optimizer updates with the same dataset identity;1 pass,1 actual-CUDA skip. Equal small/large total gradient contributions and area-resolution/smoothing tests also pass in the math set. |
| Existing behaviour | `regressions.log`:10 passes covering original1D–3D coordinate contracts, ordinary-MSE optimizer/accumulation/resume/extensions, continuous reference targets/export/order, integrated fixed-grid and representation workflows. `evaluation-regression.log`:1 overlapping continuous/evaluation pass after shared metrics were added to saved reference comparisons. |
| Notebook | `notebook-final.log`:2 passes, including actual sequential execution of all six modified cells in an isolated local CPU/IPython environment, source setup, disposable3-update profiling, AE2/flow2, persistence/gate/export. Full logs and exported models under `notebook-final/`. The earlier successful notebook run is retained under `notebook/`. |
| Frontend | `frontend-final.log`:19 focused passes. `build-final.log`:TypeScript/Vite pass with the existing large legacy chunk warning. `lint-final.log`:zero errors, three existing fast-refresh warnings. |
| Actual browser training | `browser-first.log`:new default selected α.5, actual training-distribution region preview, explicit2-update encoder/flow training via UI, frozen loss retained in run. `browser-training.json` retains the run/preview record and `training-loss-preview.png` its rendering. |
| Actual browser comparison | `browser-comparison-final.log`:real imported notebook checkpoint selected by its SHA256 model ID, actual CPU reconstruction, all six metric choices, labelled contribution/error maps, region overlay, preserved observations, one inference and one metric computation independent of display selections. `browser-metrics.json` stores numerical output and counts. Measurements-only mode omits invented whole-field metrics. |
| Screenshots | `training-loss-preview.png`, `region-overlay.png`, `metric-selection.png`; profiles remain spatial integrals. |
| Controlled comparison | `comparison-plan-final/` contains four matching200-update representation plans, no executed study. The script requires explicit `--run` and a fresh root to train later. |

Do not sum overlapping counts as a full repository suite. Commands were scoped to
affected numerical/training/API/frontend/browser paths. Successful suites were
reused; only affected checks or lost temporary smoke artifacts were repeated.

Primary commands (from the repository root unless stated):

```bash
.venv/bin/python -m pytest tests/unit/test_field_losses.py tests/integration/test_field_loss_api.py -q
.venv/bin/python -m pytest tests/integration/test_field_loss_training.py -q
PDEFLOW_NOTEBOOK_PYTHON=/private/tmp/pdeflow-colab-notebook-venv/bin/python .venv/bin/python -m pytest tests/integration/test_colab_notebook.py -q
# frontend/ with supported Node22.23.3 on PATH:
npm test -- src/FieldLossControls.test.tsx src/TrainModel.test.tsx src/HeatPlayground.test.tsx src/heatProfiles.test.ts src/heat.test.ts
npm run build
npm run lint
npx playwright test e2e/field-losses.spec.ts --workers=1
```

The temporary interpreter path above records the actual local verification setup,
not a runtime requirement or a hard-coded application/Colab path. The notebook
uses the user's configured source and storage roots. CUDA and actual hosted Colab/
Drive execution remain unverified. BF16-input loss reductions on CPU and real MPS
FP32 training establish only their stated coverage, not actual CUDA AMP execution.

Failures and repairs retained honestly:

- The first mathematical attempt imported SciPy, which is not in the tested core
 environment. Replaced it with bounded NumPy periodic components, Fourier
 interpolation and physical-scale Gaussian filtering; no mandatory dependency added.
- Pure zero-time Fourier resampling first passed D0 to a solver whose point
 evaluator requires positive D. Resampling now uses D1 at t0; no evolution occurs.
- An initial browser result computed metrics for a cached frame then the exact-time
 decode. Metrics now wait for the final saved-latent decode. Scalar display changes
 make no additional inference or metric requests.
- Chromium's inspector evicted the upload response body while other requests ran.
 The test selects the imported model by its independently verified checkpoint
 content hash and asserts the actual reconstruction request uses that ID.
- A native NumPy/BLAS segmentation fault occurred during concurrent numerical
 verification at the existing Fourier coordinate matrix product. The isolated
 rerun passed, including the decisive optimizer test. Cause is not established;
 the log is retained in `gradient-effects.log`. No numerical tolerance was loosened
 and no claim that a native library issue was fixed is made.
- React lint caught a synchronous state update effect for contribution scales.
 The update now occurs in the asynchronous metric-response callback; final lint
 has zero errors.

Existing49 checkpoint hashes and the 90 logical staged Git entries are checked in
`preservation-before.json` / `preservation-after.json`. No existing weights were
retrained or replaced, and no staging/reset/commit/push was performed. Copies of
actual browser jobs/training manifests are preserved under `browser-artifacts/`.
The isolated services use `/private/tmp/pdeflow-losses-20261029`; only those task-owned
services are stopped after checks, with `process-check.txt` recording final ports.

Limitations and mathematics: [field-reconstruction-losses.md](field-reconstruction-losses.md).
The next bounded scientific task is the controlled, validation-only representation
comparison with repeated seeds, before substantial latent-flow training. Region
balancing remains experimental and may trade whole-field/background accuracy for
retained contrast-region accuracy; no winning loss or α has been established.
