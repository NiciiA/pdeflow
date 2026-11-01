# Operator-aware heat/wave v2 verification

This record distinguishes saved execution from the final documentation/packaging
continuation. All paths below are relative to `outputs/operator-wave-20261030/`.
No substantial training was launched. No CUDA, Colab GPU or Drive execution is claimed.
The supplied extension PDF was read in full; it is design guidance, not performance
validation. Architecture, formulas, tensor contracts and capacity counts are in
`operator-wave-model.md`; exact app/training commands are in the current README.

## Saved implementation execution

| Check | Actual evidence | Scope and limits |
|---|---|---|
| Focused Python | `python-final.txt`:46 passed,2 CUDA skips | Mathematical/graph/gradient/API, all four field losses, tiny A1/A2/B, resume/freeze/export, actual MPS representation training |
| Final state/graph mathematics | `operator-math-final.txt`:20 passed | Includes non-square reference modes, real Nyquist interpolation, zero mode, first-order derivatives, graph relabelling/influence, non-unit model periodicity, velocity-only gradients; overlaps the focused set |
| Final retained checkpoint | `final-artifact-check.txt`:11 passed | Fresh two-update A1/A2/B, interrupted/resumed stages, frozen targets, known/unknown D/c, standalone export, API isolation, optional gradient correction; overlaps focused tests |
| Workflow/numerical regressions | `workflow-numerical.txt`:54 passed,5 failed; `workflow-repair.txt`:22 passed | Five obsolete heat/coverage/grid fixture assumptions were corrected; all five failed checks passed in the focused recheck, alongside17 repeated relevant checks. Includes existing true3D numerical coverage. No tolerance loosening |
| Frontend | `frontend-final.txt`:61 passed in13 files | Display/control regressions; not numerical browser integration |
| TypeScript/build | `build-last.txt`:production build passed | Final quantity labels and controls compiled |
| Lint | `lint-final.txt`:0 errors,3 existing Fast Refresh warnings | Warnings in shared badge/button/tabs components |
| Real Chromium | `browser-release.txt`:2 passed | Four actual numerical reconstructions: heat/wave × known/unknown scalar through ONE final checkpoint; six wave presets, independent velocity drawing, all four measurement placements. No mocked numerical responses |
| Actual local notebook cells | `notebook-final.txt`:2 passed; `notebook-final/notebook-run-all.log` | Actual six cells: source setup, disposable3-update A1/A2/B profile,2-update training, gate/persistence/export on local CPU/IPython. Not actual Colab/CUDA/Drive |
| CPU/MPS comparison | `cpu-mps-inference.json`, `mps-final.txt` | Final checkpoint, identical explicit initial flow noise in all four modes; maximum output difference2.384185791015625e-7. Physical higher derivatives execute on MPS. No cross-device bitwise guarantee |
| CLI | `cli-plan.json`, `cli-demo.txt`, `demo/` | Resolved managed plan and bounded known/unknown heat/wave demo. Older demo weights are separately retained; final checkpoint is specified below |
| Reference distribution | `wave-training-preview.png`, `wave-preview.json` | Training-only signed u/v cases, c range and spectral energy; no test-based tuning.32 phase strata vs maximum3.394 cycles per unit interval for max component mode4 |

A saved collection snapshot contains331 tests (`collection-final.txt`); no full-suite execution is claimed.
Historical tests for retired architectures are not evidence for this model. Superseded
physical-contract test sources are preserved in predecessor test archives. Initial
failure logs remain saved. The repaired API list/tuple graph comparison and unchanged-
preset pending-state bug were rechecked. Browser screenshots were visually inspected;
velocity profiles show integrals of v, not mislabeled u or error histograms.

Screenshots: `browser-heat-known.png`, `browser-heat-unknown.png`,
`browser-wave-known.png`, `browser-wave-unknown.png`,
`browser-wave-velocity-drawing.png`. The unknown-wave screenshot shows velocity,
region contribution map/overlay, scalar metric selection and poor smoke predictions.

## Checkpoints and scientific status

The retained final files are:

- `final-cpu/autoencoder/autoencoder.pt`: A1,2 updates.
- `final-cpu/evolution/evolution.pt`: A2,2 updates; hashed A1 dependency.
- `final-cpu/flow/coordinate.pt`: B,2 updates; standalone portable inference export.
- Matching `resume.pt`, configs, dataset recipe, histories and validation reports in
  the same stage directories. Optimizer/scheduler/scaler/RNG live in resume files.
- `notebook-final/notebook-run/export/coordinate.pt`: separate actual local notebook
  A1/A2/B2-update export. Temporary preflight used3 updates per stage and was disposable.
- `first-smoke/`, `shared-cpu/`, `verified-cpu/`, earlier notebook outputs: preserved
  earlier implementation smoke artifacts. They do not all have the final source hash;
  strict resume requires their recorded source, never silent migration.

No new checkpoint is scientifically evaluated as an accurate model. The final A1
normalized whole-field validation/constant-baseline ratio is **4.25209493** on the
bounded two-realization report: worse than the baseline. Low-level implementation
smoke execution exercised later stages; it does not represent a passed quality gate.
The final continuation did not override the failed gate or launch further training.
Normal managed/Colab training must stop for inspection on a failed representation gate.

Final bounded conditional validation (`final-cpu/flow-validation.json`), one sample,
2 Heun steps,8² grid, explicitly recorded physical times; these are unweighted node/
time-list RMSE summaries from `error_summary`, distinct from trapezoidal uniform-time
metrics also saved in the same per-quantity report:

| Family / parameter status | u RMSE | v RMSE | Scalar absolute error |
|---|---:|---:|---:|
| Heat known D |0.91744|absent|0 supplied authoritative value|
| Heat unknown D |0.91833|absent|0.00451187|
| Wave known c |2.54089|1.29310|0 supplied authoritative value|
| Wave unknown c |2.53844|1.29395|0.16433446|

These small reports neither estimate population accuracy nor establish identification.
The conventional Fourier inverse remains better in the historical recorded studies;
test quality there was non-monotonic across budgets. No new matched comparison or
winning budget was chosen using test performance. Region-balanced α=.5 remains an
experimental starting point and may sacrifice whole-field/background accuracy.

## Final continuation: packaging, compatibility and preservation

This continuation inspected the working tree and the evidence above and reused the
completed training, browser and test runs. It performs source/archive integrity,
strict checkpoint/source/dependency loading, documented-path checks, whitespace and
service shutdown checks only. Their exact results are written to:

- `compatibility-final.json`: current scientific-source fingerprint, checkpoint
  identities, stage/dependency hashes, dataset identity, portable standalone CPU load.
- `preservation-after.json`: all226 pre-task PT hashes compared with
  `preservation-before.json`; no changed/missing checkpoint. A timestamp audit checks
  older output trees separately. No old dataset/result/checkpoint is moved or deleted.
- `package-verification.json`: ZIP integrity, safe paths, expected files, every manifest
  byte count/hash and excluded output/dependency/weight trees.
- `pdeflow-operator-wave-source.zip`, its `.sha256`, and `source-manifest.json`.
- `final-checks.json`: source/whitespace/Git state, documented files and task-owned
  service shutdown. Existing work remains unstaged; no commit is created by this task.

Checkpoint hashes provide byte-for-byte preservation evidence for the226 recorded PT
files. Other old datasets/results lack pre-task content hashes; their unchanged
modification-time audit is supporting evidence, not a substitute for such hashes.
No file outside this task's output directory was newly modified in the older output
scan. New evidence remains under the task directory. Existing reference PDFs and
personal/unrelated files are excluded from packaging and untouched.

Unsupported: static/weak/shock/forced branches, nonperiodic or new geometries, neural
1D/3D in this specialist, accurate/calibrated inference, proven resolution generalization,
exact learned conservation and universal PDE understanding. Large/XLarge parameter
counts are exact; GPU memory fit and speed are unmeasured. See model documentation
for one-realization accumulation and host-inclusive timing limitations.
