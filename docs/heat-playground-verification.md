# Heat playground verification — 27 October 2026

This record describes execution during this redesign. Earlier training studies,
MPS training and scientific comparisons remain historical evidence; they were
not repeated to make the new interface appear accurate.

Evidence root: `outputs/heat-playground-20261027/`.

| Check | Actual result | Saved evidence |
|---|---|---|
| Focused Python numeric/API/isolation/deletion/re-import regressions | 12 passed | `python-final.log` |
| Existing affected workbench queue and coordinate contracts, with earlier 8 focused cases | 18 passed (8 overlap; 10 additional existing tests) | `python-affected.log` |
| Frontend brush/scale and training component tests | 7 passed | `frontend-final.log` |
| TypeScript + production build | Passed | `build-final.log` |
| ESLint | 0 errors; 3 existing UI component refresh warnings | `lint-final.log` |
| Real rendered browser acceptance cases | 5 distinct cases passed across the focused runs below | browser logs, screenshots, copied numerical job artifacts |
| Tracked whitespace check | Passed | `git diff --check`; source preservation record |
| Existing checkpoints preserved | All 312 original `.pt` files present, content hashes unchanged | `preservation-before.json`, `preservation-after.json` |
| Read-only storage audit | 131 model groups / 312 recognized standard and Gaussian verification weight files | `storage-readonly-inventory.json` |
| Git preservation | Original staging index unchanged; repository has no HEAD commit | `preservation-after.json` |

Python commands actually executed (from repository root):

```sh
.venv/bin/python -m pytest tests/integration/test_heat_playground.py -q
.venv/bin/python -m pytest tests/integration/test_heat_playground.py tests/integration/test_workbench.py tests/unit/test_coordinate.py -q
```

The affected workbench test fixtures generate tiny one-update CPU models in
temporary directories. No new training workflow, development budget, study or
substantial training was launched. The real browser uses a copied preserved
AE4/flow4 coordinate export, not a newly trained model or reference prediction.

Frontend commands actually executed (inside `frontend`, with the supported
Homebrew Node 22.23.3 on PATH):

```sh
npm test -- src/heat.test.ts src/TrainModel.test.tsx
npm run build
npm run lint
npx playwright test e2e/heat-playground.spec.ts
npx playwright test e2e/heat-playground.spec.ts --grep 'existing coordinate|measurements-only|disposable storage'
npx playwright test e2e/heat-playground.spec.ts e2e/heat-responsive.spec.ts --grep 'disposable storage|responsive layout'
```

Browser services used an **isolated** workspace,
`/private/tmp/pdeflow-heat-playground-20261027`, served with:

```sh
.venv/bin/pdeflow serve --workspace /private/tmp/pdeflow-heat-playground-20261027
```

Vite used `npm run dev` at 127.0.0.1:5173. Only these task-owned services were
stopped afterward; the temporary inspection tab was closed. Ports/process
confirmation is in `process-handoff.json`. Existing user tabs were preserved.
Destructive browser tests explicitly assert that exact isolated managed root;
**do not run them against the normal project output workspace**. They import a
copy of the preserved smoke checkpoint, preview/cancel/delete only disposable
artifacts, and retain a protected personal fixture.

## Actual browser coverage and failures

The five distinct cases cover all requested journeys:

1. Fresh no-model opening, visible initial field, play/pause/restart and D;
   amplitude/wave/exact/numerical/difference controls; wrapped pointer painting,
   clear/undo; seeded random reproduction; invalid/valid formula; delayed real
   responses rejected after editing; invalid count; exactly 25 observations with
   actual sigma .03; resample seed/location change; scrub preserves observation
   time; separate training page defaults. Final success for this case is in
   `browser-verified.log` (that run subsequently exposed the import bug below).
2. Actual compatible checkpoint upload, explicit barely-trained consent, 25-point
   CPU inverse, reference-free submitted payload, post-inference reference
   attachment, aligned plots and real metrics; unavailable CUDA error and
   model D-range error. Final success: `browser-reimport-verified.log`.
3. Imported measurements-only inverse with no reference panel or field error;
   a real long-step inference locks artifacts, storage deletion is blocked,
   cancellation works. Final success: `browser-reimport-verified.log`.
4. Disposable selected/all preview, cancel leaves files, scope confirmation,
   permanent unlink, protected file retained, saved selection removed, UI model
   list refreshed, reference still plays with zero model files. Final success:
   `browser-storage-responsive.log`.
5. 390×844 mobile layout without horizontal overflow, keyboard Enter starts
   playback, hotspot and reset. Final success: `browser-storage-responsive.log`.

Initial browser attempts caught three test locator/format assumptions (a leading
zero in a numeric input, metric text inside a value container, multiple active
copies). Those assertions were corrected without relaxing scientific checks.
A subsequent run caught a **real implementation bug**: after deleting an imported
managed copy, re-importing identical content left its registry manifest pointing
at the missing old path. Explicit re-import now repairs that stale registration
only when its former export is absent and the recorded hash agrees; live links
are preserved. A Python regression and the real model browser journey passed
after this repair. Earlier failure logs are retained, not hidden.

Browser tests execute the real numerical API and real coordinate worker. The
stale-response test delays actual fetched responses; it does not supply fabricated
solutions. Frontend component tests are unit tests, not numerical integration.

Screenshots (actual rendered application, not mockups):

- `main-playground.png`: fresh default without any trained model.
- `model-comparison.png`: actual AE4/flow4 reconstruction from 25 observations,
  showing its poor, nearly constant prediction on a common reference/model scale.
- `mobile-playground.png`: 390-pixel layout.

Nine actual reconstruction job records, including cancelled probes, are copied
to `browser-jobs/`. Successful `coordinate-result/` directories retain
`result.npz`, `metrics.json`, `comparison.json` when applicable, and the scalar
observations. `request.json` shows absence of full reference/initial inputs.
Three attached immutable references are preserved in `browser-references/`.
Their original temporary reference IDs remain in the comparison manifests;
the copied directory with that ID contains the same reference array/checksums.

## Numerical and safety checks

CPU float64 FFT versus independently evaluated product cosine agrees within
approximately 5e-16 for the default example and below 3e-14 for tested shifted,
non-unit domains, nonzero phases, zero wave numbers and constant modes. Mean is
preserved. Scattered Fourier values agree with the cosine expression to 8e-16;
noise uses a separate deterministic stream and clean values are stored separately.
Random references reproduce by seed. Unsafe, nonfinite and unknown expressions
are rejected. API compatibility rejects out-of-range D independently, and its
strict inverse schema rejects hidden initial input; the captured queue request
contains no full reference or clean values.

Storage tests use disposable fixtures: selected/all scopes, resume and stage
weights, shared autoencoder retention, active trainer and queued refinement
dependency rejection, stale fingerprints, protected unknown/source files,
hardlink accounting, unsafe path/lock symlinks, optional producer-owned data/logs,
partial unlink failure reporting, selection clearing and reference after zero
models. Directory-descriptor deletion does not follow replaced parent symlinks.
No original scientific artifact was removed; all 312 hashes match the before
record. The Git staging hash also matches. Changes are left unstaged/uncommitted
alongside prior work; no commit, reset, permission workaround or training campaign.

## Backend and scientific status

| Backend | Status for this redesign |
|---|---|
| CPU | Actual FFT, sampling, save/load, inverse, comparison, queue/cancel and browser integration |
| MPS | Supported by retained runtime; earlier coordinate AE2/flow2 and prior inference evidence retained, not rerun here |
| CUDA | Unavailable on this Mac; explicit rejection tested; no execution verified |
| Colab | Workflow retained; actual Colab execution not verified |

The screenshot is a smoke-trained AE4/flow4 checkpoint. Execution, compatible
variable coordinate queries and scientific accuracy are different claims. Only
the first two are demonstrated here. The coordinate autoencoder/reconstruction
quality remains poor. Earlier conventional Fourier-inverse superiority and
non-monotonic test performance across budgets are unchanged; no winning training
budget is selected using test performance. Recommended next work is a bounded,
validation-only representation learning experiment, followed by flow only if the
representation passes its existing baseline check; inspect held-out spatial and
spectral distortion before spending a larger training budget.
