# Coordinate/source workflow verification — 27 October2026

Evidence: outputs/coordinate-20261027/. This is new execution, separate from
previous catalogue/FK/learning-budget evidence. Accepted earlier results and the
original Git index are preserved. No long training, full expanded suite, source
archive regeneration, CUDA or actual Colab execution occurred.

## Actual checks

| Check | Actual result / log |
|---|---|
| Full Python regression | 235 passed,148.60s: python-final.log. This preceded final explicit checkpoint registration, source guard and timing metadata refinements. |
| Final affected regression |56 checks of coordinate stages/portable import/source validation, default symbolic behavior and storage. Latest result is in python-delivered-final.log (5.05s); earlier successful56 runs are python-complete.log, python-final-affected.log and python-verified.log. Do not add repeated tests into a fictitious larger suite. |
| Frontend |42 passed: frontend-final.log, including stale-source blocking, independent query/measurement controls and no training on load. |
| TypeScript / production build |build-final.log: pass. Existing large MathLive bundle warning remains. |
| Lint |lint-final.log:0 errors,3 existing component-export warnings. |
| Actual browser |browser-verified.log:3 passed,37.1s; reference→measurements→latent inverse→comparison/reload/Runs reopening/stage overview; numerical full-field import→invalid counts→measurements-only measured zeros/scattered sensors; preserved fixed-grid32² heat inverse/comparison. Final explicit-device/history/responsive repeat: browser-final-device.log,1 passed,7.5s. Earlier browser logs retain failures/fixes. |
| Input/CLI artifacts |reference-cli,observe-cli,reference-export-cli,reference-import-cli,query-grid-cli,query-scattered-cli,checkpoint-import-cli and checkpoint-discovery-cli logs. Actual12²/T4 output(2,576), scattered output(2,3), including duplicated queries in preserved order with identical values. |
| CPU stages |ae-start/ae-resume and flow-start/flow-resume:4 actual updates each on8²/T4. Both stages paused at2 then restored optimizer/RNG/data/latent dependencies. |
| True spatial dimensions |Current integration checks actually train/reload1D and3D small stages, query5 and5³ grids and evaluate physical derivatives; the direct set/chunk/permutation tests cover1D/2D/3D. |
| MPS |mps-probe.json: AE2/flow2 actual8² updates,6²/T3 query, two Heun steps/two particles, scattered sensors, physical derivatives; mps-cpu-portability.json: standalone flow export remapped to CPU and finite query inference. |
| AE bottleneck |ae-validation.json: two held-out physical realizations, physical reconstruction/initial errors, common-grid residual distortion and spectral-power error. Full reference context is deliberately used for this AE evaluation only. |
| Comparison sensitivity |comparison-sensitivity/comparison.json and report.md:1 fixed validation realization, sensor counts2/5, absolute sigma0/.03, seeds91/92, two samples, two Heun steps,8²/12²/32² queries:24 retained records. This is a bounded workflow probe, not a training study. |
| Final comparison metadata |comparison-metadata/: single one-step case verifies declared AE/flow budgets, data configuration, actual recorded baseline training seconds and explicit null timings for older coordinate exports. |
| Storage |storage.json inventories actual stage files/dependencies; cleanup-preview.json contains0 eligible entries/0 bytes/applied=false. No real cleanup apply. |

## What is trained and what it establishes

The preserved user heat32 CNN has3000 updates and34,372 parameters. The new CPU
smoke has4 AE and4 flow updates,8²/T4,8,233 AE parameters plus9,744 velocity
parameters (17,977 total). The MPS probe has2/2 updates. These are completed **smoke
budgets**, not scientifically evaluated coordinate models or completed full10,000
update targets. Full MPS32² resource fit and model quality remain untested.

The CPU smoke AE's two held-out RMSEs are0.917771 and0.738686; relative spectral
power errors0.969511 and0.924995. Its displayed reconstruction is visibly close to
a constant and misses the reference structure. This is evidence that the current
bottleneck is inadequate, not an accurate learned solution manifold.

The bounded comparison preserves every sample's RMSE/MAE/relativeL2 and separate
ensemble-mean/spread, actual noisy-sensor consistency and post-run clean-sensor
errors. Both models use identical underlying cases, known parameters and noisy
measurements. The coordinate model uses learned observation conditioning; the
fixed model uses observation+PDE endpoint guidance. Their training data/compute
and architecture differ. No winner is selected from these smoke results. Querying
finer points and obtaining a lower single-case error than noisy generated baseline
samples would not establish architectural superiority.

Common-reference transfer away from native32² nodes is periodic multilinear space/
linear physical-time interpolation, explicitly approximate. Fixed output at other
resolutions is labelled baseline plus interpolation. Error norms are discrete
norms over the stated common query points, not an independent continuum certificate.
Initial/per-time error uses truth only after generation. Boundary equality is
imposed by periodic features/interpolation. Common-grid physics errors include FD
error; direct coordinate autodiff probes sample0 at a recorded evenly indexed
subset. Multiple-sample spread is uncalibrated; no empirical coverage is invented.

Original negative Fourier-inverse and non-monotonic training-budget findings remain
unchanged in their earlier artifact roots. This task neither repeats that study
nor chooses a budget by test error. The new comparison is a workflow/architecture
execution check, not new evidence overturning those findings.

## Checkpoints, source stages and timing

Every saved checkpoint/report identifies its source stage. Full235-test execution
preceded the final registration/source-guard/timing refinements; affected tests were
rerun. CLI CPU4/4 and MPS2/2 exports retain their original source hashes and remain
inference-portable. Strict optimizer resume still requires the source actually
recorded in that run; no mismatch was bypassed. New user-launched runs use the final
source and maintain their own exact resume identity.

New checkpoints accumulate training host seconds across resume and separately
record setup/reference preparation costs and solve counts. Setup/data costs overlap
and must not be summed blindly. Timing includes validation up to checkpoint
construction, excludes final checkpoint serialization, and is host wall time with
per-loss accelerator synchronization. Earlier smoke exports predate these added
fields: comparison-metadata reports null rather than inventing their cumulative
training time. The preserved fixed checkpoint actually records86.13938 training
seconds; differing timing scopes still prevent a matched-compute claim.

Memory reports process-lifetime RSS high-water mark, CUDA allocated peak only when
reset/measured, and MPS allocation after execution, never an MPS peak. The current
comparison is not a clean isolated-memory architecture benchmark.

## Failures retained and corrected

- Initial integration assertions assumed400 instead of the repository's422 and
  compared a tuple directly with its JSON list. Corrected without weakening numeric
  checks.
- Frontend test fixtures initially omitted explicit null reference/error arrays;
  corrected to the actual response contract. A later test retained previous DOM;
  afterEach(cleanup) now isolates renders/timers.
- Retained baseline browser test initially attempted a hidden settings input, then
  used an overly exact text selector for a paragraph containing storage details.
  Corrected selectors; real numerical inference/comparison subsequently passed.
- Current-source validation initially hashed0.0 differently from JavaScript0.
  Actual rejected browser request is saved; structural numeric equality now
  preserves physics while accepting equivalent JSON numbers. A regression covers
  this exact serialization difference. Unsupported/changed equations still reject.
- Invalid off-domain observations are distinguished from the duplicated periodic
  endpoint. Queries can repeat/order arbitrarily; observation duplicates cannot.
- Relative sigma is preserved through the shared ObservationSet, weighted Fourier
  inverse, diagnostics/plots and spatial remapping; no silent scalar replacement.
- Fresh stage outputs reject existing files; failed or accidental non-resume requests preserve older interrupted manifests. Real fixture regressions cover both.
- Stage summary commands initially widened the desktop page; wrapping/bounded
  provenance plus an actual browser scroll-width assertion corrected this.

Rendered comparison and stage screenshots were visually inspected. Files:
browser/observations.png, comparison.png, measurements-only.png, fixed-baseline.png,
training-stages.png. No mocked browser result counts as numerical execution.

## Final preservation and limits

completion.json records final source/Git/index/whitespace checks and task-service
shutdown. Main has no HEAD; no commit/staging/reset was attempted. The existing
90-entry staged index remains unchanged. Default fixed-grid train-all still has11
targets. The optional stages are explicitly selected; no long jobs start from UI.

No scientifically validated coordinate checkpoint, multiresolution-reference
training, irregular geometry, new latent PDE family, latent FK/SDE/guidance,
calibrated uncertainty, CUDA or actual Colab execution is claimed. Next experiment:
train/evaluate the supervised AE with fixed validation spectral/physical distortion,
then freeze an adequate bottleneck and run a deliberately matched held-out
observation/noise/seed comparison before any untouched test evaluation.

## Final handoff recovery — 27 October 2026

The interrupted handoff was recovered without changing package/frontend code or
repeating training, tests or browser execution. The final device-selector journey
had already passed in browser-final-device.log; completion.json already recorded
preservation and clean task-service shutdown. Fresh checks are recorded separately
in outputs/coordinate-20261027/recovery-check.json: package source and the original
90-entry Git index match their saved fingerprints; the existing fixed-grid export,
CPU AE/flow exports and MPS flow export match saved checkpoint hashes. Both staged
and unstaged whitespace checks pass. Ports 8000/5173 are free and read-only process
inspection found no matching PDEFlow training/server, Vite or Playwright process.
No signals were needed, no commit was created, and no scientific output was
modified. Recorded execution above remains evidence from the preceding session.
