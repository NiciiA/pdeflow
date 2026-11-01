# Continuous heat verification — 28 October2026

All execution here was bounded. No substantial training, CUDA or actual Google
Colab/Drive run occurred. Historical scientific outputs/checkpoints were preserved;
older verification records remain historical, not rerun evidence.

Evidence root: `outputs/continuous-time-20261028/` (not bundled in source archive).
Current instructions, formulas, policy definitions and migration are in
[continuous-heat.md](continuous-heat.md).

| Check executed in this session | Actual result | Evidence |
|---|---|---|
| Analytic arbitrary-time heat + placement + affected old/new Python suites | 55 passed, 3 skipped | `python-final.log`; included new11 checks, existing1D/3D coordinate and local/Colab/playground regressions |
| Follow-up quality-gate change | 33 passed, 1 CUDA skip | `gate-final.log`; local/Colab orchestration and new continuous tests rerun after matching gate measures |
| Follow-up host calibration change | 21 passed, 1 CUDA skip | `engine-final.log`; exact CPU resume/accumulation, continuous inference/export and recovery rerun after CPU-FP64 correction |
| Final decoder/constant-mode and scattered-time checks | 11 passed; selected API case1 passed | `decode-final.log`, `scattered-final.log`; tiny constants preserved and blank time selection means continuous times |
| Actual six notebook cells in order, setup rerun/cache guard | 2 passed | `notebook-final.log`, `notebook-run-all.log`; explicit CPU pipeline, local persistent filesystem, real install/profile/training/validation/export |
| Actual MPS | AE2/flow2, validation/export and CPU reload passed | `mps.log`, `mps-decode.log`, `mps-pipeline/`; model computation on MPS, reference/calibration explicitly CPU |
| Frontend unit tests | 51 passed in11 files, no unhandled errors | `frontend-tests.log`; includes initial-null rendering and coordinate mapping |
| TypeScript / production build / lint | Passed; existing bundle warning and3 fast-refresh warnings | `frontend-build.log`, `frontend-lint.log` |
| Real browser numerical workflows | 2 passed | `browser-final.log`; screenshots and `browser-placement.json` |
| Matched-policy CLI | 36 real inference records | `placement-comparison.json/.npz`, `comparison-cli.log`; six validation families × three policies × two seeds |
| Native64² / finer128² reference gallery | Six actual periodic families | `reference-gallery.png/.json`; common-node RMSE3.69e-17–1.13e-16; smooth shared-spectral references, not independent solver or arbitrary-drawing accuracy |
| Large/XL capacity retained | Config/meta counts only | `large-summary.json`, `extra-large-summary.json`; no large optimization or GPU-fit claim |
| Archive / whitespace / preservation / processes | Final records | `archive-verification.json`, `whitespace.log`, `preservation-before.json`, `preservation-after.json`, `process-check.txt` |

The repeated counts above overlap; they are **not summed as distinct tests**.
The final targeted rechecks cover the changes made after the full55-pass matrix.
The last tiny-constant conservation guard was checked on CPU; it does not change
the already-tested device computation path. The current notebook export matches
the final scientific source fingerprint (`source-match.json`).
The other two full-matrix skips depended on historical smoke checkpoint files
that were already unavailable. New continuous numerical import/inference was
executed independently, not skipped or replaced by mocks.

## What the checks establish

Analytic tests include constants, signed single/multiple modes, physical domain
lengths/origins, arbitrary times, t=0/t=1, endpoint tolerance before float32
rounding and materially invalid times. Sampling tests show many non-frame times,
fresh query points, exact saved-RNG CPU continuation and fixed per-solution
reference contexts. Changing display frames does not alter data/model identities;
changing interval or sampling does. Coordinates retain the physical time chain
factor. Query reordering/chunking reproduces the same generated samples. Legacy
snapshot inference remains accepted within its original contract, while the API
independently rejects `require_continuous=true` with a legacy model.

Placement tests cover N1/2/7/25/101/2000, rectangular domains/origins, equal-area
centroid, endpoint uniqueness, seeds, uniform coverage, signed/separated/edge
features, background counts, regional allocation, kernel bounds, negligible-
contrast fallback and bounded separation relaxation. Candidate and committed
positions agree; noise changes do not change positions. Fixed/adaptive timing
and per-time provenance are explicit. JSON records preserve actual times,
absolute sigma and placement parameters; clean values remain separate on disk.

Real browser journeys used the running local API and actual trained weights:
continuous simulation over[0,1], exact0.237 scrub,25 random/even observations,
multiple-source focusing/spread, manual add/remove with correct coordinates,
committed-observation preservation under display-frame changes, arbitrary-time
reconstruction from a genuine continuous checkpoint, repeated decoding of the
same latent, and the old-checkpoint warning. Delayed **actual** numerical responses
were used to test stale measurement/time and field guards; no solver response
was fabricated. Reconstruction requests contained only observations and public
physics, no full field, region centres, clean targets or reference payload.

Screenshots (actual rendered browser output):

- [Heat-focused preview/observations](../outputs/continuous-time-20261028/heat-focused.png)
- [Manual sensor points](../outputs/continuous-time-20261028/manual-sensors.png)
- [Continuous reconstruction at0.237](../outputs/continuous-time-20261028/continuous-reconstruction.png)

The six-cell notebook test runs in isolated Python/IPython with supported real
core dependencies, with only form/path settings changed to small CPU mode,
filesystem persistence, explicit gate-override permission and no browser download.
It exercises source extraction/install, actual profiling, both learning stages,
validation/gate and export without hidden manual cell state. It also reruns setup
and refuses a modified disposable source cache. Actual Colab GUI/upload/auth/Drive
semantics and GPU kernels remain unverified.

## Failures repaired

The first browser attempt exposed a null initial state whose optional-reference
comparison allowed a null dereference. It was fixed and a loading regression
added. Test setup then needed the missing JSDOM ResizeObserver platform stub and
explicit cleanup; final frontend tests have no unhandled errors. The first
non-square lattice had unequal row areas and a small centroid bias; row heights
now match row counts, giving equal-area cells. An API-error test was corrected
to expect the existing422 validation convention, not400. Numerical tolerances
were not loosened.

Final review found a real metric mismatch: the gate averaged errors over an
irregular evaluation-time list while the constant baseline used uniform
quadrature. The gate now uses the report's uniform numerator and denominator;
a regression checks that they agree even when the irregular average differs.
This is a corrected measurement, not an improvement to learned weights.
Train-only latent moment accumulation was moved explicitly to CPUfloat64 because
MPS cannot allocate float64 tensors. Actual MPS optimization remains on MPS;
this is not a concealed model-device fallback. The actual MPS and targeted CPU
checks passed afterward.

A browser test helper initially moved the sampling time unintentionally; it was
corrected to preserve and assert the requested0.237. Failed logs remain for
transparency (`browser-first.log`, `browser-second.log`, `affected-first.log`,
`continuous-first.log`). Current success records are the final logs listed above.

## Scientific status and outputs

`notebook-run/export/coordinate.pt` is a portable **AE2/flow2 pipeline check**, not
a scientifically evaluated model. Its corrected six-case uniform representation
MSE is0.1089510841 versus constant baseline0.1189131892, ratio0.9162237157.
Initial/early/late and spectral errors remain in the report; a gate pass alone
is insufficient. The MPS two-update run gives a similar ratio, but no bitwise
cross-device equality or model-quality conclusion is claimed. Earlier initial
CPU execution remains at `local/pipeline/`; current-source CPU notebook and MPS
runs have their own directories. No existing checkpoint was overwritten.

The matched comparison uses N25, absolute sigma.01, sensor time.237, seeds91/92,
two particles and four Heun steps, output8² at the declared validation times.
Whole-domain individual/ensemble, time-stratum, sensor and physics metrics are
saved. It establishes execution and equal-budget comparison, not a policy winner.
Heat focus uses reference information, including through positions themselves.
No test-based budget/policy selection occurred, and previous Fourier-inverse
superiority is not overturned. Flexible queries and continuous training support
do not prove useful temporal interpolation or spatial generalization.

Next bounded scientific task: profile the actual assigned CUDA GPU, then train
representation only for a predeclared pilot budget and examine held-out initial,
early and whole-interval errors across all six families before funding substantial
flow training. Keep the untouched test split reserved for final choices.

Actual browser reference/sampling/result arrays were also preserved under
`browser-artifacts/` before the isolated temporary service workspace is retired;
these are evidence copies, not relocated model-registration links.

Source archive: `outputs/continuous-time-20261028/pdeflow-source.zip`, with a
per-file SHA256 manifest. It includes source/frontend/lockfile/configs/notebooks/
scripts/tests/docs and excludes outputs, weights, datasets, environments, caches,
secrets and personal references. Existing scientific files and the staged Git
entries are fingerprinted before/after. No staging, reset, commit or push was
performed. Only the task-owned API/Vite verification services are stopped after
checks; process state is recorded. New checkpoints remain available for manual
import and are explicitly experimental.
