# Observation alignment / persistence verification — 1 November 2026

Evidence root: `outputs/observation-aligned-persistence-20261101/`.
This is NEW execution for this task unless identified as historical. No substantial
training, model-capacity change, external service or real Drive/CUDA run occurred.
Previous reconstruction diagnosis and training-budget results were preserved;
they were not rerun. The user PDF and request identities are in input-identities.json.
PDF change assessment: observation-aligned-change-map.md.

| Check | Actual result | Saved evidence |
|---|---|---|
| Affected regression set |98 passed,3 actual-CUDA skips; one Starlette deprecation warning | affected-tests.log |
| Final focused scientific objectives + complete persistence |22 passed,17.78s; includes real MPS, batched vs accumulated B gradients, A2 reuse/resume ablation, corruption/retention/failed final upload and fresh-process continuation | final-focused-tests.log |
| Final eight notebook cells |2 passed,8.26s; real archive/setup/IPython/CPU A1/save/restore; no mocked numerical training | notebook-release-tests.log; notebook-release/notebook-run-all.log |
| Frontend field-loss controls |3 passed | frontend-tests.log |
| TypeScript + production build |Passed | frontend-build.log |
| ESLint |0 errors;3 existing UI fast-refresh warnings | frontend-lint.log |
| Existing Medium checkpoint compatibility |Current schema and strict state loading passed, original hash unchanged; no matching E/A2 recovered | compatibility.json/log |
| Capacity counts |Exact meta-module numel, presets unchanged; not a fit/memory claim | capacity-counts.json |
| Tiny B compute overhead |CPU and actual MPS; no CUDA peak-memory measurements | cpu-overhead.json; mps-overhead.json |
| Preservation |17,242 pre-existing output files,1,785,363,183 bytes:0 changed,0 missing | preservation-before.json; preservation-final.json |
| Whitespace / compilation |git diff --check and cached check passed; affected Python compileall passed | final-checks.json |
| Package |CRC, exact inventory, per-file SHA256, source tree and scientific identity checked | source-package-verification.json; adjacent ZIP .sha256 |

Suites overlap; do not sum their test counts into a unique total. The98-test
regression ran before the final metadata-only FM log correction and complete-save
status refinement; the22 focused tests and actual notebook were rerun after final
changes. No expensive completed experiment was repeated.

Commands actually used (from repository root, with environment-specific evidence
variables for the optional preserved fixture):

```bash
.venv/bin/python -m pytest -q tests/test_complete_persistence.py tests/test_observation_aligned_objectives.py tests/test_operator_wave.py tests/test_batched_training.py tests/test_batch_settings.py tests/integration/test_colab_training.py tests/integration/test_field_loss_training.py tests/unit/test_field_losses.py
.venv/bin/python -m pytest -q tests/test_observation_aligned_objectives.py tests/test_complete_persistence.py
# Real IPython8.37 was installed only in a temporary directory, preserving .venv:
PYTHONPATH=/private/tmp/pdeflow-observation-notebook-deps PDEFLOW_NOTEBOOK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m pytest -q tests/integration/test_colab_notebook.py
npm --prefix frontend test -- --run src/FieldLossControls.test.tsx
npm --prefix frontend run build
npm --prefix frontend run lint
python scripts/joint_model.py counts --output <new-counts.json>
python scripts/observation_auxiliary_benchmark.py --device cpu --output <new-benchmark.json>
python scripts/observation_auxiliary_benchmark.py --device mps --output <another-new-benchmark.json>
```

The supported user Node on PATH was used. macOS sandbox denied OpenMP shared memory
for child Python and initially denied network access for a temporary IPython install;
approved bounded executions resolved those environment restrictions. No rejected
launcher/commit was retried or bypassed. No verification service was launched; final
read-only process inspection found no pytest/uvicorn/Vite/Playwright/PDEFlow worker.
Git HEAD remains7a048496e91d24ee89b42a0a53649634dad53934; no commit or staging performed.
The working tree contains prior task work plus this implementation.

## What the focused checks demonstrate

- Beta0 exact loss equivalence and no decoder call, correct straight endpoint and
  one inverse calibration, exact x/y/t/quantity against independent decoder calls.
- Physical sigma units, optional Gaussian constants/discrepancy, explicit zero-noise
  rejection/tolerance, invalid/duplicate observations, padding NaNs, zero observations,
  activation range and configuration validation. Uniform selection is partition-
  invariant with independent per-example seeds.
- Actual observation encoder/flow gradients through frozen T/D, frozen state tensors
  unchanged, known-scalar authority, unknown-scalar hidden-reference independence,
  incomplete microbatch and accumulated/batched loss/gradient equivalence.
- Manufactured heat and first-order wave residuals, nonzero temporal/spatial
  derivative paths through actual learned T/D, periodic seams/derivatives, separate
  particle coordinate derivatives and finite actual CPU/MPS gradients.
- Existing signed/periodic RB detection, area/noise/no-region behavior, alpha endpoints
  and estimators in the affected regressions. Actual tiny A1/A2 training for MSE/RB/
  spectral; prior MAE/contrast tests retained and rerun. Spectral phase shift gives
  nonzero loss despite equal magnitudes, Parseval with power0, CPU/MPS gradients.
- Optional A2 physics reuses an identical frozen A1 checkpoint while changed same-
  stage objectives still reject resume; source/objective identities persist.
- CAS verified first/cadence saves, stale-pointer scan, incomplete/corrupt objects
  ignored, all conflicts checked before restore writes, conservative owned retention,
  failure traceback/status and bounded retries, explicit continue never mislabels a
  new failed final upload as persistently verified.
- Matching E+T+D+B loaded standalone, full stage optimizer/scheduler/scaler/RNG/counters,
  exact CPU stopped/resumed replay and fresh-process B extension2→3 updates.
  Repeated finalization returns the identical snapshot when unchanged; no extra ZIP.

## Saved fixtures and their qualification

`artifact-evidence-final/complete-fixture/` contains a low-level objective-plumbing
fixture with A1=2, A2=2, B=2 successful updates (each2 attempts,0 skips). Each update
uses one heat and one wave realization, actual neural microbatch2, RB alpha.5,
explicit B beta=.0001 and A2 physics weight=.0001/query1. This deliberately exercises
losses and artifacts, not a gate-passing or accurate model. It was created through
the low-level trainer for testing; production orchestration still enforces the gate.
`complete-result.json` records hashes/stages and fresh-process2→3 extension evidence.
`complete-store/` is its verified local persistent-store fixture. Diagnostic/resumable
views include exact E and dependencies; downloads are optional views, not weights
selected for scientific performance.

`notebook-release/notebook-run/<unique-id>/` contains the actual notebook A1 fixture:
2 successful updates, no A2 or B training. The quality gate is **FAILED**, whole-field
validation MSE4.7508358623, constant baseline1.1744008912, ratio4.0453271945. Gate
override=False. Final inference view is unavailable, diagnostic/resumable views
explicitly partial. The actual restore cell recovered source/config/E/optimizer.
This failed gate was not concealed or bypassed.

Existing Medium remains at its original path under
`outputs/local/model-imports/2de8489ab6f34bc7985a583451f62b6a/coordinate.pt`:
B2500 successful updates, SHA256
`ce489d7b97c402317349c712545fb599bbc36e30abccd107f92abdad0e2a206e`.
Its matching E/A2 remains unavailable. Current strict tensor/schema loading passes;
changed scientific source is NOT exact-resume compatible by default. Earlier
single-case correction/poor-reconstruction evidence remains historical, unchanged.

Exact parameter counts (training / portable inference, excluding buffers):

| Capacity | Training | Inference |
|---|---:|---:|
| Smoke |60,148 |50,620 |
| XSmall |13,171,972 |11,697,668 |
| Small |36,960,772 |32,459,908 |
| Medium |90,363,908 |80,228,868 |
| Large |252,707,332 |229,935,364 |
| XLarge |644,270,084 |587,017,220 |

## Cost evidence and limitations

Compute-only random-latent fixtures, same2-example mixed-family batch, same Smoke
architecture/points/conditioning/precision, s=.75 (all auxiliary endpoints active),
2 warmup +5 measured forwards/backwards. No optimizer steps or trained predictions:

| Device | FM median | FM+endpoint median | Ratio | Examples/s FM → aligned |
|---|---:|---:|---:|---:|
| CPU FP32 |.002982s |.010596s |3.55× |670.6→188.8 |
| MPS FP32 |.027963s |.168128s |6.01× |71.5→11.9 |

These omit reference/frozen-target prep, validation, persistence and optimizer;
these ratios are NOT end-to-end training/T4 estimates. No CUDA peak-memory figure
exists; MPS reports current allocator bytes after each variant, not peak or memory
improvement. Use the unchanged-workload Colab preflight and acceptance procedure
in observation-aligned-training.md before a substantial run. Optional physics
higher derivatives and structured N² grids have additional cost. No compute-equal
scientific comparison or held-out learning benefit has been measured.

Failures preserved honestly: initial objective tests lacked the graph argument in
the test fixture (corrected); sandbox OpenMP child-process denial; actual MPS
unique_dim unsupported (fixed with explicit CPU metadata validation); a test-only
CPU normalization tensor in an MPS fixture (corrected); frontend expectation for
an unnecessary extra zero setting (UI preserves old payload shape). Logs include
objective-tests.log, focused-tests-sandbox-attempt.log, mps-unique-failure.log,
mps-test-fixture-failure.log and the final successful logs. No tolerance was loosened
to hide a numerical error. The real MPS residual and spectral gradients passed.

Actual Colab, Drive mount/failure behavior, T4/CUDA/FP16 and cross-device deterministic
replay remain UNTESTED. No claim of improved reconstruction, physical conservation,
posterior calibration or universal PDE capability. The historical conventional
Fourier inverse remains better; recorded test-budget quality was non-monotonic,
and test performance was not used to choose a budget.
