> Historical implementation record. The active heat/wave physical-state v2 workflow is documented in [operator-wave-model.md](operator-wave-model.md) and the current README. Previous whole-trajectory checkpoints are preserved and incompatible.

# Joint latent implementation verification —30 October2026

All execution below occurred during this implementation. Earlier verification
records remain historical; their successful runs are not counted as checks of
the replacement. No substantial training or accuracy study was launched.

Evidence root: outputs/joint-latent-20261030/. Current portable pipeline-check
artifacts: verified-cpu/export/coordinate.pt and verified-mps/export/coordinate.pt.
Both have two representation and two flow optimizer updates, with a real stop and
resume after the first update in each stage. They are not adequately trained models.
Initial prototype smoke directories remain preserved as development evidence.

## Checks and boundaries

| Check | Actual result | Evidence |
|---|---|---|
| Affected Python suite | 84 passed, 2 CUDA-only skipped | affected-tests-final.log |
| Notebook active structure after package exclusion update | 1 passed | notebook-structure-final.log |
| Actual six-cell notebook Run-all on local CPU | Passed setup/archive caching, preflight, AE/flow, persistence/export | notebook/notebook-run-all.log; notebook/notebook-run/ |
| Affected frontend tests | 19 passed in five files | frontend-tests-final.log |
| TypeScript + Vite build | Passed | frontend-build-final.log |
| ESLint | Zero errors; three existing UI fast-refresh warnings | frontend-lint-final.log |
| Real managed browser training | Passed, two optimizer updates per stage | browser-first.log (second test); browser-training.json/png |
| Real browser known/unknown D | Passed explicit final checkpoint selection, numerical generation, independent API validation, metrics/regions and exact t=.237 decoding without new generation | browser-final-selected.log; browser-inference.json; browser-known-D.png; browser-unknown-D.png |
| CPU | Real training/resume, portable export, no-encoder reload, inference, derivatives | hardware-verification.json; verified-cpu/ |
| Apple MPS | Real training/resume, FP32 export, CPU remap, known/unknown inference and coordinate derivatives | hardware-verification.json; verified-mps/ |
| CPU/MPS matching | Max absolute2.38e-7 known D,2.09e-7 unknown D, from identical explicit CPU-created latent noise | hardware-verification.json |
| CUDA / actual Colab | Not executed | Two CUDA skips; local notebook is not Colab hardware evidence |
| CLI generation/reconstruction | Known/unknown demo and direct CLI reconstruction passed; static initial-head arrays finite and exported | verified-demo/; cli-reconstruction/; initial-output-check.json |
| Python wheel | Built without dependency installation | wheel.log; package/pdeflow-0.1.0-py3-none-any.whl |
| Source ZIP | Manifest/CRC/hash/content checks recorded | archive-verification.json |
| Whitespace | git diff --check passed | final checks; no staged diff |
| Existing artifacts | All110 original .pt hashes unchanged; original HEAD unchanged | preservation-before.json; preservation-after.json |

The Python suite covers joint group interaction, masks/statuses, scalar/vector/
sampled-field context paths, set permutation/padding/empty behavior, periodic local
anchors, chunk/query order, actual second derivatives, true1D/3D module execution,
straight-path/group reductions, real region-loss gradients, train-only moments,
cache identity, frozen dependency, exact same-backend resume, changed-loss/policy
rejection, portable export without encoder/data, independent API compatibility,
legacy rejection and an actual optional decoded auxiliary update. Generic field
module tests do not establish a trained variable-coefficient specialist.

Exact final affected command (from repository root):

```bash
PDEFLOW_NOTEBOOK_PYTHON=/private/tmp/pdeflow-colab-notebook-venv/bin/python \
PDEFLOW_NOTEBOOK_EVIDENCE=outputs/joint-latent-20261030/notebook \
.venv/bin/python -m pytest tests/test_joint_latent.py tests/unit/test_coordinate.py \
 tests/unit/test_field_losses.py tests/integration/test_coordinate_pipeline.py \
 tests/integration/test_field_loss_training.py tests/integration/test_field_loss_api.py \
 tests/integration/test_continuous_heat.py tests/integration/test_colab_training.py \
 tests/integration/test_colab_notebook.py -q
```

The isolated notebook interpreter was already present with real core/IPython
dependencies. On another machine set that variable to an equivalent isolated
interpreter; absent it the real Run-all test explicitly skips. Frontend commands
used the user's Homebrew Node22.23.3, not a Codex runtime path:

```bash
npm --prefix frontend test -- --run src/TrainModel.test.tsx src/HeatPlayground.test.tsx src/FieldLossControls.test.tsx src/heat.test.ts src/heatProfiles.test.ts
npm --prefix frontend run build
npm --prefix frontend run lint
# With local API/frontend running, from frontend/:
npm run e2e -- e2e/joint-latent.spec.ts
```

The final browser inference test selects SHA256-prefix92b02fcfcaf38ff807eefb4c202e42c5
explicitly, rather than relying on discovery order. It verifies the inverse
request has no full initial/reference input and no known D in joint mode. It
changes metric/map and physical time while counting generation requests. The
managed-training browser check was not needlessly repeated after it passed.

## Retained failures and fixes

Initial tensor smoke found a module attribute named type colliding with nn.Module;
it was renamed. Initial new test fixtures violated the existing six-family split
minimum; the fixture was corrected, not the data contract. The first affected suite
had74 passes and six failures: snapshot training had an inner import shadowing
select_context; microbatch comparisons exposed softmax-invariant key biases whose
near-zero floating gradients were amplified by Adam. The import was corrected and
redundant key biases removed. No tolerance was loosened; repaired checks passed.

Real browser joint inference exposed a legitimate post-inference comparison
mismatch: unknown D differs from known evaluation truth. Comparison now admits only
this declared unknown scalar difference, uses generated D for per-sample residuals,
and attaches true D solely as evaluation metadata. A comparison error no longer
causes an endless successful-job polling loop. Repaired real browser checks passed.

Optional PDE auxiliary validation initially ran under no_grad; actual coordinate
derivatives now execute in an explicit grad scope even during validation. The
one-update auxiliary stage passed with a byte-identical frozen AE dependency.

The first unprivileged Vite launch encountered local-socket EPERM. The permitted
local listener launch then succeeded. No automatic approval rejection remains.
Task-owned API/frontend services were stopped after verification; process-check.txt
records no listeners on8000/5173. No unrelated processes were signaled.

## Scientific limits

The CPU smoke representation objective was .33235 then1.00326; flow velocity loss
was2.52203 then2.19177. Two updates are not a convergence or throughput benchmark.
MPS warmup was several seconds; second updates .176s(AE) and .387s(flow), versus
CPU .017s/.013s. These tiny profiled shapes do not predict large-GPU performance.
No GPU utilization claim follows from memory statistics.

The retained single validation demo has true D=.0659557993. Known-D field MSE is
.05304454; unknown-D field MSE is .05915397, with D absolute errors .00246038 and
.00236525. MSE is averaged over both generated samples and all equal-weight query
points at the listed demo times, not a uniform-time integral or ensemble-mean MSE.
The supplied known D roundtrip differs by3.5e-9 in FP32. These tiny-run examples
provide no evidence of reliable identification, accuracy, calibration or improvement.
Unknown initial fields and D may be non-identifiable from sparse measurements.

Only heat2D continuous joint generation is integrated into the normal playground.
Spatial-coefficient/forcing/vector paths have module-level support only; no trained
joint Darcy/chemistry specialist is advertised. The initial and solution heads do
not enforce exact u0=u(t=0), conservation or PDE satisfaction. Local-neighbour
switches are piecewise smooth. Auxiliary PDE probes are bounded, not exhaustive.
Old scientific studies are retained; conventional Fourier inversion was better
there and test quality was non-monotonic across budgets. No budget or loss was
selected using test performance, and this new architecture has no matched accuracy
benchmark yet.

Use joint-latent-model.md for exact counts/formulas/decisions and README.md for
current commands. Next: validate the representation on fixed held-out families,
time strata, regions/background, spectra and D before allocating a substantial
flow budget. An unexecuted500-update XSmall four-loss plan is retained in
future-comparison-plan/; equal updates do not mean equal compute.
