# Flow workspace/capacity verification — 26 October 2026

New evidence root: `outputs/flow-workspace-20261026/`. Prior accepted roots,
checkpoints/studies/screenshots/source archives were preserved. There was no long
pilot, full high-resolution training, new PDE family or repeat of historical
training-budget studies. CUDA and actual Colab remain untested.

## Delivered behavior and actual verification

| Check | Actual execution/evidence |
|---|---|
| Reported model-ID bug | Real Chromium against the saved preceding frontend build and an empty registry emitted model_id=""; `legacy-empty-id-browser.json`, `legacy-browser.log`, screenshot. New contract retains the strict legacy pattern and explicit registered/auto selection. |
| Python delivery regression | **204 passed**, zero failures/skips,138.00s: `python-delivery.log`. Includes existing PDE/reference/FK/controller/refinement/provider/managed suites plus new CFM, capacity, snapshots, local terms, selection, native reference/sampling and actual automatic reference association jobs. |
| Frontend | **33 passed**: `frontend-delivery.log`; TypeScript/Vite build passed (`build-delivery.log`); lint zero errors/three existing UI component-export warnings (`lint-release.log`). A known main-bundle size warning remains; linked SVG workspace is lazy-loaded and adds no large rendering dependency. |
| Real current browser workflows | **11 passed** in about1 minute (`browser-delivery.log`): heat2D,Darcy3D,chemistry2D full reference→sampling→actual reconstruction→automatic reference/error→generation→reopen/clear; delayed registry/parse; cancel/keep/return during clear; real FK ancestry; native1024² reference/bounded display; stale explicit selection; manual reference-free inverse; valid explicit selection. |
| Separate empty registry | **1 passed**,4.0s (`empty-browser-delivery.log`), deliberately empty real API workspace with PDEFLOW_EMPTY_REGISTRY=1. No job was posted, no empty model_id serialized; readable/expandable diagnostics and draft persistence. Total12 current real journeys; original bug reproduction is separate historical-build evidence. |
| CFM mathematics | 1D–3D paths/finite-difference velocity/shape/reduction, actual weight gradients/updates, analytic Gaussian Euler/Heun order, learned Gaussian toy and snapshot invariance/ancestry. `test_flow_capacity_snapshots.py`; audit details in flow-matching-audit.md. |
| Learned toy |2,000 updates/batch256,8,192 actual source→Heun64 samples; mean1.30336/std0.66601 vs1.25/.65; errors.05336/.01601, target mean MC SE.00718. `gaussian-toy/{toy.json,toy.pt}`. No exact-moment/PDE-quality claim. |
| Matched capacity | Two real production managed20-update runs,32²/T4,batch1, identical physical/calibration IDs and fixed validation. Small1,956 params CFM1.51396 vs medium132,436 CFM1.53007. All per-mode inverse results retained. `capacity-comparison/{small,medium,matched-summary}.json`, child runs/logs. |
| Native1024² MPS | Actual73,939-param heat U-Net,T3,batch1: native reference, CFM forward/backward/Adam update with changed weights, physical endpoint guidance gradient, actual2-step production Heun source→candidate plus real snapshots. Native candidate[1,1,1,3,1024,1024]. `probe-1024-delivery-mps/`. |
| Large local MPS | Actual1,257,436-param128²/T4 profile: same bounded update/guidance/Heun stages; `probe-large-128-mps/`. No full3000-update pilot. |
| MPS managed resume / CPU mapping | Medium64²/T4 actual update1→resume2; unchanged calibration namespace solve count4; safe architecture metadata roundtrip and rejected deliberately wrong parameter count; CPU load and real64² synthetic experiment. `medium-portability.json`, `medium-cpu-reload/`. Final-source resume/export/CPU reload: `delivery-resume/`, `delivery-checkpoint.json`. |
| Native metric/display checks |1024²/T3 reference and sparse observations tested. Controlled+0.25 perturbation on3,145,728 native field-time points gives MSE.0625,RMSE/MAE.25 with native quadrature, not model-quality evidence: `native-1024-evaluation-check.json`. Actual browser128² display payload<1MB retains1024² metadata. |
| CLI/configurations | Medium/large/high-res profiles resolve with explicit CPU dry-run for inspection; high-res CUDA execution is not implied. `medium-plan.json`,`large-plan.json`,`highres-profile-plan-cpu-override.json`. Capacity-suite plan and wrong-suite budget rejection tested; no full suite launched. |

Browser tests use actual local API/PyTorch jobs, not mocked numerical success.
Separate React unit seams are not counted as numerical integration. The recorded
source hashes distinguish stages: resource probes/matched comparison preceded the
final sample-inspection/suite-planning changes, which do not change the CFM/numerical
operations under test. Exact source for each stage remains in its manifest. Final
Python source SHA256 is
`342510081329e8e8fe40c85a7e4396a76d13b2146caf35b08378f44c95acdbc7`.
The final-source medium checkpoint is
`delivery-resume/runs/20261026T164956-heat-2d-medium-3bf9a78e/training/inference.pt`.
Strict resume of a pre-change source requires its recorded source; old inference
exports remain supported. No source mismatch is bypassed.

## Resource facts and scientific limits

Actual MacBook arm64 MPS,float32. Native1024² probe: reference.1999s, CFM update
1.0088s, guidance.9305s, two-step Heun1.2045s. Four ODE model evaluations plus two
passive endpoint calls. Process RSS470,695,936 bytes; MPS driver2,189,869,056 and
current52,457,216 bytes measured after the probe. MPS allocations are **not peak**
accelerator memory. Timings include cold/contended effects and are not a training
budget prediction. Large128² probe has its own measured stages/memory manifest.
A raw tensor-size bound is never presented as total neural-memory prediction.

The native field is generated by the actual neural sampler after one update, with
no CSS/image upscaling, tiling or hidden reference prediction. It is **not** a
scientifically validated trained1024² inverse model. No long guided schedule,
high-resolution FK population,1024³, CUDA or actual Colab is verified. Native
component limits and exact profiles are in capacity-resolution.md.

Matched validation obs+PDE unobserved relative L2: small1.47368,medium1.55722;
conventional Fourier inverse.52520. Larger was not better in this bounded test.
Both capacities/results are retained; no test-selected capacity/budget/particle.
Earlier test-budget non-monotonicity and negative baseline findings remain intact.

## Design and interaction evidence

The supplied Hessian reference animation was located at
Downloads/1792840994683.gif and inspected at frames0,30,60,89. Its visible source
attribution is linkedin.com/in/rajjoub; no license/author identity is inferred and
no animation is redistributed in the source archive. The adaptation uses its
linked mathematical explanation, not its unrelated Hessian subject. Read-only
inspection frames remain only in this evidence root.

Screenshots in browser/ include heat-2d-linked-focus.png, linked-narrow.png,
darcy-3d-spatial.png, chemistry-2d-spatial.png, actual generation panels,
fk-ancestry.png and native-1024-reference.png. Desktop/narrow screenshots were
visually inspected. A wire-mesh row-indexing issue was fixed and the fixed footer
was moved into document flow so it no longer covers fields. The referenced views
use real numerical data; no generated decorative animation is counted.

Clear confirms nonempty draft loss and separately handles keep job/cancel/return.
A component remount plus removed persisted draft/selection keys prevents late
responses from restoring the old editor. Actual keep-job completion and refresh
were checked; saved jobs/models still appear. Reopen retains the automatically
associated reference/error. Point inspection is native; figures can stride their
display only. Physical τ, generation s and training update remain distinct.

## Development failures retained and corrected

- Initial capacity gradient tests accidentally supplied float64 conditions to a
  float32 network; fixture now explicitly matches the selected dtype.
- Initial local Darcy test built its expected forcing at float32 against float64
  fluxes; forcing dtype corrected, without loosening the numerical tolerance.
- Initial isolation assertion searched the word "reference" in human description
  strings; it now checks the structural absence of reference/hidden-input keys.
- Full first pass201 passed/1 failed: an old test hard-coded exactly8 presets.
  It now verifies the four new real profile names. Final full run204 passed.
- Browser validation exposed a pending-parse race; validation waits, and invalid
  advanced JSON is distinct from loading. An old frontend error-text assertion
  was updated to the actual readable invalid-source message. Final33 tests passed.
- Native-grid browser test expected label "checkpoint:no"; actual UI accurately
  says "unavailable". Selector corrected; no model/physics behavior was weakened.
- Existing three lint export warnings and bundle-size warning remain recorded.

## Packaging, Git and handoff

`completion.json` records final archive/wheel hashes, payload integrity/exclusions,
backend/test status, service shutdown and Git checks. Portable archive excludes
outputs, weights/data, caches, node_modules/.venv, private files and the supplied
animation. SOURCE_PACKAGE.md explains installation and generating omitted artifacts.
New configuration/example files and package presets are included.

The original90-entry staged index remains unchanged (SHA256
`9c5f4385c42492042ad13d7a58854995f74e30e47ed9ba06ae7f4b65daead9c8`).
No commit was created or attempted; main still has no HEAD. New changes remain
unstaged/untracked alongside existing work. Staged/unstaged whitespace checks pass.
No action remains blocked by automatic approval review. Test services are stopped
and ports released at handoff, with saved shutdown evidence.

Exact startup, local/Colab-oriented training, native probes, resume, suites,
reconstruction/evaluation, transfer and verification commands:
[flow-workspace-handoff.md](flow-workspace-handoff.md).

The final wheel was additionally installed into an isolated target outside the
checkout. Its source identity matches the final hash above. A true 3D reference,
capacity-local CPU plan and actual 32² CFM update/guidance/Heun probe passed
(`wheel-smoke.json`, `wheel-probe-32/`). Archive payload hashes and exclusions
were checked (`packaging.json`, `source-manifest.json`). No dependencies were
downloaded for this isolated wheel check.
