# Verification record: 2026-10-21

This is a chronological record. The latest multi-specialist recovery section
below supersedes earlier scope/hardware/Git statements for the current source.
Current commands/status are in [multispecialist.md](multispecialist.md).

Executed locally on macOS 15.2 arm64 with Python 3.12.3, PyTorch 2.5.1,
NumPy 2.1.3, Matplotlib 3.9.2, pytest 8.3.4 and two CPU threads for smoke
profiles. `pdeflow hardware` reported CUDA unavailable and MPS unavailable to
this process. No accelerator training or Colab execution was performed. The
supplied directory has no Git metadata; checkpoint source SHA256 is recorded,
and Git revision is null. Scientific references/ was empty.

## Actual commands and outcomes

Dependencies were installed in `.venv` using uv with the exact direct versions
in pyproject.toml. The initial sandboxed PyPI access failed DNS resolution;
the authorized dependency download then succeeded. The default uv cache was
not writable in the sandbox, so a configurable temporary cache was used.
The editable install completed:

```bash
UV_CACHE_DIR=/tmp/pdeflow-uv-cache uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -e .
.venv/bin/pdeflow hardware
.venv/bin/python -m compileall -q src
.venv/bin/python -m pytest tests/numerics tests/unit -q
.venv/bin/python -m pytest -q
.venv/bin/pdeflow smoke --config configs/experiments/cpu_2d.json --output outputs/accepted_cpu_2d
.venv/bin/pdeflow smoke --config configs/experiments/cpu_3d.json --output outputs/accepted_cpu_3d
```

The first focused numerical/unit run passed 27 tests. The initial integration
suite passed 35. A later expanded run found one test assumption failure:
`np.isfinite(None)` when all four 1D nodes were observed and the newly corrected
metric returned null for an empty unobserved set. The test now explicitly checks
null for this case; no numerical tolerance was relaxed. Added independent
candidate-population and nonuniform-time tests. The final full suite passed
**37 tests in 22.77 seconds**, with 14 upstream Matplotlib/Pyparsing deprecation
warnings. These warnings do not affect output. Earlier Fontconfig cache-write
messages were fixed by placing its configurable cache under the output path.
Matplotlib successfully produced real PNGs; final 2D and 3D layouts were viewed.

Tests cover:

- Analytic derivative accuracy and second-order convergence in true 1D/2D/3D,
  including nonunit lengths/shifted origins and vector divergence.
- Independent analytic Fourier heat decay, conserved spatial mean and random
  profiles varying along every spatial axis; heat residual convergence.
- Poisson manufactured fields, forcing compatibility and zero-mean gauge.
- Nonuniform physical-time differentiation on an analytic quadratic.
- Training-only normalization, reversible time/channel packing, point indices,
  physical coordinates, observed zeros, missing masks and observation gradients.
- Safe scalar expression evaluation, capability/layout/configuration rejection.
- Euler/Heun first/second order, uniform/quadratic schedules, forward counts.
- Autograd through the actual CNN endpoint Jacobian versus central finite
  differences in all dimensions; a controlled gradient update reduces energy.
- Bitwise equality of same-backend uninterrupted/resumed model state, training
  losses and RNG streams; saved null scheduler/scaler; actual optimizer changes.
- Removing the entire training array and manifest before inference, then
  reconstructing from checkpoint and standalone known metadata/measurements.
- 1D and genuine 3D train/save/resume/reconstruct integrations in addition to
  the 2D/3D CLI acceptance profiles.
- Fourier-basis inverse recovery in well-determined analytic 1D/2D/3D cases.
- Hash integrity, incompatible resume/layout rejection, atomic-write failure
  preserving the last complete artifact, and candidate independence.
- An artificial CUDA metadata tag loaded on CPU tests metadata portability;
  it is explicitly not a CUDA or cross-device execution test.

The two requested CLI smoke profiles completed real data generation, optimizer
updates, half-budget save/resume, safe inference reload, three sampling modes,
Fourier inverse reconstruction, physical metric JSON, sample NPZ, figures and
an explicit re-evaluation of each result. No hidden initial condition was
provided to the network, guidance or inverse baseline.

## Measured smoke results

One held-out trajectory and two candidates per profile. Not tuned on test truth.
All values below are physical-space measurements. Relative error uses the
ensemble mean on unobserved nodes/times with spatial/time quadrature weights.

| Profile / method | Observation RMSE | Unobserved relative L2 | FD PDE RMSE | Forward calls |
|---|---:|---:|---:|---:|
| 2D unguided | 1.208357 | 1.301965 | 17.325999 | 24 |
| 2D observation | 0.676390 | 1.248729 | 16.668391 | 35 |
| 2D observation + PDE | 0.676075 | 1.248278 | 16.602790 | 35 |
| 2D Fourier inverse | 0.056241 | 0.211206 | 0.128648 | 0 |
| 3D unguided | 1.271374 | 1.409297 | 15.092225 | 20 |
| 3D observation | 0.999255 | 1.387994 | 15.042923 | 29 |
| 3D observation + PDE | 0.999238 | 1.387957 | 15.038685 | 29 |
| 3D Fourier inverse | 0.052103 | 0.957921 | 0.163721 | 0 |

2D: 30 trajectories, grid 8x8, four physical times, 30 optimizer updates,
10 sensors across all saved times, 12 Heun generation steps, two candidates.
3D: 18 trajectories, grid 8x8x8, four physical times, 20 optimizer updates,
20 sensors, 10 Heun steps, two candidates. Per-step sample counts include one
extra endpoint model call for every guided step except the last; backward
passes are counted separately (12/10 per guided run). Counts are batched
forward invocations, not per-candidate scalar equivalents.

Data generation took approximately .025 s (2D) and .035 s (3D). Accumulated
training-loop/validation/checkpoint time was .126 s and .113 s respectively;
this excludes Python startup, imports, data loading, plotting and environment
installation and is not a training performance benchmark. The training process
RSS high-water marks were 247,676,928 and 254,754,816 bytes. Sampling alone
ranged .011–.038 s (2D) and .015–.046 s (3D). CPU Fourier inverse solves took
about .002–.0024 s. These are single tiny runs with substantial library overhead
outside the timed regions, not reliable throughput or speedup estimates.

The learned models are strongly undertrained and the conventional inverse is
better. Observation correction improves observation fit in these particular
runs; this is not evidence of general scientific performance. The 3D baseline
also has substantial unobserved error despite good observation fit: sparse
measurements and a truncated Fourier prior do not identify the full field.
No calibrated posterior or generalization claim follows from these results.

## Portable sampling and packaging checks

Created `outputs/euler_quadratic.json` by changing only inference.stepper to
`euler` and inference.schedule to `quadratic` in the 2D profile. Executed:

```bash
.venv/bin/pdeflow reconstruct --config outputs/euler_quadratic.json --device cpu \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --problem outputs/accepted_cpu_2d/data/test_problem.json \
  --observations outputs/accepted_cpu_2d/data/test_observations.json \
  --output outputs/accepted_portable_euler_quadratic
UV_CACHE_DIR=/tmp/pdeflow-uv-cache uv build --python .venv/bin/python \
  --no-build-isolation --out-dir outputs/dist
.venv/bin/python scripts/package_source.py --output outputs/pdeflow-source.zip
```

The portable reconstruction succeeded without any evaluation reference, using
12 Euler forward calls (23 total with endpoint guidance). The build produced
a pure-Python wheel and source distribution; the source archive contains
package, configs, tests, docs and the notebook, not datasets, weights or caches.
All five notebook Python cells compile. Colab source acquisition, installation,
storage and CUDA execution remain untested; the notebook requires explicit
user-selected source and persistent storage and contains no numerical algorithms.

## Output locations and limitations

The current accepted outputs are `outputs/accepted_cpu_2d/` and
`outputs/accepted_cpu_3d/`. Each contains `data/`, `training/` and
`reconstruction/`; `comparison.json` and each method's `metrics.json` record
actual values and costs. Figures and arrays are in each method's subdirectory.
`outputs/accepted_portable_euler_quadratic/` holds the no-reference alternate-sampler
run. Distributions and source ZIP are in `outputs/dist/` and
`outputs/pdeflow-source.zip`. These are ignored runtime artifacts.

Periodic boxes, scalar heat, known constant diffusivity and fixed physical-time
windows are the trained scope. Off-grid observations, unknown parameters,
interacting particles, neural Poisson inference, other boundaries/geometries,
resolution generalization, Darcy, chemical systems, waves, fluids, diffusion,
CFG, stochastic sampling, FK and refinement are not implemented. See the
capability matrix and next complete Darcy slice in README.md. Nothing was
published, pushed or sent to a remote service for project execution.

Final interface review moved parameter-array meanings to the data adapter and
known-scalar construction to the core Conditioning adapter. The trainer and
sampler no longer extract heat-specific parameters. After this source change,
the full 37-test suite and both CPU acceptance profiles were rerun successfully.
The final source distribution includes configs, tests, docs and notebook; a
separate wheel installation loaded the exported true-3D checkpoint on CPU.

# Continuation: scientific audit and local workbench

The accepted first-slice outputs above were preserved. New outputs use
outputs/workbench-20261021/; workbench API job history uses outputs/workbench/.
Initial baseline verification re-ran the existing 37 tests successfully in
22.98 s. Source/checkpoint/data inspection confirmed the reported modular
heat slice, with one reference-boundary improvement: CLI inference now finishes
all algorithms before opening optional evaluation truth.

## Integrity and mathematical evidence

Added a three-reference regression across unguided, observation-guided,
observation+PDE and Fourier reconstruction: absent/correct/unrelated references
yield bitwise identical CPU samples. Only evaluation changes. The API worker
request cannot include a reference; an explicit post-inference endpoint attaches
or removes evaluation truth while preserving sample values. Added physical
residual scaling through a nontrivial affine decode, incomplete-checkpoint
rejection, direct inference configuration validation, duplicate observation
rejection and strict physical-coordinate CSV/JSON validation. Twelve actual
stepper/schedule/mode combinations preserve the same physical problem and
produce the expected forward-call counts. No tolerance was loosened.

The API tests cover real subprocess completion, no-reference diagnostics,
bounded orthogonal slices, reference attachment/removal, schema/physics/model
errors, oversized input, path rejection, idempotency, cancellation, worker crash
propagation and interrupted restart/shutdown. They create their own temporary
small 2D/3D checkpoints; no accepted outputs are required for pytest. A managed
worker first failed under the execution sandbox with OpenMP shared-memory
error #179. The same tests passed with normal host access; no numerical
fallback was introduced. A persisted-active-record test checks startup recovery;
this is not a power-failure/filesystem-durability guarantee.

## Actual backend verification

A sandboxed torch probe reported MPS false. An explicit host-access probe
reported MPS true, CUDA false. Executed:

```bash
.venv/bin/python scripts/verify_backends.py --device mps \
  --output outputs/workbench-20261021/mps-regression
```

This performed six actual float32 MPS optimizer updates, checkpoint/resume,
endpoint estimation, observation and PDE guidance gradients, reconstruction,
export, CPU device remapping and CPU reconstruction. The recorded runtime is
mps, not a concealed CPU fallback. Numerical references and post-sampling
spectral diagnostics intentionally remain CPU float64. Cross-device bitwise
equality was not claimed. See mps-regression/backend_verification.json.
CUDA remains unavailable and CUDA/Colab execution remains UNTESTED.

## Budget study and dimension regression

A 10-update initialization/load/timing probe took .799 s and justified the
bounded 0/100/500 ladder. Executed:

```bash
.venv/bin/python -m pdeflow.experiments \
  --config configs/experiments/local_budget.json \
  --output outputs/workbench-20261021/budget-study
.venv/bin/pdeflow smoke --config configs/experiments/cpu_3d.json \
  --output outputs/workbench-20261021/cpu-3d-regression
.venv/bin/pdeflow smoke --config configs/experiments/cpu_1d.json \
  --output outputs/workbench-20261021/cpu-1d-regression
```

The budget study took 31.61 s including comparisons/figures, uses one training
trajectory and fixed split/mask/source seeds, and selects settings solely on
two validation problems. All test budgets use those frozen settings and two
fixed held-out cases with four candidates. See pilot.md for all results.
Validation objective decreases 2.3326 → .8795 → .5743, but observation+PDE test
relative error is 1.1097 → .7515 → 1.1899; Fourier inverse .2364. These negative,
non-monotonic results were retained. No best model was selected from test error.
Individual-sample errors, absolute field error, observation/residual/mass,
spread/coverage, actual timing/calls and process RSS are in metrics.json.
Learning, budget/cost-quality, field comparison and spread PNGs were generated.
The genuine 3D regression includes 20 updates, resume, reload and all four
comparisons. A small 1D 20-update profile additionally supports real UI curves.

## Workbench verification and fixes

Used real React 19.3.1, TypeScript 6.0.3, Vite 8.3.5, Tailwind 4.3.3 and
registry-generated shadcn components; Node 24.19.0/npm 10.2.4. Early old-lockfile
peer conflicts were resolved with a clean compatible dependency resolution,
not --force. The one-time shadcn generator introduced advisory-affected tooling;
after retaining generated source/attribution, it was removed from dependencies.
The resulting dependency audit reported zero vulnerabilities.

Executed npm production build/type checks, eight frontend unit tests and lint
(no errors; three generated-component fast-refresh export warnings). UI unit
coverage includes canonical form/JSON synchronization, invalid draft retention,
dimensions, readable operator preview, invalid import, submission gating,
duplicate prevention and failed-job reopening. Initial browser failures exposed
accessibility-selector assumptions and macOS headless render throttling; actual
selectors and browser launch flags were corrected. Browser verification then
found a real cancellation defect (GET instead of POST); the button now sends
the correct method and cancellation was re-executed successfully.

Live Chromium tests exercised the actual localhost API and real numerical
workers: editable diffusivity, synthetic sensors, CSV export/import, compatible
model/request validation, reconstruction, real result/slice retrieval,
configuration/NPZ export, reference-absent metrics, post-inference reference
attachment, true 3D slices/sample spread, unsupported equation rejection,
malformed imports, cancellation and history reopening after refresh. Desktop
1440x1000 and narrow 430x900 screenshots were visually inspected; no horizontal
page overflow or uncaught page errors were observed in the recorded workflow.
Plot scale/labels and reopened-run status were corrected during review.

The documented scripts/dev.py launcher was actually started, both services
returned HTTP 200, Ctrl-C was sent, and both child service ports closed with
launcher exit code 0. See launcher-verification.json and launcher.log. The API
binds localhost, uses configured roots, opaque IDs, safe checkpoint loading,
strict bodies, one compute child and bounded slice/artifact export. It is a
local development tool, not a hardened multi-user or network-facing service.

## Colab and remaining limits

Notebook cells compile. The actual archive bootstrap was executed locally:
package-root discovery, stable cache reuse, changed-source rejection and
traversal rejection passed. The notebook now checks source identity/commit,
restores or generates data in a disposable cache, keeps a persistent copy,
resumes persistent training checkpoints and copies a portable export bundle.
This verifies setup logic, not actual Colab storage, Git networking, CUDA
installation/training or cross-device CUDA agreement; those remain untested.

Darcy has a concrete conservative finite-volume/gauge/positive-coefficient
specification in darcy.md. No Darcy execution, neural Poisson reconstruction,
chemistry, waves, fluids, diffusion/CFG, FK, refinement, off-grid interpolation,
variable time-window or new-grid generalization is implemented. The heat pilot
is limited to two test cases and one seed. Process RSS is a lifetime high-water
mark, not isolated accelerator memory. Initial accepted outputs were neither
overwritten nor deleted. No cloud service, publishing, account or remote Git
project was created.

## Final continuation check results

- Final full Python suite: **45 passed in 30.91 s**, 15 upstream deprecation
  warnings. The twelve real inference substitutions are included.
- Frontend: **eight unit tests passed**, production build and explicit
  TypeScript check passed. Lint has no errors and three generated shadcn
  fast-refresh export warnings. Clean npm ci from the lockfile was verified.
- Real Chromium suite: **four passed in 50.5 s**: 2D/import/export/history/narrow
  layout; true 3D/reference/slices; rejection/cancellation; actual 1D curves.
- The localhost services were stopped through the tested launcher after QA.
- Source and the npm lockfile were staged in local Git. The attempted commit
  did not execute; recovery confirmed no HEAD/commits and no remote. Accepted
  outputs remain ignored and preserved.

Browser screenshots are browser-problem-debug.png, browser-results-desktop.png,
browser-results-narrow.png, browser-3d.png and browser-1d.png under the new
continuation output root. The final source ZIP includes frontend source and
lockfile as well as Python, configs, tests, documentation and notebook; excludes
weights/data/caches/node_modules. pilot-source.zip preserves the scientific
source snapshot corresponding to the executed pilot.

## Recovery session

See handoff.md for the separate recovery evidence table and exact commands.
No scientific source changed and no training or browser workflows were rerun.
The full Python suite remains the recorded prior 45-pass execution; recovery
collected the same 45 tests and inspected the empty cached failure map. Eight
frontend tests, production build (including TypeScript) and lint were executed
again successfully; lint retains three nonfatal generated-component warnings.
Recovery loaded all three dimension exports on CPU and validated their public
observations. Saved MPS, launcher, job, browser and budget records were audited.
The source packager now excludes generated egg-info and adds payload checksums
and regeneration instructions; the final ZIP was extracted and verified.
Logs/manifests and the final archive are under the continuation recovery/ folder.
No Git mutation was attempted. Existing staged files and accepted outputs were
preserved; recovery documentation/package changes remain outside the index.

# Symbolic editor and larger-profile continuation

The latest task's outputs are under outputs/symbolic-20261021/. Existing
scientific datasets, checkpoints, arrays, manifests, pilot metrics and archives
were preserved. The unchanged staged Git index was not modified; no commit,
remote, push or publication was attempted. An unrelated untracked .idea/ was
left alone.

## Actual checks in this implementation session

- Full Python suite: **78 passed in 33.71 s**, 15 upstream deprecation warnings;
  stdout saved in python-tests-final.log. Earlier focused parser/shard run:
  31 passed; an added declared-subscript regression passed with all 29 parser
  cases. API coverage now reparses current mathematics, rejects stale/unsupported
  source, checks measurement consistency, and exercises a real worker result
  retaining the full request. It includes JavaScript numeric JSON spelling.
- Frontend: **16 tests passed**, including the preserved eight behaviors adapted
  to the symbolic-first UI, full request state, parameter/preset preservation,
  conflict review, invalid draft blocking and malformed import. MathInput is
  mocked only in React unit tests because MathLive's Node export intentionally
  lacks a DOM MathfieldElement. Real MathLive is exercised in Chromium.
- TypeScript/production build passed. Npm install added exactly MathLive 0.111.1;
  audit reported zero vulnerabilities. Vite bundles 19 local font assets and
  reports a nonfatal >500 kB chunk warning (MathLive plus app is ~1.19 MB JS,
  ~344 kB gzip). Lint passes with the same three generated shadcn fast-refresh
  export warnings. No force/legacy-peer workaround was used.
- Final live Chromium suite: **five passed, no skips/flakes/failures, 61.12 s**.
  browser-results-final.json is the complete report. Actual workflows include
  empty equation, preset insertion, fraction toolbar/focus, keyboard +u editing
  inside MathLive and undoing it, λ/Ω/ℬ/ℐ/noise changes, real reconstruction,
  editable unsupported Darcy, complete-request export/reimport, malformed source
  blocking, actual 1D curves and true 3D slices, CSV import/export, reference
  attachment/detachment, cancellation, history reload and 430 px narrow layout.
  The symbolic workflow observed zero external HTTP requests and page errors.
  New screenshots are in the current output root; desktop/results and narrow
  layouts were visually inspected. The earlier passing run was five in 24.73 s;
  final report timing is the current complete run, not a performance benchmark.
- CPU 32² intended-profile preflight passed. Actual host MPS float32 preflights
  at 32² (width32/3blocks/batch16) and true 16³ (width16/2blocks/batch2) performed
  two optimizer updates each, validation, checkpoint save/resume, endpoint and
  observation/PDE guidance, export and CPU load/reconstruction. They use an
  explicit 12-instance 8/2/2 mini dataset with the intended layout/distribution,
  not the full large training sets. preflight-mps-{2d,3d}/preflight.json retain
  actual backend/version/size/allocation evidence. Resource values and limitations
  are in local-training.md. Cross-backend bitwise equality is not claimed.
- Four-instance reference/compression probes produced estimate-{2d,standard,3d}-final.json.
  Manifest planning is separated from per-instance reference/compression time.
  Larger 2400/12000/1200-instance datasets and long training were not launched.
- CLI argument names were checked with generate/train/estimate-data/preflight
  --help; generate/train/preflight were also executed on the bounded cases.
  CUDA remains unavailable and **CUDA / actual Colab execution is UNTESTED**.

## Failures found and resolved

The first React unit import used MathLive's Node/SSR export, which has no
MathfieldElement. Unit tests now use an explicit component seam; browser tests
still instantiate the real mathfield. TypeScript caught an invalid Testing
Library `exact` option in two new tests; it was removed without changing match
behavior. Application lint caught an obsolete Textarea import, now removed.

The first browser run found a real execution comparison defect: JavaScript
serialized 0.0/-1.0 as 0/-1, while a JSON digest treated those spellings as
unequal. Typed ProblemSpec structural equality now tolerates numeric spelling
and retains exact physics/operator checks; a regression covers it. Real keyboard
editing also emitted duplicate equivalent input notifications and invalidated a
pending parse without scheduling a new one. Identical-state notifications now
leave the pending/current parse intact. No physics tolerance was relaxed.

One intermediate 2D browser assertion was interrupted by a development-page
reload while files were being edited; its numerical job succeeded. Sources were
then frozen during browser execution and the entire suite passed, followed by
the final five-workflow run with the completed preset metadata and new paths.

Preservation exception: the original browser test screenshot paths still named
outputs/workbench-20261021/. Rerunning them refreshed four QA PNGs there
(browser-results-desktop.png, browser-results-narrow.png, browser-3d.png,
browser-1d.png). Original pixels were not retained in a backup. Those refreshed
captures were copied into the new task root, and all test screenshot paths now
write only outputs/symbolic-20261021/. Scientific data/weights/metrics and the
historical pilot-source/archive artifacts were not changed. This exception is
recorded rather than claiming all older QA files remained byte-identical.

## Scope and remaining evidence limits

All nine templates parse/roundtrip within the documented subset; only recognized
heat becomes executable. Vector/coupled parsing does not establish a vector or
chemical residual, generator or trained model. Standalone Poisson numerical
verification remains separate. Off-grid/static/system observation-file adapters,
new boundaries/geometry, unknown coefficient inference, arbitrary time/grid
layouts, diffusion/CFG, FK and refinement remain planned. See
symbolic-editor.md, equation-specialists.md and local-training.md. The earlier
Fourier inverse advantage and non-monotonic test-budget results remain intact;
brief preflights do not establish training quality.

A final small check confirmed that passing a sharded large profile to the
historical small-NPZ budget-study runner raises a descriptive error before
creating output. Its existing NPZ experiment path is unchanged; no pilot was
rerun. Source/package syntax, documented links/CLI profiles, archive checksums
and staged/unstaged whitespace were checked after documentation packaging.

A final rank audit added explicit rejection of vector powers (no implicit
componentwise/tensor convention). All 29 parser cases, including that rejection,
passed again in parser-tests-final.log; other numerical code was unchanged.

Current source delivery: outputs/symbolic-20261021/pdeflow-symbolic-source.zip
with SOURCE_PACKAGE.md and per-file SOURCE_MANIFEST.json checksums. It includes
Python/frontend source and lockfile, configs, scripts, tests, notebook and docs;
excludes outputs, model weights, datasets, environments, caches and .idea/.
verification-summary.json records the final artifact hash and check outcomes.

The final preset review also covers an equation left blank after user
declarations or initial/domain information were entered. A regression verifies
that insertion cannot silently discard that content. Sixteen frontend tests,
TypeScript/build and lint passed afterward; logs are frontend-*-final.log. The
five-browser report predates only this additional conflict predicate and the
final scalar-vector power rejection; affected unit/parser checks were rerun.

## Multi-specialist implementation and recovery — final source

Current source executes two distinct physical families: transient periodic heat
(1D–3D) and static periodic variable-permeability Darcy (2D/3D). Other equation
presets remain editable/parsed without physical execution adapters. See
multispecialist.md, darcy.md and capabilities.md. The original reference-free
and chemistry/steering goals remain unchanged.

### Saved execution inspected and reused

Before recovery, tests/numerics/test_darcy.py passed six independent numerical
checks **before training**: constant/variable manufactured solutions on8/16/32
cells per axis in2D/true3D, second-order convergence, harmonic flux balance,
periodic compatibility/gauge, positivity/failure handling, separate log-a/u
transforms and finite-difference agreement of physical residual gradients.
The complete saved python-tests.log reports **87 passed in40.05s**. It includes
real API workers and both families' reference-isolation cases, not mocked
numerical jobs.

Saved outputs/multispecialist-20261021/cpu-2d has24 instances (16/4/4),8²,
20 real optimizer updates; cpu-3d has12 (8/2/2),true8³,12 updates. Histories,
checkpoint.pt/inference.pt, manifests and three-mode reconstruction/metrics
exist. Last validation losses are1.226635 and1.490055 respectively. Midpoint
interruption/resume completed in both. These accepted training runs were **not
repeated during recovery**. Metrics/poor pressure errors are in experiments.md.
No scientific accuracy, posterior calibration or identifiability is inferred.
The Darcy conventional inverse baseline remains explicitly unfinished.

Both saved MPS preflights completed before recovery, including the one still
running at the original interruption. mps-2d/preflight.json records32²,
batch8,width32/four blocks, two updates and7.11s; mps-3d/preflight.json records
true16³,batch2,width32/four blocks, two updates and5.00s. Each used12 instances
split8/2/2, real float32 Metal training, save/resume, observation/PDE guidance,
export and CPU reload. Runtime is macOS15.2 arm64,Python3.12.3,PyTorch2.5.1,
NumPy2.1.3. Recovery inspected these files and checkpoint progress, and did not
repeat preflight training. Full larger datasets/training were not launched.
CUDA is unavailable; **CUDA and actual Colab execution remain UNTESTED**.

### Checks actually executed during recovery

All paths below are under outputs/multispecialist-20261021/ unless stated.

| Check | Actual result | Saved evidence |
|---|---|---|
| Final complete Python suite, `.venv/bin/python -m pytest -q` | **92 passed,32.23s** | recovery-python-tests-final.log |
| Additional heat-only cost-probe guard regression | **1 passed**; Darcy rejected rather than estimated as heat; heat still works | recovery-cost-probe-test.log |
| Notebook after profile-comment update | Structure, code compile and actual archive/cache/path bootstrap passed | recovery-notebook-tests.log |
| React/Vitest with Homebrew Node22.23.3/npm10.9.9 | **16 passed** | recovery-frontend-tests-final.log |
| `npm run typecheck`, `npm run build` | Passed | recovery-typecheck-final.log; recovery-build-final.log |
| `npm run lint` | Zero errors; three pre-existing shadcn fast-refresh warnings | recovery-lint-final.log |
| Actual Chromium on final UI/API source | **8 passed,0skipped,0flaky,51.10s** | recovery-browser-audited-final.json |
| Existing2D Darcy CLI reconstruct/reload/evaluate | All three modes, no retraining; reproduced prior sample metrics | recovery-cli-2d.log; recovery-evaluate.log; cpu-2d/reconstruction-recovery/ |
| Staged/unstaged whitespace; source archive/hash checks | Passed | recovery-delivery-checks.json; verification-summary.json |

The extra cost-probe guard was added after the92-test complete run; its targeted
regression passed separately. It only rejects calling the existing heat cost
estimator with a Darcy config, and does not change either physical pipeline.
Notebook comments were updated after that suite; its affected bootstrap test
was rerun. No other numerical/training change followed final verification.
Build retains the nonfatal MathLive/application >500kB chunk warning.

Actual browser commands, from frontend with the API/frontend running:
`PLAYWRIGHT_BROWSERS_PATH=../cache/playwright npm run e2e -- --reporter=json`.
This uses the documented existing Chromium cache; no mocked browser/server
replaced numerical execution. The eight workflows exercise actual heat2D,
trueheat3D and1D jobs; Darcy2D/true3D joint reconstructions; automatic family
selection; supported residual=0 rearrangement; editing heat into Darcy;
incompatible manual heat override; dimension/boundary revalidation;
unsupported terms and stale/invalid source blocking; static a/u selection and
orthogonal slices without physical-time sliders; real CSV roundtrip; evaluation
reference attach/remove; history/refresh, cancellation, real MathLive keyboard
editing and430px layout. Routing UI displays the canonical equation, selected
specialist/content ID, rationale and checkpoint exclusions. API tests reject
wrong-family jobs before queuing and reparse the full current request.
Both families' reference-isolation regressions run actual sample paths; truth
changes/attachment/removal cannot alter generated samples. Darcy tests cover
partial observations of either physical channel and all three modes.

New browser screenshots include browser-darcy-2d.png and browser-darcy-3d.png;
the3D capture was visually inspected. Screenshots for all rerun historical
workflows now write only this task root. Earlier accepted scientific results,
weights, datasets, QA images and archives were not overwritten during recovery.

### Failures found and resolved

The prior scripts/dev.py launch failed automatic approval review because the
reviewer itself reached a usage limit. It was not executed and the rejection
explicitly did not judge the action unsafe. During this recovery direct local
API/frontend startup reached sandbox bind restrictions at127.0.0.1:8000/5173.
Fresh reviewed host launches were permitted; no approval bypass or permission
change was used. The user's Homebrew Node22.23.3 replaced the earlier private
runtime PATH. No external service was launched.

The first browser attempt found no Chromium in the default home cache: all
eight failed **before opening a browser**, so they are not execution evidence.
The installed repository cache resolved that setup. The next actual run passed
seven; an old unsupported-template test selected Poisson but still typed an
undeclared Darcy a symbol. Correcting it to -Delta u=f+u restored the intended
parsed-but-unsupported check. A subsequent8/8 run passed in37.88s; after the
final selected-specialist display/strict-validation audit, the final8/8 run
passed in51.10s. Reports preserve these failures and outcomes separately.

An early2D static observation export used a zero internal storage index in
physical_times metadata. A narrowly validated backward-compatible loader now
accepts exactly that old encoding, while canonical static metadata is null.
Arbitrary nonzero times still fail; no sensor positions or accepted files were
changed. A regression and actual reconstruction of that old artifact passed.

Discovery originally cached IDs when a trainer overwrote inference.pt. It now
retires obsolete IDs, and workers check selected identity before/after load.
Regression tests verify stale selection and queued replacement rejection.
Heat mathematical requests now reject unsupported known_fields rather than
silently dropping them. Darcy rejects attempts to relabel its FV evaluator as
spectral. These are included in the92-test final suite. No tolerance was loosened.

### Delivery and Git state

Source package: outputs/multispecialist-20261021/pdeflow-multispecialist-recovery-source.zip.
It contains package/frontend source and npm lockfile, project/dependency metadata,
configs, notebook, tests, scripts and docs. SOURCE_PACKAGE.md explains generating
or transferring omitted data/checkpoints; SOURCE_MANIFEST.json hashes every
payload file. It excludes outputs, data/weights, environments, caches, unrelated
.idea/ and references. Archive integrity and every manifest checksum were checked.

The branch is main, with **no HEAD/commit and no remote**. The original90 staged
additions remain in the index; existing unstaged/untracked implementation and
new recovery edits remain available. Index SHA256 remains
0ea22de80d407ff705f4598473fdc26f786d3af7be731862bc03aadbf2053fb3.
No commit/staging/reset/push or Git-permission change was attempted. Existing
unrelated .idea/ content was left alone. Machine-readable delivery/provenance is
in verification-summary.json. See multispecialist.md for current commands,
scientific limits and the next A+B reversible C chemistry specialist.

After verification, the task-owned API and Vite processes shut down cleanly;
ports8000/5173 had no listeners. The commands in multispecialist.md restart
them. An auxiliary CLI-help subprocess in the sandbox aborted with SIGABRT;
the same argparse help/flags were verified directly in-process, alongside
successful real CLI reconstruction/evaluation and approved host worker/browser
execution. This auxiliary check did not replace integration tests.

## Online providers, Darcy inverse and chemistry — 22 October 2026

This section records new execution during the current implementation, not a
restatement of earlier test totals. New artifacts are exclusively under
outputs/online-chemistry-20261021/. Previous accepted heat/Darcy runs and the
heat0/100/500 study were inspected and preserved, not retrained. The older
sections above describe their historical source/state; Darcy inverse and
chemistry are now implemented at the levels stated here.

Host: macOS15.2 arm64, Python3.12.3, PyTorch2.5.1, NumPy2.1.3; user's Homebrew
Node22.23.3/npm10.9.9. Normal sandbox execution hides MPS and denies loopback
binds; reviewed host execution permitted actual MPS, API and browser runs.
No approval bypass or permission changes were used. CUDA is unavailable and
untested; actual Colab execution is untested. No larger3000/1500-update profile
was launched.

| Check actually executed | Result | Evidence relative to new output root |
|---|---|---|
| Full Python suite, `.venv/bin/python -m pytest -q` | 114 passed,73.00s;15 dependency deprecation warnings | python-tests-final.log |
| Notebook cache deletion, persistent evaluation restoration, online resume | 2 passed,1.92s after full suite; only test code strengthened | notebook-cache-resume-tests.log |
| Frontend `npm test` | 16 passed; mocked component tests, not numerical integration | frontend-tests-final.log |
| `npm run typecheck`; `npm run build` | Passed; build retains nonfatal >500kB chunk warning | frontend-typecheck-final.log; frontend-build-final.log |
| `npm run lint` | Zero errors;3 existing generated UI fast-refresh warnings | frontend-lint-final.log |
| Actual Chromium/API/worker numerical workflows | 10 passed,0 skipped/flaky,83.52s | browser-final.json; final-browser/ screenshots/jobs/data |
| CPU heat/Darcy common providers | Offline/online/hybrid actual update/save/reload; online true3D reconstructions | throughput-final-heat/; throughput-final-darcy/; online-heat-3d/; online-darcy-3d/ |
| CPU chemistry2D8² / true3D8³ | 8 /4 actual updates, midpoint resume, exports and all3 guidance modes | chemistry-2d/; chemistry-3d/ |
| Actual MPS online heat8² / Darcy8² | 2 updates each, resume, guidance and CPU reload | mps-online-heat/preflight.json; mps-online-darcy/preflight.json |
| Actual MPS chemistry16² / true8³ | 2 updates each, resume, guidance and CPU reload | mps-chemistry-2d/preflight.json; mps-chemistry-3d/preflight.json |
| Current config/export/CLI audit | 40 configs valid without requiring every optional device;4 chemistry exports match final package source hash | delivery-checks.json |

The final browser command was run from frontend with local servers running:
`PLAYWRIGHT_BROWSERS_PATH=../cache/playwright npm run e2e -- --reporter=json`.
The JSON reporter files contain npm's textual preamble before the JSON object.
All ten workflows use real checkpoints and numerical workers. They cover the
previous heat1D/2D/3D and Darcy2D/3D routes plus chemistry2D/3D, one-species CSV
observations, known-parameter edits, current-equation routing, invalid kinetic
law rejection, unsupported drafts, wrong-family overrides, cancellation,
request export/import, reference attach/remove, field/time/orthogonal slices,
and data-mode config/CLI export without launching training. API compatibility
is independently tested. CLI/API reference-isolation regressions cover all
three families; the Darcy API regression also checks baseline arrays are
unchanged by attaching/removing truth. The true3D chemistry final screenshot
was visually inspected, including species c, physical time, y-z slice, separate
per-species errors and correctly labelled a+c/b+c invariants.

Provider tests verify disjoint deterministic namespaces, immutable held-outs
across modes, calibration-only frozen transforms, exact CPU online/hybrid
consumed-ID and model-weight resume, actual SIGINT after an optimizer update,
bounded replay, bounded solve failure propagation, stale manifest rejection,
and unsupported worker rejection. No parallel sequence guarantee is claimed.
Numerical targets always come from reference solvers, never model predictions.
Cost probes charge offline generation, online solves and calibration; their
measured times, unique problems, presentations, reuse and zero observed
failures are in timing-summary.json and online-training.md.

Darcy inverse directional finite differences pass in2D and true3D, with actual
harmonic CG solves/adjoints, positivity and zero-mean gauge checks. The bounded
study used one0/20/80-update path (0/40/160 unique consumed train problems),
two validation/two test physical problems, pressure-only/both-field sensors,
densities4/12 and noise.005. Only validation selected guidance/ridge. No test
budget winner was selected. Conventional inverse mean unobserved relative L2
(a,log(a),u) was(.16895,.76222,.09630);80-update observation+PDE gave
(.27780,1.25194,.50603). Permeability test quality was non-monotonic. Six of
eight test inverses hit24 updates;1 gradient tolerance/1 objective tolerance.
All299 forward/299 adjoint test calls passed CG checks, with zero solver
failures. Capped optimization is not convergence. See darcy-study.md and
darcy-study/{study.json,stopping-audit.json,budget-comparison.png} for all cases,
metric definitions, tuning, costs and honest negative conclusions.

Chemistry tests compare reaction to independent RK4, diffusion to analytic
solutions, homogeneous equilibria, zero-rate/stiff limits, spatial/time
refinement (second-order temporal ratio>3.7), true3D, positivity/invariants,
zero-safe transforms and physical residual/observation gradients. Reference
Strang/uniformization completion does not certify a global error tolerance.
Audited reference calls:24 in the2D smoke,16 in3D, including calibration and
fixed evaluation. Maximum integral drift of a+c/b+c was1.34337e-14/2.02061e-14;
minimum concentrations .062346/.059507, zero clipping and zero failed attempts.
For the saved learned observation+PDE reconstructions, unobserved relative L2
was .37637/.37019 and a+c drift .05587/.01095. Positive square decoding gives
zero negativity but does not conserve reaction invariants. The legacy metric
mass_max_drift measures individual species integral change here; it is not a
chemical invariant. UI displays total species sum change and actual a+c/b+c
drift separately. Sqrt negative-root folding and near-zero guidance limits
are documented, with no hidden concentration projection.

Initial failing checks are retained: provider tuple/list specification mismatch
(canonical JSON fixed); inverse einsum spatial contraction (explicit sum fixed);
chemistry test collection typo; old heat-only parser expectation after adding
chemistry (updated to real recognition). No tolerance was loosened. Initial
successful112-test run preceded added notebook and baseline-isolation coverage;
114 is the final complete suite. Final UI diagnostic labels/export defaults
were followed by frontend/build/lint and the final10-workflow real browser run.
The final notebook change strengthened a test only, followed by its2-test run.

Source package is pdeflow-online-chemistry-source.zip in this output root.
SOURCE_PACKAGE.md explains omitted artifacts and regeneration; manifest hashes,
ZIP integrity, expected sources/lockfile/configs/notebook/tests/docs and output/
environment/cache/secret-file exclusions were checked. Provider/study phase
source archives retain their earlier source identities; all four final
chemistry CPU/MPS exports match current package source. verification-summary.json
contains machine-readable outcomes; delivery-checks.json records configs, CLI
help, exports and Git entries. Source/package checks do not rerun training.

Git remains main with no HEAD/commit and no remote;90 staged additions remain,
alongside unstaged/untracked implementation. No add/commit/reset/push was
attempted. Actual index byte hash is
9c5f4385c42492042ad13d7a58854995f74e30e47ed9ba06ae7f4b65daead9c8,
which differs from the older recovery report's hash; bytewise identity against
that historical value is therefore not claimed. No Git permission was changed
and unrelated .idea/ was left alone. Task-owned API53011 and Vite53099 stopped
through their original sessions; ports8000/5173 had no listeners afterward.
See online-chemistry-handoff.md for exact restart, training and transfer commands.

## Process diagram, feedback and physical heat refinement — 25 October 2026

Actual new evidence is outputs/process-20261025/. The final full host Python
suite passed126 tests (79.63s); after the final all-attempt cost/trace correction,
21 affected process/sampler/CLI/API tests passed. Additional deterministic budget/
cost coverage passed12 unit tests. Frontend18 tests, TypeScript/build and ESLint
passed (3 existing generated UI warnings, nonfatal large bundle warning).
The actual complete browser suite passed12/12 in74.44s; two affected real
feedback/refinement workflows passed after the final cost correction. These
are real numerical workers/checkpoints, not mocked browser execution.

Heat8², true3D Darcy8³ and chemistry8³ ran the new reference-free controller.
Each bounded four-step example consumed14 model calls and returned explicit
FINISH_BUDGET_EXHAUSTED with unconfigured mandatory thresholds. An actual new
heat16² checkpoint performed2 updates and enabled8²→16² fresh inference,
unchanged physical measurements and compatibility revalidation. The final
refinement consumed28 calls/eight guided steps/two attempts and also stopped
budget-exhausted. Its higher residual is reported; no accuracy improvement is
claimed. Full phase/source, formula and failure details are in
process-verification.md; every diagram box/arrow is mapped in process-audit.md.

The new process/refinement execution is CPU-only. Earlier MPS evidence was
inspected, not rerun/relabelled as new verification. CUDA/actual Colab remain
untested. Full request routing, component checks, final-grid metadata,
reference-isolated decisions, CLI/API equality, serialization/reopen, cancellation,
invalid numerics, budgets, guidance placement and unavailable actions are covered.
The resampling/FK branch remains absent; the whole diagram is not complete.
Source package pdeflow-process-source.zip includes source/configs/tests/docs,
not accepted outputs/weights/data or environments. Existing staged/user work and
accepted experiment roots were preserved; no Git mutation was attempted.

## FK implementation and execution — 25 October 2026

Primary new evidence: outputs/fk-20261025/; full derivation fk-mathematics.md and
self-contained commands/results fk-verification.md. This is newly executed
FK work, separate from historical accepted experiments. Final Python suite:
143 passed/16 warnings in86.70s (python-verified.log). Frontend20 tests,
TypeScript/Vite build passed; lint zero errors/three existing UI export warnings.
Actual browser initial15 passed; final affected FK4 passed including rejection
and cancellation. The full final combined suite passed16 real workflows in100.31s, zero skips/
failures/flaky results (browser-verified.json). No mocked frontend run is counted as numerical integration.

Four-seed analytic Gaussian/mixture Cartesian budgets and all methods are saved;
Gaussian FK4096/128 mean/covariance errors.01287/.00810; mixture.03414/.02850.
One validation-selected heat test gives negative unobserved-error results for
FK/IS relative to unguided; all seeds/ancestry/costs retained. Actual MPS model
execution covers heat2D/3D,Darcy2D,chemistry3D stochastic stepping, energy/weights,
resampling, endpoint/final checks/export/CPU reload/cancellation; legacy gradient
controller and8²→16² refinement run separately. Explicit CPUfloat64 weights/RNG
are recorded, not hidden model fallback. CUDA/actual Colab are untested.

Initial MPS float64 transfer and toy summary errors and a browser ambiguous-alert
selector failure were fixed; development logs remain. No large training was run.
Source archive/manifest and current Git state are reported in fk-verification.md
and verification-summary.json. Existing90-entry Git index remains unchanged.

## Local recovery and scientific UI continuation — October 25

The latest workflow/UI record is [workbench-verification.md](workbench-verification.md).
It distinguishes the completed prior local recovery evidence from newly executed
155-test Python verification, actual visual API checks, frontend checks and real
numerical browser journeys. Earlier dated records remain historical evidence.

## Reference experiments and training suites — 26 October 2026

[reference-verification.md](reference-verification.md) records the new implementation
and actual tests, including the initial full regression failures/corrections,
182 passing final Python tests,33 frontend tests, nine real numerical browser journeys,
six-child CPU quickstart, actual SIGINT/resume, two-child MPS probe and CLI/artifact
checks. This is distinct from historical verification above. No long pilot or
scientific training-budget study was repeated. CUDA/actual Colab remain untested.

## Flow workspace, selection and capacity continuation — 26 October 2026

[flow-workspace-verification.md](flow-workspace-verification.md) records the captured
original empty-model-ID request and the implemented contract/timing fixes,204 passing
Python tests,33 frontend tests,12 real current browser journeys, actual CFM toy/
matched capacity measurements, native1024² MPS operations and managed resume/CPU
reload. This is separate from prior execution records. No full high-resolution
pilot, CUDA or actual Colab execution is claimed.


## Catalogue, native512 and training overview — 26 October 2026

[Catalogue verification](catalogue-verification.md) records this continuation: full
Python regression220 passed, followed by affected management/CLI/suite25 and final
storage/inventory15 checks;38 frontend tests, TypeScript/build pass, lint0 errors
with3 existing warnings. Three new and11 retained real browser journeys passed;
final monitoring/storage repeats verified counter and presentation fixes. These
counts are distinct executions, not additive test totals. Actual native512 MPS
backward/update, physical guidance, Heun sampling, managed resume to10 updates
and CPU-remapped forward passed. Ten updates remain partial against10000.

The user's six original local-pilot targets completed their3000/1500 budgets;
coverage and exports were recovered without modifying their frozen suite. The
expanded catalogue has11 targets and5 uncovered default-workspace targets. No
expanded full suite, long native512 training, CUDA or actual Colab execution ran.
Cached inventory/preview mutated no user scientific artifacts. Final source,
Git/index preservation and task-service shutdown: outputs/catalogue-512-20261026/
completion.json. Exact commands: [catalogue handoff](catalogue-handoff.md).

## Optional coordinate heat and source workflow — 27 October2026

[coordinate-verification.md](coordinate-verification.md) records the235-test full
regression followed by56 final affected stage/source/symbolic/storage checks,
42 frontend tests, TypeScript/build and lint (0 errors,3 existing warnings), and
three actual numerical browser journeys including retained fixed-grid inference.
CPU4/4 and MPS2/2 AE/flow smoke stages, independent held-out AE diagnostics,
coarse/native/fine/scattered queries and count/noise/seed comparison were executed.
No full10,000-update stage, large suite, CUDA or actual Colab execution occurred.
These stages and older fixed-grid results are not scientifically matched or ranked.
Exact commands and capabilities: coordinate-specialist.md. Final source/Git/index/
whitespace preservation and task-service shutdown are in the new evidence root's
completion.json. No user scientific output cleanup was applied.


## Integrated training handoff — 27 October 2026

The shared UI/CLI training workflow has57 affected Python checks,46 frontend
checks, a passing TypeScript/build and3 real numerical browser journeys.
Current evidence, failures, preservation and limits:
[training-workflow-verification.md](training-workflow-verification.md).
The final handoff reused those runs; no further training was launched.

## Focused heat playground, 27 October 2026

See [heat playground verification](heat-playground-verification.md) for this
session's focused numerical/API/state/storage tests, actual browser inference,
main/comparison screenshots, preservation hashes and service shutdown record.
Earlier studies remain historical evidence, not newly rerun accuracy evaluations.

## 27 October 2026 — substantial Colab heat workflow

See [colab-verification.md](colab-verification.md) for current-session evidence:
34 affected Python passes with one actual-CUDA skip; two notebook checks including
real ordered execution of all six CPU cells; one retained legacy notebook check;
49 frontend tests; TypeScript/build/lint; and one real browser import/reconstruction
of the notebook export. Exact logs, repaired failures, CPU-only limits,
source/archive inspection and preservation records are listed there. No substantial
training, actual Colab/GPU or mounted Drive execution is claimed. Existing
scientific results are unchanged and these tiny models do not improve accuracy.

## 28 October2026 — continuous heat time and placement

See [continuous-heat-verification.md](continuous-heat-verification.md) for the
55-pass affected Python matrix, focused33/21-pass follow-ups after gate/calibration
corrections (overlapping counts), real six-cell CPU notebook execution, actual
MPS AE2/flow2 and CPU reload,51 frontend checks, TypeScript/build/lint and two real
browser numerical journeys. Screenshots, matched-policy comparison, source package
and preservation/process evidence are linked there. No substantial training,
CUDA/actual Colab execution, scientific accuracy improvement or test-based winner
is claimed. Old snapshot capabilities are preserved and labelled accurately.

## Spatial-profile frontend slice —28 October 2026

See spatial-profiles.md and outputs/spatial-profiles-20261028/ for 13 focused
passing tests, TS/Vite/lint evidence, actual rendered field/reconstruction journeys
and requested screenshots. No training, model architecture or backend changes.
Historical backend/scientific results are reused as history, not rerun or expanded.

## Gradient-updating field reconstruction losses —29 October2026

field-loss-verification.md records focused math/API tests, real2-update runs for
four losses, actual encoder/decoder update dependence on α, exact same-loss resume,
changed-loss rejection, CPU/MPS and sequential local notebook execution,19 frontend
tests, build/lint and real training/comparison browser checks. Logs distinguish
overlapping counts, native/Chromium failures and successful isolated follow-ups.
No substantial training or scientific superiority claim; CUDA/hosted Colab untested.
All prior checkpoints/scientific outputs retained.

## Current joint latent replacement —30 October2026

See joint-latent-verification.md for this session's84 passing affected Python
checks/two CUDA skips,19 frontend tests, real UI training/inference, local CPU
notebook execution, CPU/MPS explicit-noise agreement and preservation evidence.
These are execution/numerical contracts, not scientifically demonstrated accuracy.
Prior generative results above are historical and use incompatible architectures.

## Operator-aware heat/wave v2 —30 October2026

The active shared model now uses A1 state representation → A2 learned graph-conditioned
physical evolution → B initial-state/scalar conditional generation. Saved focused
CPU/MPS numerical/training/API execution, actual browser workflows and local execution
of notebook cells are recorded separately in `operator-wave-verification.md`, with
initial failures and repairs retained. The two-update checkpoints are poor and the
representation fails its constant-baseline gate; no scientific accuracy improvement,
CUDA execution or actual Colab execution is claimed. Final continuation work checks
source packaging, dependency/source compatibility, preservation and service shutdown
without repeating training or bypassing the failed gate.

## Training-equation wiring audit —30 October2026

The focused training page inherited heat-only intent while hiding the family selector.
A visible independent Heat/Wave/joint selector, joint default and configuration-derived
summary now match actual data/stages. Region-balanced alpha=.5 remains actual decoded
A1/A2 training; B remains latent CFM. Fresh tiny CPU stage/alpha-control execution,
real API configuration traces and rendered browser evidence are recorded in
`training-equations-verification.md`. Existing artifacts and accuracy limitations
remain unchanged; no substantial training ran.

## Training performance recovery — 31 October 2026

See [training-performance.md](training-performance.md) and
outputs/performance-20261031/ for exact workload, baseline source, CPU timing and
loss/gradient parity, MPS failure/fix, local notebook execution and source-package
verification. Actual CUDA/Colab timing and FP16 remain unverified; supplied T4
benchmark preserves requested capacity/points/objective and never launches full
training. Existing scientific outputs are covered by before/after hash records.

## Actual A1/A2 neural batching —31 October2026

See [batched-training.md](batched-training.md), outputs/batching-20261031/.
93 affected tests passed,2 CUDA-only skipped; final preparation suite35 passed;
actual final MPS A1/A2/B plus cache-key check2 passed,1 CUDA check skipped.
Counts overlap. Notebook2 tests passed including the actual local six-cell CPU/
IPython path. All8/16/32 smoke-capacity cold/warm benchmark variants completed
at the same effective32, FP32 and immutable dependencies. CPU full-gradient
comparisons cover all four field objectives and A2/B; no actual T4 speed/memory
or CUDA FP16 verification is claimed. Packaging/preservation checks are recorded
separately at completion, with no full training or existing artifact alteration.

## Batch application workflow —31 October2026

See [batch-settings.md](batch-settings.md) and outputs/batch-settings-20261031/:
shared configure/apply API, actual worker metadata, per-stage/remainder tests,
subprocess benchmark isolation, budget/skip/scheduler/resume, real seven-cell local
notebook chain, frontend checks and actual rendered MPS benchmark/apply/training.
CUDA/actual Colab/Drive and1024 memory fit remain unverified. OOM/overflow mocks are
labelled as control tests. Scientific artifacts are covered by before/after hashes.
