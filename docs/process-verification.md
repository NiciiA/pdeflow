# Process verification and runnable handoff — 25 October 2026

Historical process slice: the gradient-feedback loop and a bounded global heat
spatial refinement path execute. The later FK continuation now implements the
resampling branch; see fk-verification.md for its separate mathematics and actual
CPU/MPS/browser execution. FK plus refinement remains unsupported. The detailed box-and-arrow evidence
matrix is process-audit.md. Existing successful online providers/chemistry and
accepted scientific outputs were inspected and preserved. Only one new fine
heat checkpoint was smoke-trained; no prior study or large training was repeated.

## Actual call graphs

Heat browser request: MathLive full 𝒞 → symbolic.analyze / HeatSpecialist
recognition/request_problem → API.resolve (reparse current source, measurement
agreement, exact checkpoint compatibility) → JobManager.compute_job →
sampling.sample → process.run → sampling._sample_once → known D condition /
Euler-or-Heun model proposal → SplitGradientGuidance(ReconstructionEnergy,
PointOperator.apply, HeatPhysics.evaluate on decoded estimated endpoint) →
process.diagnostics.report / ProcessState.intermediate → next interval guidance
choice → actual final decode/report → boundary retry/refinement/finish decision
→ export_result → API result/slice → React ProcessPanel / FieldPlot.

Darcy follows the same call graph through DarcySpecialist, known mean-zero f
map, affine(log a,u) decode with positive a/pressure gauge, static conservative
harmonic face-flux evaluator and point observations of actual generated a/u.
There is no physical-time derivative or initial condition. Conventional BFGS
inverse executes separately after neural inference, without truth; its result
is not substituted into the controller's neural population. Chemistry uses the
same loop with three decoded positive species, actual known D/rates and coupled
reaction residuals. Its invariant checks are a+c/b+c, not a+b+c.

The real CLI calls cli.reconstruct → the identical sampling.sample/process.run
and only then read_reference/export_result. The new actual CLI-versus-API-worker
test produces bitwise-identical fields and identical complete process summaries
for the same resolved heat request; CLI had unrelated truth while API had none.
After refinement, the final-state request view also aligns the fine checkpoint
ID, physical observation indices, symbolic domain and actual last-attempt seed.
original_request retains the complete process input for reproduction; the
final-state view is not a replacement for the original coarse process input.
The API reparses/revalidates independently of React. Fine checkpoint IDs are
server-owned; client filesystem paths are not accepted. CLI --fine-checkpoint is
an explicit local-file interface.

## Evaluation conventions and inference state

The differentiable steering path remains enable_grad through endpoint estimator,
model and physical decode. Correction is -eta*h*clipped_gradient after the full
Euler or Heun interval, once per interval. It is not a continuous drift/FK/exact
posterior rule. Separate no_grad diagnostic calls do not supply gradients.
At s=1 the endpoint estimate is the state; there is no singular conversion.
Zero guidance reproduces baseline samples. Directional/finite-difference tests
cover observation masks/ordering, physics perturbations and affine physical
scaling; Euler/Heun placement is explicitly tested.

Observation H is exact saved-node/channel/time gathering, in row order. Masks
mark missing values; missing entries remain null in reported residuals rather
than observed zeros. All-false problem masks, duplicates/conflicts, out-of-domain
or off-grid positions and unsaved times are rejected. Scalar fields only execute;
vector components, spatial/time interpolation and correlated noise are absent.
For independent Gaussian measurements, report raw e=H(U)-y and e/sigma for
positive sigma, otherwise null. Guidance/policy uses rho=max(sigma,tolerance)>0
and .5*mean_measured((e/rho)^2) per candidate. This is explicitly tempered,
count-normalized misfit, not an untempered exact likelihood. Per-field RMSE,
maximum absolute mismatch, measured/missing counts and scales are retained.

Physical component RMSE is the **worst candidate** per equation, using normalized
trapezoid physical-time and uniform spatial-volume quadrature. Transient
residuals cover interior saved times; Darcy is static. Do not compare differently
scaled equations through one weighted average. Separate mandatory reports cover
observed fields, each PDE equation, relevant conservation and gauge; positivity
violation count must equal0 for Darcy/chemistry. Each entry has raw value, units,
reduction, component/count, threshold, finite flag and status. Periodic topology
is structural, not a measured independent boundary pass; unknown initial data
is not invented as zero. Those residuals are marked not_applicable.

Policy reports identify estimated_endpoint, final_field/final_corrected_field,
or incomplete_generation_field. Actual final verification always evaluates the
returned decode, not the last intermediate estimate. It shares the FD/flux
operator with steering; existing final metrics additionally evaluate spectral
space for transient families. Neither is a continuum-accuracy certificate.
Reference errors and empirical spread are separate from physical/observation
consistency and never enter stopping, retries, refinement or selection.

SamplingState retains model tensor, generation coordinate/step, source RNG and
latest estimated physical endpoint. Returned NPZ also retains final_model_state;
JSON inference_state records checkpoint, specialist, full physical grid/times,
particle IDs, seed, RNG state, model-state shape/hash. Global budgets, recent
reports and actions are in metrics.process. Intermediate history retains compact
observation summaries; final verification retains full raw/masked residuals.
No reference arrays, errors or lookup are part of controller state.

## Legal decisions and limits

| Phase / condition | Actual action |
|---|---|
| Intermediate unresolved endpoint, guided mode | CONTINUE_WITH_CONFIGURED_GUIDANCE for subsequent intervals |
| Passing endpoint estimates for patience checks / unguided mode | CONTINUE_SAMPLING; never early final success |
| New endpoint mismatch after passing estimates | Restore configured guidance immediately |
| Completed path; all mandatory bounds configured/finite/passing | FINISH_CRITERIA_MET after actual final report |
| Completed mismatch with compatible fine model and remaining level/attempt/grid budget | REFINE; fresh fine Gaussian source with seed+1 |
| Completed mismatch/stagnation, remaining attempts | RETRY_WITH_DEFINED_POLICY; fresh source seed+1 modulo2^63 |
| No legal remaining budget / missing required thresholds | FINISH_BUDGET_EXHAUSTED; unresolved mismatch listed |
| Nonfinite state/energy/gradient/final evaluator | FAIL_INVALID_NUMERICS; last finite completed state if available, explicitly labelled |
| Cancellation at sampler boundary / API worker termination | STOP_CANCELLED; no invented final certificate |
| Requested FK or unsupported physical refinement | RESAMPLE_UNAVAILABLE / REFINE_UNAVAILABLE with reason |

Retries/refinement cannot reset a partial generation path. The last attempted
population is returned without truth/best-score cherry-picking. These heuristics
still change the distribution; it is not an unbiased posterior. Stagnation is
lack of relative improvement of the largest configured normalized component
score over patience checks. It changes the boundary retry reason, not the ODE's
physical/generation clocks. Contradictory data are unresolved (duplicates are
rejected); a failed run does not prove PDE nonexistence.

Default controller limits: check_interval2,128 network evaluations,32 sampling
steps,32 guidance steps,1 total attempt,1 permitted refinement level,32768
spatial grid points,16 particles, no wall limit, patience2 and relative
stagnation tolerance.001. The inference particle count remains separately set.
Limits are global across attempts/grids; a fine reconstruction consumes an
attempt. Only zero/one refinement levels execute. Explicit optional wall time
is checked at model/step boundaries, not an interruptible hard real-time bound.
Tests leave it disabled for deterministic decisions.

Thresholds are per-name upper bounds, finite and nonnegative. For heat the
mandatory configurable names are observation.u.standardized_rmse, pde.u.rmse
and conservation.u.max_drift. Darcy adds gauge.u.abs_mean and each observed
field's observation component; chemistry has pde.a/b/c.rmse and both
conservation.a_plus_c/b_plus_c.max_drift. All mandatory diagnostics must pass;
unknown names are rejected and absent thresholds remain unconfigured. Optional
configured guards must pass too. No universal physical-accuracy thresholds were
invented. Ordinary workbench defaults deliberately have no acceptance thresholds.

## Executed refinement and its scientific meaning

A new CPU heat16² width8/one-block checkpoint performed **two real updates**, with
resume/save/export in heat-fine-smoke/. Its training/calibration/evaluation were
numerical heat references, not coarse generated fields. The config is
configs/experiments/process_heat_fine_smoke.json. fine-training-source.zip
preserves that exact training-phase package source; the final source differs
by all-attempt cost/trace aggregation and aligned final-state API metadata,
with compatible model/layout inference metadata.

refinement.refine_request requires same heat family, periodic nodal domain,
known D and physical times, exactly2× grid shape, supported layout/formulation
and grid budget. It calls validate_compatibility again. Observation indices ×2
retain exactly the same physical positions, y, masks, noise and ordering. Known
scalar D transfers unchanged. Physical-time transfer is identity; finer-time
execution is unavailable. Diagnostic restriction is every-other-node injection;
prolongation is periodic multilinear interpolation, unused by generation.

Fine generation is a fresh compatible inference from Gaussian source seed+1,
conditioned on the same actual measurements/known physics. Coarse predictions
are not pseudo-observations, Gaussian source or an untrained conditional bridge.
All calls/costs are attributed across both attempts. Coarse truth is not
interpolated into a claimed fine reference. API reference attachment validates
the final layout and rejects coarse truth after refinement.

The final direct run changed8²→16² with preserved measurements,8 total guided
steps and28 network calls. It stopped FINISH_BUDGET_EXHAUSTED, not success.
Coarse component PDE RMSE was15.3072; the fresh fine population's was48.3968.
The fields/source/model differ, so this is **not** a convergence comparison or
an accuracy improvement. The browser also executed a real16² result and showed
its final grid. True3D controller execution is tested, but actual3D physical
refinement is not claimed. Darcy/chemistry, temporal and local AMR refinement
remain unavailable.

## Actual checks and saved evidence

All paths below are relative to outputs/process-20261025/.

| Actual execution | Result | Saved evidence |
|---|---|---|
| Full host Python suite |126 passed,79.63s;15 dependency deprecation warnings | python-final.log |
| After final all-attempt cost/trace correction: process/sampler/CLI/API tests |21 passed,26.39s | cost-audit-tests.log |
| Final deterministic budgets/cost unit coverage |12 passed | budget-cost-tests-final.log |
| Frontend unit tests |18 passed | frontend-tests-final.log |
| TypeScript + production build |Passed; existing >500kB bundle warning | frontend-build-final.log |
| ESLint |0 errors;3 existing generated UI warnings | frontend-lint-final.log |
| Complete actual Chromium/API/worker suite |12 passed,0 skipped/flaky,74.44s | browser-final.json |
| After cost/trace correction: two actual feedback/refinement browser workflows |2 passed | browser-cost-final.json |
| After final-state metadata alignment: actual CLI/API/reference isolation |4 passed,26.14s | final-api-tests.log |
| Final physical refinement browser metadata assertions |1 passed | browser-metadata-final.json |
| Actual bounded three-family process examples |Heat8²; Darcy/chemistry true8³; no references | final/executed-summary.json; final/*-process/ |
| Actual heat global refinement and CLI comparison |8²→16², unchanged measurements,28 model calls | final/heat-refinement/; final/cli-heat-refinement/ |

New execution uses CPU on the saved macOS15.2 arm64/Python3.12.3/PyTorch2.5.1
host, Homebrew Node22.23.3/npm10.9.9. Prior CPU/MPS chemistry/online evidence was
inspected, not repeated as a new MPS claim. This controller/refinement task was
not executed on MPS/CUDA or actual Colab; those are not newly verified backends.

The initial sandbox Python run could not start numerical subprocesses (OpenMP
SHM error); reviewed host runs passed. One new test originally supplied invalid
zero permeability as a constructed unrelated evaluation reference, causing log(0) export;
it now uses an admissible unrelated reference. No physical tolerance was loosened.
A JSX brace typo was caught by formatting/build and fixed. The first full
browser run passed9/11: the new process test tried resubmission from history
without creating a draft (correctly rejected); diagnostic text caused mobile
overflow (fixed). Three affected workflows then passed, followed by the final
12/12 suite. Old successful output roots/screenshots were not overwritten.

Reference isolation is tested for all three specialists' final fields and
controller summaries. The actual CLI/API comparison additionally checks the
entire serialized process under unrelated truth versus no truth. Reopen tests
recover the same actions; true3D, unsupported actions, cancellation, masks,
noise-free observations, gradients, numerical failures, all budget types used
by the policy and deterministic same-seed traces are covered. The criteria-met
browser test uses deliberately permissive1e8 bounds on every mandatory heat
component to test the transition, **not meaningful scientific acceptance**.

Example final trace (heat, four steps):
RESAMPLE_UNAVAILABLE → three CONTINUE_WITH_CONFIGURED_GUIDANCE checks →
REFINE_UNAVAILABLE (no remaining attempt budget) → FINISH_BUDGET_EXHAUSTED.
It consumed14 model calls/4 guidance steps. Darcy and chemistry used the same
bounded sequence; chemistry retained separate a/b/c residuals. The heat fine
trace has REFINE between the two complete paths and ends budget-exhausted.
Final guidance_trace identifies attempt, physical grid, exact generation
position, gradient norm, pre-correction energy and update placement. Costs now
sum all attempts, with attempt_costs and process_seconds_total distinguishing
sampling from policy/transfer overhead. Preloaded checkpoint I/O and conventional
baseline work are separate costs.

## Exact reproduction commands

From the repository root, using its installed Python environment and user Node
on PATH. Choose fresh output roots. These commands use saved compatible artifacts:

```bash
.venv/bin/python -m pdeflow.cli reconstruct \
  --config outputs/process-20261025/final/heat-config.json --device cpu \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --problem outputs/accepted_cpu_2d/data/test_problem.json \
  --observations outputs/accepted_cpu_2d/data/test_observations.json \
  --output outputs/my-heat-process
.venv/bin/python -m pdeflow.cli evaluate --result outputs/my-heat-process/observation_pde

.venv/bin/python -m pdeflow.cli reconstruct \
  --config outputs/process-20261025/final/heat-refinement-config.json --device cpu \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --fine-checkpoint outputs/process-20261025/heat-fine-smoke/training/inference.pt \
  --problem outputs/accepted_cpu_2d/data/test_problem.json \
  --observations outputs/accepted_cpu_2d/data/test_observations.json \
  --output outputs/my-heat-refinement
```

For Darcy/chemistry use final/darcy-config.json or final/chemistry-config.json,
the corresponding existing multispecialist-20261021/cpu-3d/training/inference.pt
or online-chemistry-20261021/chemistry-3d/training/inference.pt, and their inputs
under data/test_{problem,observations}.json (Darcy) or
evaluation/test/0/{problem,observations}.json (chemistry). Input paths and flags
were checked; final/ contains actual executions, not only proposed commands.

For an extracted source archive without these omitted artifacts, create a
fresh real bounded run instead:

```bash
.venv/bin/python -m pdeflow.cli smoke --config configs/experiments/process_heat_2d.json --output outputs/my-process-heat
.venv/bin/python -m pdeflow.cli smoke --config configs/experiments/process_darcy_2d.json --output outputs/my-process-darcy
.venv/bin/python -m pdeflow.cli smoke --config configs/experiments/process_chemistry_2d.json --output outputs/my-process-chemistry
.venv/bin/python -m pdeflow.cli smoke --config configs/experiments/process_heat_fine_smoke.json --output outputs/my-fine-heat
```

Workbench terminal1:

```bash
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training:outputs/workbench-20261021/cpu-1d-regression/training:outputs/multispecialist-20261021/cpu-2d/training:outputs/multispecialist-20261021/cpu-3d/training:outputs/online-chemistry-20261021/chemistry-2d/training:outputs/online-chemistry-20261021/chemistry-3d/training:outputs/process-20261025/heat-fine-smoke/training"
export PDEFLOW_JOB_ROOT=outputs/my-process-workbench/jobs
export PDEFLOW_DATA_ROOT=outputs/my-process-workbench/data
.venv/bin/python -m pdeflow.workbench.api
```

Terminal2: `cd frontend` then `npm run dev -- --strictPort`.
Open http://127.0.0.1:5173 (API8000). Insert a supported equation, generate/import
observations, configure and validate, then run. Feedback is enabled by default.
Advanced process JSON can set bounds and fine_model_id (registered16² heat
content ID from /api/models), refine:true and max_attempts:2. It shows final grid,
component checks/decisions and stop reason. Unsupported FK has an explicit
explanation. Default absence of thresholds deliberately yields budget exhaustion.
Stop both servers with Ctrl-C. Verification-owned servers are shut down afterward.

Tests: `.venv/bin/python -m pytest -q`; frontend `npm test`, `npm run build`,
`npm run lint`; with API/frontend running, from frontend:
`PLAYWRIGHT_BROWSERS_PATH=../cache/playwright npm run e2e -- --reporter=json`.
The existing local Chromium cache was used. npm's text preamble precedes JSON
in saved browser reports.

## Remaining work before the entire diagram is complete

FK needs an explicitly derived compatible base stochastic process, justified
marginal correction if noise is introduced, incremental/telescoping potentials,
stable log weights/normalization, ancestry/RNG/ESS/diversity and resampling; first
verify a tractable tilted target before PDE integration and legal action timing.
None of this is replaced by independent retries or final best-sample selection.

Other gaps: off-grid/time interpolation, correlated noise, vector/irregular
geometry execution, temporal/other-family refinement and scientific validation
of adaptive thresholds/selection. The next research slice should compare this
controller and fresh fine inference against fixed-budget baselines over repeated
seeds and more independent held-out problems with finer physical verification.
No low residual, fine plot or permissive criteria-met test is an accuracy claim.

Source archive: outputs/process-20261025/pdeflow-process-source.zip. It contains
source/frontend lockfile/configs/tests/scripts/notebook/docs, excludes accepted
outputs/weights/data/caches/environments/personal files and has a checksummed
manifest. SOURCE_PACKAGE.md explains regeneration/transfer. Git remains main,
90 staged additions plus unstaged/untracked work, no commit/remote; no Git
mutation was attempted. Machine-readable final evidence is verification-summary.json
under the new output root.

## Subsequent FK/MPS verification

The former CPU-only/new-FK-absent status is superseded by fk-verification.md.
Actual MPS runs execute gradient controller, heat8²→16² fresh refinement, FK
Euler–Maruyama/energies/systematic resampling/final reports/export/CPU reload,
true3D and cancelled incomplete state. FK prescribed actions do not share the
gradient controller's arbitrary guidance/retry decisions. Full FK tests and
negative heat comparison reside in outputs/fk-20261025/. Older process evidence
and accepted outputs remain in their existing locations.
