# Catalogue, native 512² and training overview verification

Evidence root: `outputs/catalogue-512-20261026/`. Original accepted outputs,
user-trained runs, checkpoints, staged changes and working interfaces were kept.
No full training suite, 10000-update native run, CUDA or actual Colab execution
was launched. No user scientific artifact was deleted or quarantined.

## Recovered compatibility failure

The original local-pilot suite 20261026T171753-local-pilot-281503ff actually
completed its six original children: 3000 updates for 2D heat/Darcy/chemistry and
1500 for their 3D targets. Exports exist and load. It never contained the later
medium/large/512²/1024² targets. The chemistry export is valid at 16²/T4; a usual
8² editor draft is not that layout. Exact original failed chemistry draft was not
saved among jobs; the open browser contained a heat draft. We therefore reproduce
layout/kinetic mechanisms rather than claim recovery of the exact lost draft.

`compatibility-report.json` records requests, candidates, expected/actual values,
precise mismatches and remedies. Bounded chemistry polynomial normalization accepts
supported rearrangements, commutation and explicit substitution of r=k_f*a*b-k_r*c;
different kinetics remain unsupported. Names a,b,c and declared rate/diffusion
parameters remain fixed. Wrong-family aggregate errors no longer obscure an
otherwise matching family's layout problem. Routing uses the current full request
and prefers actual optimizer updates, then completed-budget/export provenance;
it never reads truth or promises higher quality from more updates.

## Checks and source stages

| Check | Actual result and saved evidence |
|---|---|
| Full Python regression | **220 passed**,149.76s, `python-release.log`. Includes retained numerics/CFM/FK/reference isolation/controller/refinement/provider/training plus new catalogue/recognition/storage/analytical/budget checks. |
| Final affected management/CLI/suite checks | **25 passed** in `python-final-targeted.log`; preceding CLI/management26 and storage/API18 runs are also saved. These followed final CLI presentation, target-identity and telemetry fixes. Do not sum repeated tests into a fictitious larger suite. |
| Final inventory/cache ownership checks | **15 passed**, `python-storage-cli-verified.log`; covers pin normalization, active/pinned/reference protection, symlink/traversal containment, exact reviewed-plan mutation only in fixtures, compact CLI views and explicit ownership registration. |
| Frontend | **38 passed**, `frontend-delivered.log`; includes actual preparing-state empty telemetry and stale-resume counter regressions. |
| TypeScript / build / lint | `build-delivered.log`, `lint-delivered.log`: pass; zero lint errors, three existing component-export warnings. Existing MathLive main-bundle size warning remains. |
| New real browser journeys | **3 passed**,34.9s, `browser-verified.log`: chemistry mismatch/full plan/history/storage preview; actual native512 analytic solve→measurements→inverse→automatic comparison/generation; actual CLI owner survives closing browser and completes resume. |
| Retained real browser journeys | **11 passed**,~1.2min, `browser-regression.log`: heat2D,Darcy3D,chemistry2D, actual FK ancestry, slow parsing/registry, cancellation/clear/keep, true native1024 reference display, stale/explicit selection and reference-free inverse. |
| Final monitoring/storage browser checks | `browser-monitor-final.log` verifies the monotone-counter fix with two additional existing-run updates8→10; `browser-storage-delivered.log` checks final timestamps/suite/storage fields without training. These repeat existing journeys, not extra independent numerical experiments. |
| Native512 MPS | `native-512-mps/probe.json`: actual native reference, one CFM backward/Adam update, finite physical endpoint guidance gradient and two production Heun steps, one particle,T3. Returned shape[1,1,1,3,512,512]. |
| Managed MPS resume | `native-512-start.log`/`native-512-resume.log`:1→2. Monitoring run `monitor-cli-eight.log`/`monitor-browser-eight.json`:7→8; final `monitor-cli-final.log`/`monitor-browser.json`:8→10, restored optimizer/provider/calibration. These are bounded checks, not full target completion. |
| CPU portability | `cpu-portability-final.json`: MPS-trained native512 export remapped to CPU, actual finite forward shape[1,3,512,512], original export hash unchanged. No training data needed. |
| CLI plans | Full catalogue11 targets and missing-only5 default-workspace targets; MPS512 dry-run, compact catalogue/coverage/storage and preview-only cleanup commands executed. Plans do not start jobs. |
| Storage | Cached stat inventory, explicit refresh, logical sizes, actual birth time when available, updated time, run/suite relations and protection. Real plan has zero eligible caches; no real apply. Recognized stale caches need explicit producer/user ownership. |

Source hashes in individual run/probe/report manifests identify execution stages.
The full220-test and final MPS10-update stage used source450b26b0e2ae02785432638eb4358c9f15111a5c7e2321ec6f3a539bb7397a04.
Subsequent edits concern compact CLI output, completion identity ignoring inference
controls, pin/cleanup serialization and inventory metadata; affected checks above
were rerun. Numerical CFM/reference/residual/sampling code was not changed by those
final management edits. `completion.json` records the final source hash, final Git
checks, actual backend status and service shutdown. Old inference exports stay
loadable; optimizer resume still requires its exact recorded source. No source
mismatch was bypassed. Full default training budgets were not executed by Codex.

## Native512 resource facts and limits

MacBook arm64,MPS,float32,width8/four-level direct-field U-Net,73,939 parameters,
batch1,T3=[0,.05,.1]. Native reference0.09176s; one CFM update2.81753s (cold startup
included); endpoint physical gradient0.43941s; two-step Heun0.46951s. Process RSS
287,735,808 bytes; MPS driver allocation1,162,264,576 and current30,191,104 bytes
after the probe. Accelerator values are point-in-time, **not peak memory** or full
pilot predictions. No new latent model, tiling or upscaled prediction is involved.

Exact structural dependency propagation gives94 input-axis nodes for output
index256 in one network evaluation (receptive-field.json). It is not global512²
communication; repeated integration can expand dependencies. No trained high-grid
accuracy, calibrated posterior, resolution generalization or large FK population
claim follows. The supplied user-launchable12-step/one-particle inference profile
is schema checked, not a new scientific performance result.

The real browser initially exposed an overlooked default controller budget32768
below262144 native points. API validation now rejects it before queue submission
with a controller-specific message. The UI requires explicit Use requested native
grid budget (262144), or disabling feedback. This does not change physics or claim
that the budget estimates all model memory. Completed smoke inference correctly
reports FINISH_BUDGET_EXHAUSTED with unresolved constraints, not physical success.

## Why the current model is poor: bounded evidence

The unchanged existing3000-update heat32² CNN has34,372 parameters. A copied model
fits two frozen physical/source/time pairs: velocity MSE0.0910508→0.00903343 in100
real Adam updates (2.269s). Original checkpoint SHA is unchanged. This tests
optimization/memorization, not generalization or a unique explanation of poor error.

Two fixed validation problems, three seeds91/92/93, four particles and declared
4/8 Heun budgets preserve every mode/metric. Mean unobserved relative L2:

| Integration steps | Unguided | Observation guidance | Observation + PDE | Fourier inverse |
|---|---:|---:|---:|---:|
|4|1.25864|1.24298|1.24281|0.83737|
|8|1.30638|1.28329|1.28308|0.83737|

The Fourier inverse remains better; more integration steps did not improve these
particular paired errors. Neither budget was selected as a winner. No test set or
new long training-budget study was used. Reports include parameter count, elapsed
cost/memory, actual guidance settings and all seeds/cases. Earlier matched20-update
small/medium capacity evidence is preserved in flow-workspace-20261026/capacity-
comparison/; its different validation cases must not be pooled with this diagnostic.

The fresh1000-update tractable Gaussian diagnostic has generated mean1.15594/std
0.626736 vs1.25/.65, absolute errors.09406/.02326 (2048 particles,48 Heun steps).
Finite learning/integration bias remains. Prior/held-out means, variances, radial
FFT power and per-time RMS are saved separately; repeated reference arrays are
not extra independent cases. Sparse inverse error and prior distribution fit are
different questions. Source scaling/conditioning/path and no-truth inference were
checked. Insufficient long training, global correlations, distribution mismatch,
guidance and identifiability remain hypotheses requiring controlled experiments;
we did not claim that model size alone explains the failure.

Artifacts: learning-diagnostic/diagnostics.json, baseline-{0,1}.json,
learning-summary.json; prior matched capacity manifests remain at their old paths.

## Visual, analytical and lecture evidence

Actual screenshots in browser/: chemistry-mismatch.png, training-history.png,
training-storage.png, active-cli-progress.png, native-512-linked.png and
native-512-generation.png. Retained journeys write browser-regression/ without
overwriting earlier accepted screenshots. The images were visually inspected.
The native512 smoke is visibly inaccurate; it demonstrates functioning numerical
and comparison plumbing, not a recovered solution.

Product/single-axis/multi-mode and labelled challenge heat expressions use actual
physical coordinates, domain lengths, known D and current saved times. Unit/non-
unit2D/3D checks verify independent PDE derivatives, periodic values/normal gradients
and supplied initial agreement. Edited inconsistent candidates reject. Numerical
consistency is explicitly finite verification, not unrestricted symbolic proof.
Initial/reference fields remain evaluation-only; actual inverse worker requests
contain no analytical_reference input. Comparison retains per-field/per-time
absolute metrics and reference norms with near-zero relative-L2 rules.

All seven lecture PDFs were extracted and relevant mathematical pages rendered.
lecture-index.json and lecture-inspection.png record the supplied sources;
generative-foundations.md maps actual physical PDF pages to implemented code,
optional proposals or background. No AE/VAE, autoregressive/diffusion model,
likelihood estimator, ELBO or KL term was added to CFM.

## Failures retained and corrected

- An initial catalogue import used the wrong TimeChannelLayout module; fixed before
  the successful bounded suite checks.
- Old frontend mock responses lacked the new endpoint shapes; defensive empty
  states fixed the crashes without inventing data.
- Browser testing found a clickable-before-loaded512 metadata action; it now waits.
- Nested details/duplicate text selectors were corrected; numerical assertions were
  not weakened. A batch failed because the old frontend listener disappeared;
  the task-owned standard Node/Vite server was started and journeys rerun.
- Native512 compatibility validation originally missed the controller grid limit;
  fixed both the API gate and explicit UI review before successful execution.
- Live overview attempted to format a missing rate during actual preparation;
  fixed and unit/browser tested. UI and terminal counters now floor at saved progress
  when old throttled telemetry lags the checkpoint; no stale session throughput/ETA.
- Full CPU model construction during inspection was removed in favor of actual
  meta-factory shape validation; cached storage preview no longer rescans the tree.
- Sandbox OpenMP shared-memory restrictions affected one read-only config check;
  it passed in the permitted normal local runtime. A portability script initially
  looked for path in the raw record rather than resolve_run's returned Path;
  corrected before the actual successful forward check.

No automatic approval-review action remains blocked. No commit, reset or staging
operation was attempted. Original90-entry index SHA remains
9c5f4385c42492042ad13d7a58854995f74e30e47ed9ba06ae7f4b65daead9c8;
main has no HEAD. Final whitespace/Git/service evidence is in completion.json.
No automatic archive regeneration or scientific-output pruning was introduced.
