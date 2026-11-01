# Implementation decisions

2026-10-21: Read project.md, architecture.md and experiments.md in full. They
agree on the initial scope. The broader matrix remains a research target;
periodic heat training and Poisson verification are the first complete slice.
No conflicting requirements needed resolution. The initial repository has no
Git metadata and references/ is empty. No user reference was modified. Private
poster/course notes are absent; their contents are not assumed.

Use Python 3.12, PyTorch, NumPy FFT reference numerics, Matplotlib, argparse,
JSON and NPZ, with no external service. Reference numerics stay on CPU float64.
The first neural implementation uses explicit Conv1d/2d/3d adapters, circular
padding, and a fixed time window packed into channels. Model conditions are
family and log diffusivity only; measurements enter solely through steering.
Training-only per-field scalar affine normalization is saved in checkpoints.

One problem batch axis is distinct from independent candidate particles. The
sampler exposes [problem,particle,C,T,*space]; it flattens the first two axes
only inside model calls. There is no interaction, weighting or resampling.

The initial supported physics is nondimensional scalar periodic heat with
known positive constant diffusivity and unknown initial trajectory, plus
periodic zero-mean Poisson reference verification. Unknown coefficients,
Dirichlet/Neumann boundaries and neural Poisson reconstruction are rejected.
The operator tree is a small safe scalar grammar, not an equation encoder.

Use independent Gaussian linear CFM with Euler/Heun and uniform/quadratic
schedules. Split gradient correction acts on the endpoint estimate after each
ODE proposal; its strength is multiplied by the generation step. It changes
the generated distribution and is a reconstruction heuristic. No score
conversion, stochastic dynamics or FK weighting is implemented.

Reference evolution uses exact Fourier heat multipliers. Guidance uses central
physical-time differences and periodic spatial finite differences. Independent
spectral spatial residual diagnostics are also reported. Neither residual is
an exact continuum constraint at the coarse fixed time resolution.

The inverse baseline uses a real finite Fourier basis for the unknown initial
condition and ridge regression through analytic heat evolution. Basis degree,
ridge and guidance scales are configuration values fixed before test use.
No tuning is performed on held-out truth. The initial basis is smaller than the
generator's smooth full discrete spectrum; this mismatch is reported.

Use full precision float32 or CPU/CUDA float64; reject MPS float64 and mixed
precision in this slice. No scheduler/scaler is used, and checkpoint metadata
explicitly records null states. Dedicated CPU RNG streams make same-backend
resume testable; cross-device bitwise reproducibility is not promised.
Atomic writes use a same-directory temporary file and replace; output folders
remain configurable and ignored. Interrupted training resumes the latest
completed checkpoint; partially executed optimizer steps are not persisted.

Verification follow-up: checkpoints include a hash of installed package source
and a Git revision where available. Same-config resume rejects changed source;
CPU inference can read metadata naming an unavailable training accelerator.
The current folder has no Git metadata, so revision is null. Dataset completion
manifests are written after every data/public-input artifact. Atomic file writes
were failure-tested. A failed data-generation folder is never silently reused.

Metrics return null for an empty unobserved set. Periodicity is an imposed
representation rather than an independently measured zero boundary violation;
seam gradients are reported without claiming they should vanish. A test that
expected numeric unobserved error for fully observed 1D data was corrected.
Configured Matplotlib/Fontconfig caches live under a writable output folder.
CPU memory is measured as the process lifetime RSS high-water mark and is
labelled accordingly. Full scientific validation and all accelerator tests
remain outstanding; the conventional inverse outperforms the bounded smoke
models. Exact evidence and actual commands are in verification.md.

Parameter-array meanings belong to the data adapter. Training selects and
transfers generic Conditioning batches; sampling builds a known-scalar
Conditioning from ProblemSpec and passes its mapping to the physics factory.
Heat-specific diffusivity extraction stays in the equation/data adapters.

## Workbench and validation continuation (2026-10-21)

Preserve the accepted outputs. New runs use unique output roots. Reconstruction
finishes all sampling and the inverse baseline before opening evaluation truth.
The API's worker receives only a problem, measurements, a server model ID and
inference settings. Reference attachment is a separate post-inference action.
Use exact on-grid/saved-time physical coordinates for CSV/JSON imports; reject
off-grid data and duplicate measurements rather than silently interpolating.

The local workbench uses React/TypeScript/Vite, registry-generated shadcn
components with Tailwind 4, and FastAPI. One spawned compute subprocess runs at
a time. Atomic JSON records survive refresh/restart; startup marks abandoned
queued/running jobs interrupted. Browser inputs never choose filesystem paths.

The first MPS probe inside the execution sandbox reported unavailable. A
second probe with normal host access reported MPS available. The actual MPS
regression therefore ran outside that sandbox, with float32 tensors and six
optimizer updates, endpoint gradients, both guidance modes and CPU reload.
The numerical worker's initial sandbox run similarly hit an OpenMP shared
memory restriction; the actual subprocess API tests passed with host access.

The budget study uses one training trajectory with exports at 0,100,500 steps.
Two fixed validation problems select among four guidance settings and two
Fourier ridge values at the final budget. The selected settings are frozen
across every test budget. Two fixed test problems and a single model/sampling
seed make this a bounded pilot, not a converged or multi-seed scientific study.
A 10-update timing probe justified the 500-update budget before launch.

The frontend requires Node >=22.12; Node 24.19.0 was used. Early npm attempts
exposed incompatible old lockfile peers; a clean dependency resolution selected
Vite 8.3.5, React 19.3.1 and TypeScript 6.0.3. The shadcn 4.22.1 generator
produced the component source; its one-time CLI dependency was then removed
because it introduced advisory-affected tooling. Generated components remain
real shadcn components. The resulting installed dependency audit reported zero
vulnerabilities. No force/legacy-peer-deps workaround was used.

The task explicitly requests a committed npm lockfile. Because the supplied
folder had no Git metadata, retain the completed source and lockfile in a local
Git repository, with no remote, push or publication. Runtime outputs, weights,
datasets, caches, node_modules and accepted experiment artifacts stay ignored.
Earlier checkpoint Git revision metadata remains null, truthfully recording
its creation before local version control; source hashes remain reproducible.

## Interrupted-handoff recovery

Inspection confirmed that local main has no commits: the previous automatic
approval review rejected the commit action after a usage-limit interruption.
The ninety existing staged additions are left untouched. No commit, escalation,
permission change, reset or staging was attempted during recovery. Corrected
documentation distinguishes staging from a completed commit.

Scientific/frontend sources match the previously packaged final source; reuse
the recorded numerical/browser/MPS results. Repeat only inexpensive frontend
build/unit/lint checks, collect Python tests, load existing compatible exports,
and verify the revised source packager. Add a separate final ZIP with checksums
and omitted-data instructions rather than replacing accepted archives/results.
Exclude generated egg-info from source delivery. This does not alter checkpoint
source identities because the installed scientific package source is unchanged.

## Symbolic editor and scalable-data task

The project, architecture and experiment goals remain consistent. Heat stays a
verification anchor for broader chemistry/steering; editing additional templates
does not narrow or falsely expand the executed physics scope. Scientific
references remain absent; no unavailable paper/code was assumed or reused.

Use npm MathLive 0.111.1 with Vite-bundled fonts and its compute engine disabled.
Explicit Python PDE translation owns meaning. Keep source, parsed nodes and
executable ProblemSpec separate rather than generalizing the numerical Expr
class into unsupported symbolic execution. Recognize only explicit tested heat
forms, with declarations/rank/domain/role checks; other presets are honest
editable templates. Full request JSON v1 is independent of checkpoint/job schema.
Existing API/CLI ProblemSpec clients remain compatible. Conflicting template or
checkpoint-layout changes use a visible diff while preserving observations,
boundary notes and common parameter values.

Use typed DataConfig fields and compressed shards for larger heat distributions.
Preserve the original float64 NPZ smoke semantics; reject unsupported new-profile
options there. New 128-bit per-instance seeds avoid counting repeated masks or
32-bit seed collisions as independent problems. Fixed test identities are
published before generation; resume checks both identity and shard hashes.
Sorted streaming train-only statistics and bounded batch transfer replace whole
training/validation tensors on the accelerator. Keep constant Adam learning rate,
zero loader workers and null scheduler/scaler; do not invent accumulation or
scheduling support. Large jobs remain explicit user commands.

Browser integration exposed numeric JSON spelling and duplicate MathLive input
notifications; structural ProblemSpec equality and identical-state event guards
fix those without relaxing physics compatibility. Detailed outcomes/failures
are recorded in verification.md. The local Git index remains untouched and no
commit, push, remote or service publication was attempted. The unrelated .idea/
folder was left alone.

## Darcy implementation and interrupted-task recovery (2026-10-21)

The project/architecture/experiment objectives agree: add a genuine physical
specialist while preserving chemistry and independent training/sampling
components. Resolve two specific Darcy-blueprint differences explicitly.
The conservative SPD stencil is applied matrix-free in projected NumPy CG,
rather than assembled into a SciPy sparse matrix; the same harmonic shared-face
operator and zero-mean subspace are retained without another mandatory core
dependency. The original acceptance text required an inverse benchmark before
marking the whole slice complete. The current user explicitly permits that
benchmark to remain unfinished. Mark implementation/smoke training separately
from scientific evaluation, and never substitute an oracle forward solve.

Choose the specified periodic cell-centred boundary setup, positive scalar a,
compatible known f, unknown joint (a,u), zero-mean pressure and static T=1.
Use a bounded tanh transform of a smooth Gaussian Fourier field for generated
log a; fixed expected-variance scaling does not use realized targets. This is
a concrete bounded smooth distribution, not an exact unbounded Gaussian log-a
model. Use separate train-only statistics for log a and pressure, spatial
forcing features, shared CFM/CNN/ODE/guidance. Subtract pressure mean at every
decode, charge its cost and report magnitude/distribution change. The low-level
observation schema permits per-field scales; current nondimensional presets
share an explicit scale. New grid execution requires a compatible checkpoint.

Recognition matches explicit parsed operator patterns for heat/Darcy equality
or residual=0 forms, then validates the full request. Automatic model selection
is lexical compatible content-ID order, not test-ranked quality. Manual
incompatible overrides select nothing. API revalidation is mandatory and
references remain evaluation-only. Separate the editable request, recognized
specialist, backbone, formulation, checkpoint and scientific status.

Recovery inspected saved manifests/history/metrics and 87-test evidence before
checking remaining work. Do not rerun accepted CPU/MPS training or the heat
0/100/500 study. The 16³ MPS preflight also completed before the interruption,
as confirmed by its saved preflight.json. The prior automatic launch rejection
was an approval-review usage-limit failure, not a finding of unsafe execution.
Fresh direct launches proved sandbox loopback restrictions; new approval review
then permitted the API/frontend with the user's Homebrew Node22.23.3. No bypass,
permission change or Git mutation was used.

Accept only the exact legacy static observation metadata encoding of zero
internal indices; canonical exports label physical times null. Reject arbitrary
static physical-time values and preserve sensor coordinates. Retire obsolete
content IDs when training replaces an export, and check queued checkpoint
identity before/after load. Both fixes have regressions. Browser/runtime setup
and test-source failures are recorded in verification.md; no numerical tolerance
was weakened. Preserve existing outputs, staged changes and unrelated .idea/.

## Online-provider task: execution plan and data semantics

Implement deterministic single-producer offline/online/hybrid data and frozen
calibration/resume first; verify heat/Darcy and matched costs before the Darcy
inverse study. Then independently verify chemical reaction/diffusion numerics
before real chemistry training/routing/browser integration. Preserve the staged
index and all accepted outputs; current work writes outputs/online-chemistry-20261021.

Use a separate optional provider configuration so adding modes does not change
legacy DataConfig identities/hashes. New common physical samples separate
sampled specifications, reference solve and CFM pair construction. Versioned
SHA256 namespaces allocate independent physical/calibration/held-out streams;
no Python hash or loader-worker allocation is used. The initial supported
producer has workers=0, one rank, no prefetch; hybrid is bounded round-robin
replay with explicit refresh counts. Frozen calibration and generated-ahead
versus consumed cursors are checkpointed. SIGINT defers to safe optimizer
boundaries. These decisions extend supervised reference generation; model
predictions are never treated as clean target data.

## Completed provider / Darcy / chemistry decisions

Physical manifest schema2 excludes measured runtime from logical identity but
checksums each record/target separately. This fixes reproducible held-out IDs
across independently prepared modes. Schema1 remains readable. Evaluation
counts are preparation targets: an explicitly supplied manifest controls actual
runtime split IDs/counts, which are saved. Attempt journals record starts and
completions; actual audits include regeneration/discarded work and label
incomplete calls separately from logical checkpoint consumption. Metadata/history
grows with the finite step budget; target-array storage is bounded by replay
capacity. No asynchronous workers/prefetch were added.

The conventional Darcy baseline uses the generator's bounded positive map,
restricted additive Fourier latent fields, a verified implicit adjoint and
small BFGS/Armijo optimization. Pressure is solved at every candidate, never
independently fitted as unrelated unknown pixels. Select regularization and
flow guidance on validation only. The bounded study's inverse remains better;
six/eight test optimizations hit their preassigned cap, so do not claim full
inverse optimization convergence. Preserve the earlier negative heat pilot.

Chemistry uses the explicitly supported periodic setup. Positive random-walk
uniformization approximates the FD diffusion semigroup with a recorded tiny
Poisson tail normalization; exact local Riccati reaction updates preserve the
right stoichiometric invariants. This avoids unstable explicit reaction or
hidden concentration clipping. Fixed maximum splitting step controls time error;
no adaptive global-error guarantee is advertised. Square-root coordinates give
exact physical zeros/positive decode, with folded negative branches and vanishing
zero-root gradients documented. No invariant projection is introduced.

All three data modes use the same trainer/provider contract. Unknown species
and known kinetic parameters remain separate. Chemical recognition matches the
actual four-row system/rate law, supports row order and stated residual forms,
and intentionally restricts symbol names/algebra. API validation is independent
of the UI. Workbench data-mode selection exports exact configuration/commands
without creating a training service. Reference-only field errors and conventional
baseline evaluation are shown only after explicit attachment. True3D species/time/
slice controls reuse the existing results view.

Timing reports are coarse host accounting; accelerator scalar loss/finite checks
may synchronize in addition to explicit checkpoint/total synchronization. No
extra per-step device timing synchronization was added. Larger MPS/Colab profiles
are provided but long runs were not started. Current concrete checks/artifacts
are recorded in verification.md, not inferred from successful imports or mocks.

## Process-diagram continuation — 25 October 2026

Audit the existing successful online/chemistry implementation rather than
retraining it. The sampler had real split gradients, but no diagnostic feedback
controller, FK or refinement. Add a separate reference-free component report and
bounded deterministic policy around the existing Euler/Heun path. Preserve the
full 𝒞 request and existing specialist factories. Explicit thresholds apply per
mandatory equation/observed field/invariant/gauge; no universal accuracy defaults
and no success when thresholds are absent. Estimated endpoints can toggle future
guidance with patience; only actual final fields can pass final criteria.

Retries are fresh-source seed+1 attempts at complete candidate boundaries;
return the last population without best-score/reference selection. All attempts,
model/guidance/integration calls and refinement levels share explicit limits.
Partial schedules return labelled incomplete-generation fields and budget/failure
status. Final physical checks are distinct from optional reference errors.

Implement one 2× nested periodic nodal heat spatial level with same physical
times. Fine inference starts on its trained Gaussian source and uses unchanged
actual observations/λ. Coarse candidates are not pseudo-measurements or source
noise. Restriction/injection and periodic linear prolongation are diagnostics,
not learned recovered detail; refine_request rechecks fine compatibility. No
suitable fine checkpoint yields an unavailable action. At this historical process
stage FK was absent. The subsequent FK decision below adds a separately derived
SDE and target; no noise is added to the legacy deterministic ODE.

## FK method and evidence decisions — 25 October 2026

The prior specifications' FK-planned/absent status conflicts with the now
implemented continuation; those historical statements are superseded only by
the method and evidence in fk-mathematics.md/fk-verification.md. The research
scope remains modular1D–3D chemistry/steering, not a heat-only reduction.

Choose one derived proposal: Gaussian-source linear-CFM score-corrected SDE,
sigma=sqrt(2*kappa)(1-s), fixed Euler–Maruyama. A learned plugged score does not
prove exact learned ODE marginals. Therefore identify Q_h terminal tilt as the
algorithmic target and describe learned-prior equivalence as approximate.
Use telescoping difference potentials with G0 and actual terminal energy,
fixed beta/eligibility schedules, systematic ESS resampling, explicit CPU RNG
and float64 weights for MPS portability. No gradient proposal is mixed in.
All weighted consumers and reference reexports preserve population semantics.
Live diagnostic frequency throttles notifications only, not numerical history.

Reject FK retries/refinement rather than invent coarse-to-fine weight transfer.
Exercise legacy refinement on MPS separately. Preserve nonterminal cancellation/
budget snapshots without claiming inference resume. Analytic posterior moments,
not appealing PDE images, establish the initial target checks. Retain the heat
negative result and collapsed ancestors; no learned scientific superiority is
claimed. Reference paper v1/revision and independent implementation attribution
are exact; molecular evidence is not validation of this PDE adaptation. Sensitive
MPS conversions must transfer to CPU before double precision. No large training,
remote services, Git staging/commit/reset or accepted-output overwrite occurred.

## Everyday local workflow — 25 October 2026

Resolve the earlier documentation/defaults conflict: accepted QA directories are
historical evidence, not default model roots for new users. Ordinary serve now
discovers managed exports under outputs/local; old outputs are linked explicitly
or discovered through existing advanced roots, never moved or overwritten.
Named presets and a durable filesystem run layer replace repeated shell-path
assembly. Preset configs are packaged resources so headless wheel/notebook use
does not assume the Mac checkout/cwd. The existing trainer/provider/evaluation
functions remain the single algorithm implementation.

Use a meaningful3000-update32² heat pilot and a clearly separate20-update8² quick
check. Other named MacBook profiles reuse existing physical/model layouts;
larger64² CUDA configuration is explicitly untested. Do not claim training
accuracy or duration from orchestration checks. Default validation evaluation is
one case; explicit test/full use is recorded without tuning. Latest means most
recently started run, not best accuracy. Best export means minimum fixed
validation CFM loss, ties earlier; latest remains separately available.

Only explicit total-budget extension/device remapping relax the managed resume
config comparison; constant Adam/state persist, with no scheduler to reset.
Source identity remains strict, and legacy low-level resume semantics persist.
Atomic checkpoints and completed calibration/physical records support recovery.
Kernel locks release after crashes, so no unsafe stale-PID deletion is needed.
Node minimum increased from22.12 to22.13 to satisfy the actual ESLint engine;
.nvmrc22.23.3 and the supported major ranges also satisfy Vitest/Vite. No private
Codex runtime path or global installation is used. A real restart probe exposed
TIME_WAIT false positives; match Uvicorn's reusable socket convention and retain
live-listener rejection. No scientific target or solver formula changed.

## Scientific workbench interaction — October 25 redesign

Use three free workspace stages with an always-visible seven-part request map
and separate Runs / Models & Training destinations. Keep exact mathematical
source authoritative on the server. Preserve unsupported/invalid drafts locally;
do not execute the last valid request. Saved result selection and editable draft
are separate, with an explicit stale-input banner.

Visualization endpoints project actual request arrays, observations, weighted
result ensembles and managed history. Unknown input values remain null. No
interpolation, invented metric values, test-based model rankings or reference
conditioning is introduced. Compatible-run comparison requires exact problem
and observation/noise identity; reference errors additionally require common
truth. SVG charts bound payloads and label missing values. Checkpoint-event
positions are distinct from validation observations and generative CFM loss is
distinct from reconstruction error.

Retain numerical/guidance/FK/controller/layout/source detail under deliberate
disclosure. Legacy expert browser regressions explicitly open these sections;
the primary browser journeys verify that the ordinary path starts with them
closed. Fixed boundaries/layouts and missing generalization evidence remain
visible restrictions. Algorithms, targets and accepted experiment outputs remain
unchanged by this UI continuation.

## Reference and suite integration — 26 October 2026

- Reuse existing mathematical/ProblemSpec/ObservationSet/ReconstructionResult
  contracts. Fully specified reference-only arrays are separate inputs, not a
  second neural-conditioning schema. Immutable reference directories and explicit
  sampling artifacts carry provenance; clean samples stay evaluation-only.
- Auto selects from the **compiled current equation** and actual layout/inputs.
  Only explicit finite-mode heat input receives an analytic classification.
  Other heat inputs evolve a numerical interpolant; Darcy/chemistry are numerical.
- Current joint specialists cannot condition on fully known target fields or
  unknown scalar parameters/forcing. Disclosure rejects these combinations and
  explains the required conditional-specialist work instead of leaking truth.
- Physical quadrature and null near-zero relative norms replace denominator-floor
  relative-error reporting. Two historical isolation fixtures used all-ones Darcy
  pressure; that violates the declared gauge. Their fixtures now use positive a
  and zero-mean pressure. The gauge check was retained, not relaxed.
- Fine reference restriction must also match restricted known forcing/coefficient
  inputs. Different shapes do not authorize comparing a different physical source.
  No temporal interpolation; exact shared saved times is an explicit option.
- Suites only orchestrate existing managed children, sequentially. Quickstart
  reduces update/batch budgets while preserving visible pilot grids; smaller
  single-model quickstart-heat-2d remains the 8² reference-demo route. Explicit
  config/source matching controls reuse and resume, never test quality.
- Resolve a documentation conflict: the process-continuation paragraph in
  project.md historically said FK resampling was unimplemented. The later
  fk-mathematics/fk-verification and architecture records implement/test fixed-step
  FK. That later capability is authoritative; the paragraph now says so. Generic
  temporal/local refinement remains unimplemented. No existing FK work is restarted.

## Selection, native resolution and linked explanation — 26 October 2026

- Reproduced the actual model_id="" serialization in the previous built frontend
  with a real empty registry. Valid physical mathematics had been conflated with
  resolved model selection. Use an explicit auto/explicit selector and authoritative
  server compatibility/content-ID resolution. Empty legacy IDs remain invalid;
  stale explicit selection never falls back to an unrelated model.
- Keep pending mathematical parsing distinct from invalid preserved source JSON.
  Gate validation during pending work; retain actionable invalid-source feedback.
  Structured technical diagnostics remain expandable and drafts survive errors.
- Add exactly one periodic U-Net family, with native fields and fixed physical-time
  packing. Pooling/upsampling internal features does not relabel enlarged coarse
  outputs as native inference. Exact grid/time/domain metadata continues to gate
  every checkpoint. No latent encoder, independent Darcy tiles or large untested
  architecture catalogue is introduced.
- Use explicit small/medium/large configs and a narrow heat1024² resource profile.
  The real MacBook MPS native1024² operation probe passed; trained model quality,
  long schedules and CUDA/Colab are separate unverified claims. Measured allocations
  are point-in-time; RSS is a process lifetime high-water mark, never a claimed
  full training-memory predictor. Default suites retain their original six layouts;
  capacity-local is an explicit two-profile suite. Unused wrong-suite budgets reject.
- Record bounded actual states, never interpolate an animation. Passive endpoint
  calls do not spend controller budgets or alter RNG/state. FK parents compose
  across skipped frames; unrecorded parents are gaps, not smooth particle paths.
- Adopt the supplied animation's restrained typography, linked surface/field/point
  explanation and semantic colours, not its unrelated Hessian topic. Native local
  residuals/fluxes/reaction terms are computed from selected saved mean/sample fields.
  Empirical spread has no invented physical PDE residual. Status stays in document
  flow to avoid covering plots; surface mesh indexing was checked visually/fixed.
- Associate the existing synthetic reference in the parent evaluation layer after
  inference. Keep truth outside worker selection/conditioning/guidance/stopping.
  Clearing the editor confirms and invalidates local callbacks while preserving
  saved experiments and the user's choice to keep or cancel a running job.
- Matched20-update small/medium runs use identical physical/calibration IDs and
  fixed validation. Larger was not better (CFM1.53007 vs1.51396); Fourier inverse
  remains better. No test-selected architecture, budget or particle is introduced.

## Catalogue, compatibility and native 512 continuation

The default train-all now explicitly means the versioned full 11-target catalogue.
local-pilot and quickstart remain named six-target subsets; capacity-local remains
the two medium/large layouts. Existing suite manifests are immutable snapshots,
not silently expanded. Completed loadable exports can satisfy current target
coverage across orchestration source changes when distribution/architecture and
budget match; optimizer resume still requires its recorded source. Seeds and
runtime device do not define a new neural physical capability. Partial smoke
exports remain discoverable but never satisfy a full target's update/batch budget.

Automatic routing now prioritizes update count, completed budgets and export role
within exact compatible candidates. This effort/provenance preference fixes the
possibility of lexical smoke priority; it is not reference-error ranking or a
quality guarantee. The CLI experiment uses the same rule. Bounded polynomial
normalization recognizes only the existing chemistry law in equivalent forms;
fixed declared species names remain explicit. Structured reports list every
layout/physical-law mismatch without irrelevant wrong-family aggregate messages.

The 512² heat profile uses the existing native periodic U-Net, T3 and batch1,
float32 MPS with activation checkpointing. No new latent decoder or independent
tiling is introduced. The finite per-evaluation receptive field is not global
512² information; coordinates and known D do not establish learned global quality.

Training remains one CLI process and existing run locks; a heartbeat thread only
writes status, not training commands. Per-update progress is throttled independently
of actual validation/checkpoint cadence. Inventory uses cached stat sizes and an
explicit refresh. Historical/unknown scientific, model, QA and personal artifacts
are protected. Only explicitly producer-registered, unchanged disposable caches
older than 24 hours can enter a reviewed recoverable-quarantine plan. No scientific
output deletion or quarantine was performed during this implementation.

Analytical references are separate from inverse unknowns and initial disclosures.
Edited candidates are independently checked by bounded coordinate/time finite
differences, periodic value/normal-gradient shifts and initial agreement. Passing
is finite numerical consistency, not unrestricted symbolic proof. Inconsistent
candidates do not produce accepted reference artifacts. Template insertion never
silently changes a fixed checkpoint's time layout.

## 27 October2026: optional coordinate heat model and source workflow

The previous statements that no latent/second staged trainer exists describe the
fixed-grid slice at that date. This task explicitly authorizes a different model:
we add a separate heat capability/staged trainer rather than relax old grid checks
or reinterpret fixed-channel CNN weights. The broader specialist1D–3D/chemistry/
steering roadmap remains unchanged; Darcy/chemistry latent support is not claimed.
The eleven-target default train-all stays frozen in meaning. New AE/flow stages
are explicitly selected through coordinate-train; no large jobs are auto-scheduled.

Choose supervised set-encoder/coordinate-decoder AE then freeze it for latent CFM.
This makes latent targets/normalization stable and permits independent held-out
AE distortion measurement. It is a deterministic AE, not a VAE; no KL/ELBO,
likelihood estimate or decoder-likelihood sampling claim is introduced. Reference
context is used only for training/AE evaluation. Inverse conditioning is an unordered
scalar measurement set plus known constant D; a null token and explicit training
exposure support zero observations. One training reference discretization per
checkpoint is a declared limitation, not hidden multi-resolution learning.

Retain one workbench queue/cancellation owner and the existing filesystem locks,
physical generators, runtime and atomic writes. New stage checkpoint schema is
portable and explicit; completed inference exports embed AE weights, while resume
requires source/config/data/dependency identity. UI refresh reads actual records,
not directory existence. Training metadata/history writes are throttled/bounded;
full histories are separately retained. Explicit import links, rather than moves,
completed exports. No accepted outputs were deleted or migrated.

Three input modes make u_ref, initial conditions, forcing and observations distinct.
Manufactured heat functions are accepted as labelled unverified references with
residual diagnostics; no compatible forcing is fabricated because this heat
specialist does not support forced equations. Full numerical JSON import requires
explicit exact coordinates/axes/times/fields/units; no silent extrapolation. Relative
noise uses field/time RMS and persists actual per-observation absolute sigma,
including measured zeros. Extend shared likelihood/baseline/diagnostic/refinement
paths so heterogeneous noise is not silently reduced to one scalar.

Fairness means identical held-out problems and measurements at comparable query
coordinates, with training/method/compute differences reported. It does not make
4-update latent smoke and3000-update guided CNN scientifically matched. Baseline
and reference interpolation are labelled approximations, not native fine-scale
accuracy. Decoder smoothness and high query counts do not validate a solution
manifold. No architecture winner is selected from the bounded results.

## 27 October 2026: one integrated training experience

Put a normal Train model form and a single training CLI namespace above the
existing scientific trainers. Keep lower-level commands and advanced UI tools.
The orchestrator owns stage paths, fixed splits, dependencies, validation and
registration; users choose problem/model/budget/device. No second optimizer or
model family is introduced. Existing specifications agree on these boundaries;
the change resolves workflow complexity, not a scientific scope conflict.

Reuse the existing 4/4 coordinate smoke and 10000/10000 full profile, the fixed
two-update suite convention and fixed-grid pilot budgets. Development uses500
updates (500 per coordinate stage), following the bounded learning profile and
the recommended AE-first experiment. None of these labels implies sufficient
training. Actual configuration/counts remain visible, including online unique
training-example counts and separate physical/generation/update coordinates.

Use the training-mean constant predictor as an interpretable minimal validation
comparator rather than inventing a universal RMSE threshold. Default ratio1 can
be tightened; failed representation training never launches flow, and a failed
quality gate requires recorded user override. Pipeline checks record their
explicit execution-only exemption and retain poor-quality warnings.

Own training in a detached process under the existing kernel run lock. API/browser
lifetimes must not cancel training. Cooperative stop files avoid PID-reuse hazards;
trainers preserve checkpoints at update boundaries. Tokens plus canonical resolved
request defaults prevent duplicate submissions across UI/CLI. UI poll generations
prevent an older response replacing a newly accepted run. Loading is distinct
from empty history; readiness waits account for historical export discovery.

Discover historical project-output workspaces without moving their files. Preserve
the last completed fixed-grid budget as a registered snapshot before an explicit
train-further action. Coordinate target/budget changes still require a new
experiment. Completing stages never writes the selected-model record: explicit
Use does. CLI selection preserves the mathematical draft; incompatible layout
adoption remains an explicit UI action with Undo.

### 27 October 2026: focused heat playground and explicit storage deletion

The user's latest request changes the default UI to scalar periodic heat in 2D
with coordinate inference. The project specification's wider 1D–3D chemistry and
steering research goal remains intact in Legacy and CLI; this resolves the UI
scope difference without deleting scientific components. Reference simulation is
independent of any model. Existing FFT evolution, bounded field expressions,
coordinate inference and training orchestration are reused; only the scattered
Fourier measurement adapter and focused facade are new. A 64² reference with 31
times is cached per request; output grids divide it, so comparisons use actual
nodes. No animation-frame requests or unlabelled temporal interpolation.

Permanent model deletion is a new explicitly confirmed product capability,
separate from older reversible cache quarantine. Schema/producer recognition,
content/inode fingerprints, model-stage dependency handling, root containment,
no symlink following, active kernel locks and unique-inode accounting are
required. Metadata and unknown/imported scientific data are retained. Default
scope removes all recognized weight/resume copies, not just inference exports.
Only disposable temporary artifacts were removed during acceptance tests.

## 27 October 2026 — substantial recoverable heat Colab training

The main notebook now targets the coordinate heat2D specialist; the previous
multi-specialist notebook is preserved byte-for-byte at colab-legacy.ipynb.
This is a focused training workflow, not a reduction of the project's 1D–3D,
chemistry/steering scope. No conflict with the broader project goals was found.

Local UI/CLI and Colab use coordinate_stages and the same coordinate.engine
optimizer loop. The notebook handles source/storage/settings/monitoring only.
Procedural deterministic split-indexed references replace an eager 4.25 GiB Large
float32 dataset and full latent cache. A finite fixed solution population,
bounded eight-entry cache, train-only calibration and on-demand observation
variants preserve identity without multiplying dataset RAM by worker processes.
The richer smooth periodic mixture changes dataset identity; silent distribution
changes on resume are rejected. Large preserves the requested settings; XL is
an explicit capacity expansion with no asserted GPU fit or scientific benefit.

Resume compatibility binds model/data/calibration identity, optimizer settings,
frozen representation digest and selected scientific source files. Runtime and
evaluation changes are explicit; docs/UI changes do not invalidate scientific
source identity. Deliberate extensions retain scheduler state/horizon and archive
old exports; an existing flow cannot silently change its representation target.
An explicit source-change override is recorded and forfeits exact reproducibility.
Legacy optimizer resumes require the original trainer; old inference is supported.

Persistent writes use checksum-addressed objects plus a verified commit manifest
published last, rather than assuming Drive rename atomicity. Latest resume, best,
bounded periodic local weights and three persistent complete snapshots are kept
by default. Flow resume points to its immutable AE dependency; the final export
embeds it. Retention touches only a marked owned run. Interruptions can lose
updates after the last successful sync; real Drive recovery remains unverified.

The prior 64 MiB upload cap excluded the Large model's ~87.6 MB of float32 weights
before metadata. The existing import endpoint now streams to disk up to 2 GiB;
other API request caps remain bounded. Partial valid exports are allowed and
labelled experimental, without claiming trained accuracy. PyTorch/CUDA are not
reinstalled by notebook setup; supported runtime versions are checked first.
Compilation remains unsupported; SDPA/AMP/checkpointing GPU performance claims
await actual GPU evidence. CPU execution and a real browser import are recorded
in docs/colab-verification.md.

Continuation recovery retains the original scheduler horizon on every subsequent
resume, not only the first extension attempt. Export always prefers the latest
resumable optimizer boundary; an older completed inference file retained during
an interrupted extension cannot silently replace its newer partial state. The
bounded regression covers interrupt, partial export, normal resume and completed
extension reopening.

## 28 October2026 — continuous heat time, stable targets and placement

The project specification's fixed-time stacked channels describe the retained
fixed-grid branch. They do not constrain the optional coordinate branch. New
heat2D local/Colab presets explicitly choose continuous physical time [0,1];
old configs missing that field remain snapshot-trained, not relabelled. This
resolves the specific old four-frame assumption without changing the broad
1D–3D/chemistry/steering purpose or adding another PDE/backbone.

Continuous queries and sensors use direct finite Fourier evolution. Reference
encoder sets are deterministic per underlying solution, with continuously
distributed times, while AE query/sensor variants refresh using saved RNG. This
chooses stable frozen targets over a moving random reference encoder set.
The initial/early/uniform/final objective is deliberately weighted; uniform
quadrature and time strata are separately evaluated. Animation/export frames
are excluded from continuous data/model compatibility identities. Interval,
point budgets and sampling/placement policies remain incompatible resume changes.
The old syntactic decoder flexibility remains in legacy inference with explicit
capability limits. The main playground/API requires declared continuous support.

Sensor placement and measured values/noise have separate stages/seed streams.
An equal-area balanced row lattice handles exact non-square N without one-sided
truncation. Heat focus uses signed mean-relative contrast, periodic plateau
merging/NMS, broad kernels, equal regional allocation and reserved background.
It is a disclosed reference-assisted synthetic design; selection positions can
carry unknown-field information. No field/centres/clean targets enter inference.
Manual sensor mode is distinct from painting and has keyboard input/removal.

Scrubbing requests direct Fourier values and decodes the saved generated latent;
flow is not rerun for each frame. Cached playback uses labelled display
interpolation. Request cancellation/epoch checks prevent old time/design/field
responses from overwriting newer choices. Committed observations remain unchanged
by scrubbing or display-frame changes. Fixed sensors keep positions across chosen
times; adaptive designs retain their separate per-time provenance.

Verification caught a null initial-render guard, an unequal-area non-square row
bias, and a mismatched quality-gate measure (irregular-time average versus uniform
baseline). These were corrected with regressions, not relaxed numerical tolerances.
Final portability review moved train-only FP64 latent moments explicitly to CPU:
MPS does not support device float64. Actual tiny MPS training/validation/export and
CPU reload then passed; no hidden CPU fallback for model optimization is used.
Only bounded pipeline training was run. CUDA/actual Colab and useful accuracy
remain unverified; no model-quality improvement or policy winner is claimed.

## Spatial profiles use current physical arrays —28 October 2026

Use compact SVG curves in the existing canvas layout, not a new plotting library.
Periodic nodal quadrature includes physical widths and the wrap interval once.
Offset/wrap the raster half a cell to align node centres, sensors and curves.
Keep profiles out of physics/config dependencies; signed/magnitude/visibility
changes do not invalidate artifacts. Reference/model share interval value scales,
using cached arrays and current values without new model/solver calls. Profiles
are omitted from error/measurements-only views. Nonperiodic quadrature is outside
this periodic-only playground. See spatial-profiles.md for evidence and limits.

## Real region-balanced representation gradients —29 October2026

Use one shared field_losses.py implementation for local/Colab training, validation,
preview and comparison. Preserve old configs/preset() as MSE; make new workflow
plans and training_preset() explicit experimental RB α.5 variants. Group continuous
physical-time supervision so segmentation has spatial context without reverting to
fixed snapshots. Separate uniform domain queries, additional fixed region queries
and conditioning observations. Fail when a fixed region budget cannot cover every
retained region. Keep per-time and per-solution means, detached masks and FP32 loss
reductions. No latent-token masks or auxiliary flow objective. Preserve checkpoint/
resume source and loss identities; changed objectives require new runs. Use separate
scalar-metric/map controls and unchanged spatial marginals. Formula, cost and
limitations: field-reconstruction-losses.md; evidence: field-loss-verification.md.

## 30 October2026 — joint latent replacement contract

The supplied PDF leaves known-group masking and latent normalization open. Adopt
fixed full-state targets, mask known scalar slots in flow and decode through typed
known context; unknown scalar uses its dedicated bounded head. This preserves one
representation across disclosure masks. docs/joint-latent-model.md establishes the
shapes and semantics before code. The earlier project/architecture descriptions of
stacked-frame and solution-only models are historical evidence, superseded for
active heat playground training by the joint model. Their scientific outputs and
all existing checkpoints remain untouched; other PDE numerical modules and the
broader 1D–3D chemistry goal remain. A heat2D joint slice does not imply a trained
spatial-coefficient, Darcy or chemistry joint generator.

## Operator-aware extension —30 October2026

Replace v1 complete-trajectory groups by initial_u/initial_v/coefficient with
within-group representation pathways and coupled generative flow. A shared
heat/wave first-order state supports wave composition only with BOTH u and v.
Use staged A1 representation, A2 learned direct-time evolution, B frozen-target
CFM; optional decoded auxiliary stays explicit. Static/weak branches are interfaces
with unsupported execution, not success-returning placeholders. Smooth periodic
local kernels replace hard nearest-anchor routing for strong residuals. Full
pre-implementation contracts are in operator-wave-model.md.

### Final shared operator integration —30 October2026

The project specification permits specialists and calls joint training experimental;
the current user request specifically requires one shared heat/wave model. We retain
the modular research goal while making the active2D dynamic model shared. This is a
bounded adaptation of the supplied extension PDF, not evidence for arbitrary PDEs.
Old trajectory/test contracts were replaced by physical-state contracts; their source
snapshots and all existing checkpoint hashes are preserved in the task evidence.

The spatial decoder receives neither physical time nor an unrestricted trajectory
latent. A2 evolves complete(u,v) wave states; identity is structural, composition and
conservation are measured/soft. A single typed edge-aware graph enters flow and
transition; executable physics still comes from an explicit smooth registry. JSON
list/tuple normalization uses canonical digest equality at the API boundary. Typed
known scalars are authoritative; unknown true scalars and full reference u/v never
enter reconstruction. Explicit-noise CPU/MPS tests and API forbidden-input tests
cover both families. Hard local-neighbour selection is replaced by smooth all-anchor
weights for derivatives. Wave Nyquist splitting defines the finite interpolant.

Field/quantity losses average within examples before batch/family aggregation;
observation and supervision placements stay separate. Time-mixture uniform components
use32 jittered phase strata; all sampling generators and stage dependencies resume.
Preview scale calibration is bounded and explicitly approximate; actual run transforms
use the full recorded training-only calibration. Effective example accounting is
persisted across resumes, including changed batch-size attempts. Component clocks are
host-inclusive; only synchronized step wall time supports elapsed-time reporting.
No substantial training or CUDA run was used to qualify the architecture.

### Training-equation routing audit —30 October2026

The focused training page hid the selector and initialized its request from the
playground PDE. Consequently opening it from Heat selected actual heat-only data,
although both scalar ranges were displayed. Explicit joint configuration already
reached the correct shared trainer; the defect was both presentation and intent
routing. Advanced family_weights could also contradict the selector/summary.

New local/UI/API/CLI training defaults to Heat + Wave, with a visible independent
Training equations selector. The resolved summary comes from validated family_weights
and the registered state schemas, not the playground or an equation-name heuristic.
Advanced weights may adjust proportions within the selected family set; a different
set rejects. Initial-field shape mixtures are explicitly distinguished from PDEs.
Stage manifests/checkpoints persist normalized family probabilities and enabled state
schemas, while batch histories record actual example counts. Existing raw configuration,
loss identity, resume checks and numerical losses remain intact.

Region-balanced alpha=.5 remains the new-run default. A1 applies it to heat u and
wave u/v with separate reference-only region masks/normalizations; A2 uses the same
decoded field objective through evolution plus its existing alignment/composition
terms. B remains masked latent CFM. A two-update-per-stage CPU probe and an alpha0
A1 control establish objective/gradient wiring only, not accuracy. Existing models
and recorded resume objectives are unchanged. New metadata changes the scientific
source fingerprint, so older checkpoints still need their recorded source for strict
resume; inference compatibility is unchanged.

## T4 performance recovery — 31 October 2026

Optimize declared band-limited heat/wave references without changing architecture,
point budgets, arbitrary physical-time sampling or region-balanced estimator.
Unrestricted references keep full spectra; invalid declared bandlimits reject.
Bounded frozen-state caches include source/dependency/data/normalization/sampling/
precision identities. No prediction-dependent segmentation or observation caching.
B default CFM now executes true microbatches; A1/A2 retain per-example accumulation.
Native BF16 support determines auto precision: T4 uses FP16 + GradScaler. Preserve
blocking CPU/MPS transfers after a real nonfinite MPS regression with async copies.
Per-update wall clocks replace cumulative data-cost averages; nested host and CUDA
stream clocks have explicit scopes. Recomputable target caches do not enter new
persistent snapshots. Existing artifacts remain untouched and strict source resume
checks are retained. See docs/training-performance.md for before/after evidence,
limitations, original-source benchmark and exact notebook reload steps. No T4 speed
or scientific accuracy improvement is claimed from local CPU measurements.

## Actual representation batches and immutable preparation —31 October2026

A1/A2 now execute real mixed heat/wave neural microbatches; wave-only velocity
indices avoid pooling absent quantities. Sum per-realization losses over the
unchanged effective batch. Preserve the accumulated execution control for strict
loss/gradient comparisons. Share each trainable A2 graph embedding across its
transition/composition calls without detaching gradients.

Hash the large frozen identity once at setup, not once per state query. Resolve
unique exact-time requests through a bounded CPU LRU and one inventoried local
initial-state disk cache; batch frozen encoder misses, with one next CPU reference
chunk prefetched. Keep newly encoded current-batch latents on the device and pack
CPU hits for one transfer. Separate lookup/key/reference/encoder/serialization/disk
clocks and per-step/cumulative counters. No observation/noise/disclosure cache.
Actual8/16/32 sweep uses one explicit effective batch>=32 for every variant; an
original effective8 comparison must instead use microbatches<=8. Never silently
change the training preset to achieve a throughput result. Evidence and limits:
docs/batched-training.md. Medium/T4 measurements still require the supplied sweep.

## Batch experiments become explicit session configuration —31 October2026

Use shared training_settings for validation, per-stage resolution, budget conversion
and actual Session replacement/application. New UI requests consume the returned
configuration; Colab rebinds the returned Session. Do not mutate an active worker,
reuse stale resource profiles or rewrite original continuation provenance. Auto is
conservative and explicitly unqualified until measured. Batch sizes up to1024/custom
are allowed subject to effective/microbatch validation, not a memory-fit promise.

Separate efficiency (same effective/ordered draws/optimizer/science) from optional
learning comparisons (examples, wall, or successful-update budgets). Isolate each
candidate in a subprocess, preserve partial results, distinguish real OOM from errors,
and recommend only completed stable conditions without skipped steps. Larger batch
memory is never assumed to imply throughput. Applying a result is distinct from
benchmark completion and actual training. Successful updates alone advance scheduler/
progress; overflow attempts/examples and consecutive failures persist independently.
See docs/batch-settings.md for exact semantics and evidence; CUDA remains unverified.

## Disposable FP16 anchors use successful-update completion —31 October2026

Anchor exports require two successful optimizer updates. The old two-attempt
budget could be exhausted entirely by scaler backoff. Preserve the existing
consecutive-overflow limit; derive the total anchor-attempt ceiling from that
limit and the successful goal, and keep one scaler alive. Timing candidates
retain their explicit attempted-example comparison budgets. A handled overflow
is not numerically finite or a successful update. Preserve invalid-gradient
locations before clipping, fail a norm-only overflow before optimizer mutation,
and retain worker/parent tracebacks in reports and notebook output. The supplied
T4 logs cannot establish whether the two initial overflows are temporary; native
CUDA recovery remains unverified locally. See docs/anchor-fp16-recovery.md.

## Diagnose reconstruction before replacing architecture —1 November2026

Use actual imported Medium tensors and saved observation sets; do not attribute
poor UI images to smoke models, markers, physics ambiguity or a missing encoder
without checking records. The latest three relevant jobs replay exactly on MPS.
Their generated initial fields already fail, and the informative periodic sensor
system can recover its finite-Fourier initial field. Matching E/A2 is absent, so
A1 versus B cannot yet be causally separated. Do not train a replacement decoder
or enable a decoded FM auxiliary on that evidence alone.

Keep staged oracles, a bounded deterministic diagnostic head and explicit
observation-only latent correction outside production training/inference defaults.
The correction improves the one measured case but remains a changed sample with
remaining physical errors. Use strict dependency/tensor checks, no source-hash
edits or strict=False. Preserve old inference artifacts and source snapshot;
scientific source changes still block ordinary training resume.

Fix inert masked padding, retain actual stage/gate provenance in future exports,
and provide explicit opt-in atom-at-zero noise sampling. The observed noiseless
feature column never moved from initialization, but its controlled ablation has
only a tiny effect on this failure; do not call it the dominant cause. Keep
noiseless_probability absent/0 exactly backward compatible and reject changed
resume distributions. Do not silently alter existing presets, field loss,
region alpha, architecture, point counts, datasets or gate outcomes.

### 1 November: observation-aligned objectives and durable complete runs

The user scope is authoritative; the design PDF offers alternatives, not a demand
for simultaneous attention/decoder/Stage-C redesign. Keep physical-state v2, all
capacity presets and the broader1D–3D chemistry purpose. No project/architecture
scope conflict was found: the periodic2D operator remains a declared slice.

Use a separate beta=0-default decoded clean-endpoint likelihood with exact sensor
coordinates/quantities/noise and independent selection RNG. Freeze A1/A2 in B,
retain gradients through them, preserve real microbatches. Explicit Gaussian
model discrepancy and finite noiseless tolerance avoid disguising sigma flooring
as likelihood. Region-balanced alpha=.5 remains the new field-preset default,
separate from FM, observation likelihood and optional A2 residuals. Structured
complex-DFT supervision is an explicit domain-MSE variant, never an FFT of scattered
queries. Richer attention, direct sensor decoding, joint Stage C and new physics
remain deferred until representation/evolution/conditioning errors are separated.

The old portable inference export intentionally omitted E; the missing original
Medium E/A2 cannot be recreated. New snapshots retain exact source and complete
canonical stages from the first valid checkpoint. Inference remains portable;
diagnostic/resumable artifacts are verified manifest views so frozen files dedup
rather than multiplying multi-GB archives. Drive rename atomicity is not assumed.
Objects and complete manifests verify before the advisory pointer is updated;
restore scans if it is stale. Default save-failure policy stops after bounded
retries. Unique run ID differs from display name; restore reads recorded science.
Resuming different source/objectives is never silently authorized. Explicit
new-run warmup rescaling is available for budget conversion; otherwise reject an
incompatible warmup. Functional resume differs from cross-device exact replay.

An actual MPS check found unique_dim unsupported. Duplicate/selection metadata
validation now runs explicitly on CPU for all devices, not via concealed model
fallback. Mathematical loss/reductions and gradient paths remain on-device.
The CPU checkpoint replay remains exact within the tested compatible environment.

### 1 November — Standard MSE becomes the NEW-run default

The latest user request supersedes the earlier experimental RB-default policy.
Keep the exact existing RB formulation/alpha/detection/stratified estimator;
select it explicitly, starting at alpha.5. All new-run entry points now use
Standard MSE. This resolves the policy disagreement with the previous handoff;
historical run metadata and scientific outputs remain immutable. No architecture,
flow/observation/physics/correction or display-metric change is implied.

Uniform domain objectives do not need region analysis. Skip analysis frames,
segmentation, regional reference queries and cache allocation while preserving
the original domain/time RNG and count. Mask zero-weight invalid points before
subtraction. Version-aware checkpoint resolution preserves recorded/legacy loss
semantics, rejects ambiguous/conflicting new records, and never inherits a new
preset on resume. Source checks remain strict. See standard-mse-workflow.md.
