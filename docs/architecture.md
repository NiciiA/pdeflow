# PDEFlow: architecture and component contracts

Status: semantic architecture with implemented heat/Darcy/chemistry slices (see multispecialist.md). Contracts below are semantic; use lightweight dataclasses, protocols, and concrete implementations rather than a deep abstract-class hierarchy.

## 1. Dependency boundaries

Core data contracts must not import neural backbones, CLI code, experiment runners, or a particular PDE. Equations depend on core contracts and a supplied numerical backend. Generative formulations depend on model adapters, not equation-specific code. Steering depends on evaluators and sampler capabilities, not on a hard-coded U-Net or PDE name. Notebook code calls public package functions.

Suggested layout:

```text
AGENTS.md
README.md
pyproject.toml
docs/{project,architecture,experiments,mathematics,decisions}.md
references/
configs/{data,models,training,inference,experiments,runtime}/
src/pdeflow/
  core/ equations/ numerics/ data/ observations/ conditioning/
  models/ generative/ sampling/ steering/ refinement/ evaluation/ runtime/
notebooks/
prompts/
scripts/
tests/{unit,numerics,integration}/
```

Outputs, caches and checkpoints are configurable external/runtime paths, ignored by Git. Do not create empty Python modules for every future feature. A capability table can describe planned components without fake implementations.

## 2. Data model

### ProblemSpec

Immutable/serializable description: schema_version, family, operator_spec, domain, field_specs, known_parameters, known_field_inputs, boundary_specs, initial_specs, unknown_specs, observation_spec, noise_spec, discretization, units/nondimensionalization. References to large arrays belong in data records rather than enormous YAML files.

Validate dimensions, field names, compatible boundary conditions, coefficient domains (e.g. positive diffusivity), and known/unknown exclusivity. Unknown coefficient priors and transforms must be explicit. Some unknowns are scalar vectors rather than gridded fields: allow a structured StateBundle, do not force every parameter into a spatial image.

### GridSpec / DomainSpec

Spatial dimension d in {1,2,3}, axis names/order, extents, cell- or node-centered convention, coordinates/spacings, quadrature weights, boundary labels, periodicity, geometry masks, physical time coordinates. A periodic grid must not duplicate its endpoint. Uniform Cartesian grids are the first backend; irregular meshes require a separate implementation.

### FieldBatch / StateBundle

Canonical grid value convention: [B,C,T,*spatial], with d explicit spatial axes. Static fields use T=1. Validity masks and variable-length time support, if added, are explicit. Record channel meanings; vector components and chemical species are named, not inferred from channel count. Known static coefficient fields may be separate blocks, without wastefully repeating them over T.

Distinguish raw physical fields from transformed model states. TransformBundle owns encode/decode and records normalization computed on TRAINING data only. Scaling must propagate correctly through residual derivatives. StateBundle can group gridded fields and scalar unknown parameters.

### ObservationSet

Values, field/component identifiers, coordinates and/or indices, physical time, validity mask, ObservationOperator, likelihood/noise metadata. Missing is not zero. H.apply(physical_state) gives predicted measurements with stable ordering; interpolation and its gradients are explicit. Noiseless constraints use a specified tolerance; never divide by zero observation variance.

### Conditioning

Contains only features consumed by the model: physical knowns, equation encoding, geometry, optional observations. Record missing/unknown flags. Never include hidden reference targets. Keep fixed equation/parameter conditioning separate from droppable observation conditioning for CFG.

### SamplingState / ReconstructionResult

SamplingState: current model state, generation coordinate/noise level, step index, RNG handle, optional particle weights/ancestry, solver diagnostics. If using particles, expose [problem,particle,...] semantically; do not confuse independent problem batch items with particles that interact.

Result: physical samples and inferred parameters, final weights where meaningful, optional summaries, diagnostics, stopping reason, model/config/data/code identifiers, elapsed time, network evaluations, peak memory where measured. Mark whether values have undergone numerical correction. Sample variance is not automatically calibrated uncertainty.

## 3. Physics and numerical interfaces

PDEFamily.sample_problem(rng, config) -> ProblemSpec + input arrays.
ReferenceSolver.solve(problem) -> physical reference state + solver diagnostics.
PhysicsEvaluator.evaluate(physical_state, problem, backend) -> named ResidualBundle and diagnostics.
Constraint.evaluate(...) -> named violations; optional projection/parameterization with declared semantics.

ResidualBundle retains pointwise/component residuals, masks, quadrature weights, units/scales, and named reductions. Do not blindly average incompatible units. Initial conditions may be unknown: only enforce the known portions or the defined prior.

NumericalBackend supplies derivatives/fluxes in spatial axes and physical time, with explicit boundary rules. Provide periodic finite differences and spectral derivatives first. Dirichlet/Neumann support is added and verified explicitly, not approximated by unrelated padding. Geometry masks alone do not implement an embedded-boundary solver.

Implement analytic/manufactured checks separately from data generation. Heat reference trajectories can use exact Fourier evolution for smooth periodic data, while residual validation uses independent finite differences plus grid-convergence checks. Poisson may use a periodic spectral solve with mean-zero source and a fixed zero-mean gauge. Verify compatibility and document its nullspace; generic Neumann Poisson is not uniquely determined without a gauge and compatibility conditions.

Reference solvers may run CPU-only; differentiable physics guidance must use a tensor-compatible evaluator without detach/NumPy conversions in the gradient path.

## 4. Equation representation

Supported expression nodes: Field, Constant, Coefficient, Add, Multiply, Power, TimeDerivative, Gradient, Divergence, Laplacian, Component/Vector where implemented. Validate fields, tensor ranks, operator order and axes. Serialize expression trees safely; never execute arbitrary eval of configuration strings.

Initial family factories generate known expressions. A residual evaluator consumes the expression with numerical operators. A separate condition encoder consumes a family identifier and parameter features initially; generic expression encoding is future work. Do not claim arbitrary-equation neural understanding because the evaluator accepts a tree.

## 5. Dataset and persistence

Each record has a stable problem ID, ProblemSpec, physical reference arrays, generator version and seed. Split by complete problem/trajectory before generating masks or crops. Store train-derived transforms and split manifests. A test trajectory must never appear as a training crop.

Use one simple storage format initially (e.g. NPZ shards plus JSON metadata); large/chunked formats can be adapters. Chunking and compression changes must not alter science metadata. Cache materialization is separate from persistent storage. Check writes atomically; prevent partial checkpoints masquerading as complete.

## 6. Model and generative interfaces

Backbone.forward(model_state, generation_embedding, encoded_condition) -> prediction of the declared shape. Backbones declare spatial dimension, channels, fixed/variable physical-time handling, geometry/boundary assumptions and dtype/device support.

Initial practical backbone: small residual spatial CNN, with fixed physical-time frames packed into channels by a reversible adapter. Use Conv1d/2d/3d for d=1/2/3, not physical time as a spatial coordinate by accident. Save T and time coordinates in metadata. Supporting a new T requires an explicit compatible model/adaptation, not a silent reshape.

ConditionEncoder handles scalar embeddings, spatial coefficient maps, masks/coordinates, and optional observation encoding. It can later use attention. Use one concrete encoder before introducing a complex registry.

GenerativeFormulation defines:

- sample_training_state(target, source_rng, generation_rng, condition) -> intermediate state, target prediction, time, loss weights;
- training_loss(prediction, target, weights, valid_mask);
- dynamics(model, state, time, condition) -> declared ODE drift or SDE drift/diffusion;
- estimate_endpoint(...), only when supported;
- optional score(...), only when valid for its path.

Linear CFM: X_s=(1-s)Z+sU and target U-Z; endpoint estimate X_s+(1-s)v_theta is an approximation. In the ideal independent Gaussian-source linear path, the marginal score is (s v_s(x)-x)/(1-s) for s<1. Derive conventions, treat endpoint singularities explicitly, and do not use the formula outside its assumptions. A numerical implementation should specify truncation/noise schedules, not divide by zero at s=1.

Diffusion is a separate formulation with an explicitly chosen continuous or discrete noise process, prediction type, loss weighting and sampler. Preserve source/target time conventions in mathematics.md.

## 7. Training orchestration

Trainer composes data loading, transforms, source/time sampling, conditioning/dropout, backbone, formulation, optimizer, scheduler and validation. Hooks may record metrics/checkpoints but must not contain PDE-specific branching. Training can be driven by a max-step or epoch budget, always explicit.

Checkpoint schema: version, model+encoder weights/config, formulation/path, field transform, field/channel/time metadata, capability declaration, resolved config, split/data IDs, code revision if available, training progress. Resume checkpoints also include optimizer/scheduler/precision-scaler states and RNG states. Exportable inference checkpoints omit optimizer state but retain all inference metadata.

Use portable loading with device mapping and validated metadata. Treat untrusted checkpoints according to the underlying loader's safe format options. Do not require training data for inference.

## 8. Sampling decomposition

Dynamics: mathematical ODE/SDE including compatible steering.
Schedule: ordered generation coordinates/noise levels with endpoint conventions.
Stepper: advance dynamics, e.g. Euler, Heun for ODE or Euler–Maruyama for SDE.
Controller: optional error-based step adaptation, separate from schedule definition.
Sampler: orchestrates dynamics, stepper, schedule/controller, RNG, callbacks and output.

Stepper interfaces must return diagnostics including model evaluation counts. Fixed-step Euler/Heun ODE implementations are required first. Adaptive ODE and SDE implementations are real additions with tests; never present adaptive SDE stepping as a trivial ODE substitution (Brownian increments require consistent treatment).

## 9. Evaluators and steering

ObservationLikelihood evaluates H(U) against y using the stated noise model. PhysicsPotential combines appropriately scaled PDE, boundary, initial, and admissibility terms; it is a soft preference unless an actual hard constraint mechanism is used.

For first gradient steering, implement a documented split algorithm: an ODE proposal followed by a configurable negative-gradient correction of E(endpoint_estimate(current_state)). Differentiate through the endpoint estimator and model prediction. Label this heuristic accurately; do not claim exact posterior sampling or silently identify it with a published solver. Log scales and schedules; suppress/define endpoint behaviour.

CFG is a separate model-output combination using trained condition dropout. It is not an FK strategy and not available on a checkpoint never trained for missing conditions.

FK requires: compatible stochastic dynamics, endpoint estimator, incremental potential sequence, normalized log weights, stable resampling, particle genealogy, RNG ownership and effective sample size (ESS). Prefer derived potentials with a clearly stated telescoping target; repeated absolute exp(-E) factors change the target. Any proposal changes or rejuvenation must have stated weighting/invariance implications. Noise injection must use a justified drift correction when preserving base marginals is intended.

FK is now verified on tractable Gaussian and mixture targets before bounded PDE integration. The implementation uses fixed Euler–Maruyama steps and no adaptive SDE/global synchronization claim. See fk-mathematics.md and fk-verification.md.

## 10. Refinement

ResolutionPolicy assesses physical error indicators/convergence and requests a supported grid level. TransferOperator implements restriction/prolongation with relevant conservation rules. Refiner invokes a compatible fine model, learned conditional refinement, numerical correction, or a documented hybrid. Record origin of recovered detail and cost.

First implement global resolution selection and evaluation across separately validated grid/model configurations. Do not imply a fixed-grid checkpoint supports arbitrary resolution because convolutions happen to run. Local AMR and irregular mesh networks remain additional backends.

## 11. Runtime and configuration

Use validated serializable configs, a small explicit component registry or factories, and command entry points. Useful commands: generate, train, reconstruct, evaluate, inspect-checkpoint, validate-config. The exact CLI can be simple argparse; do not add a heavy orchestration framework without a concrete need.

Resolve paths relative to a documented project/data root, never a developer's home directory. RuntimeSpec includes device auto/cpu/cuda/mps, dtype, precision policy, seeds, worker count, cache path, persistent output path. Log resolved settings. Fail early on unsupported precision or operations; any fallback is visible and deliberate.

CPU smoke profile is required. CUDA and MPS must only be marked tested when executed on that hardware. On MPS use conservative supported precision initially; keep numerical reference float64 checks on CPU if necessary. Save/load across devices must work for CPU roundtrips and be hardware-tested where available.

Colab is a thin installation/config/launch/export notebook. It may mount user-selected persistent storage interactively but cannot assume credentials or a repository URL. Remote training can use a pinned Git revision or uploaded source archive. No duplicated algorithms in notebooks.

## 12. Compatibility and observability

Validate before expensive work: family, dimension, channel/time layout, grid, boundary and geometry support; conditioning keys; transforms; sampler/formulation compatibility; steering requirements; particle axes; dtype/device support. Unknown operators and unsupported combinations produce actionable errors.

Resolved configuration and run manifest are written alongside outputs. Log seed, backend, versions, checkpoint ID, PDE/grid specification, model calls, elapsed time, measured memory, numerical failures and stop reason. Do not record imaginary hardware or performance claims.

## Implemented workbench and inference boundary (continuation)

The current implementation adds workbench/api.py (validated local transport),
workbench/observations.py (physical-coordinate CSV/JSON conversion) and
workbench/jobs.py (one spawned compute worker, durable atomic records).
Reconstruction workers receive only server-owned checkpoint IDs, known problem
metadata, observations, runtime and inference settings. No reference field is
accepted. Reference attachment and metric/figure generation happen after
inference. CLI comparison also completes all sample generation before opening
optional truth. See workbench.md for endpoints, containment and resource limits.

frontend/ contains React/TypeScript/Vite with generated shadcn components and
Tailwind 4. It consumes capabilities and real result slices from Python; it
contains no scientific solver. Canonical JSON and form fields share one draft;
syntax, residual, reference solver and checkpoint support are separate states.
The bounded experiment runner is experiments.py; it selects settings only on
validation cases and exports snapshots along one training trajectory. No new
PDE execution was added. Darcy's concrete interface requirements are specified
in darcy.md rather than placeholder classes.

## Symbolic requests and scalable heat data

symbolic/ owns bounded source normalization, PDE parsing, scalar/vector ranks,
rate-definition checks and explicit heat recognition. The versioned mathematical
request holds source, declarations, domain, boundary/initial roles, measurements
and operator metadata. It is distinct from the parsed AST and executable core
ProblemSpec; no broad core-contract rewrite was needed. The workbench parse
endpoint returns support layers; reconstruction rechecks current source and
measurement agreement. The frontend MathLive component owns editing only,
SymbolicEditor owns the full request draft/review/import/export, and App retains
job/observation/result orchestration. Invalid drafts do not become executable
through a stale previous parse. Details are in symbolic-editor.md.

DataConfig chooses original smoke NPZ or restartable sharded_npz. data/shards.py
owns immutable instance/split identities, atomic shard sidecars, first-read
hash checks, two-shard lazy cache and training-only streaming statistics. Trainer
now encodes/transfers only a sampled batch and validates in batches, preserving
Adam/RNG/checkpoint boundaries. Profile estimation and preflight are explicit
CLI actions; opening the workbench never generates data or trains. No new
neural PDE family, scheduler, accumulation or worker pool was introduced.

## Implemented specialist boundary (2026-10-21)

`specialists/` contains the small explicit protocol/registry, with heat and
Darcy implementations. It owns canonical operator recognition, assumptions,
physical roles, transforms, data/conditioning, residual factories, observation
compatibility, baseline and evaluation adapters. `darcy/` holds the independent
NumPy reference, tensor flux residual, positive two-channel transform and
restartable data implementation. Registry dispatch selects physical behaviour;
backbone name, formulation and trained checkpoint remain separate choices.

The common trainer/sampler have no family-specific PDE algebra. Static Darcy
uses physical [B,2,1,*space] (a,u), model coordinates (log a,u), known spatial f
through a map encoder. Heat retains legacy model weights and nodal transient
layout. Grid temporal_kind defaults to transient for old heat exports. Static
T=1 is an internal storage index, absent from physical-time observation/UI
controls. Core known_fields carries only explicit physical conditions.

API parsing builds an executable ProblemSpec only after validating the current
full request. Checkpoint routing checks its specialist/layout/schema and emits
compatible IDs/exclusions/rationale; automatic choice is lexical content-ID
order. Explicit incompatible overrides never silently fall back. Submission
reparses mathematical source independently of React and rejects stale source,
wrong-family checkpoints or measurements. Discovery retires replaced exports;
workers verify the selected content ID before and after load. Numerical
references remain evaluation-only and cannot enter the reconstruction schema.

## Common physical data providers and third specialist

data/physical.py separates PhysicalSample, stateless ID/specification sampling,
reference solving and batch conditioning from generative CFM pair construction.
Each specialist owns sample_spec/solve_spec; the trainer remains independent of
PDE algebra. data/providers.py implements one synchronous producer plus legacy
offline adaptation. New modes share a clean physical sample schema and frozen
calibration; optional provider config preserves old DataConfig dataset hashes.

Fixed manifest schema2 identities exclude runtime measurements while record/
array hashes enforce integrity. ID allocation is one producer/rank only.
Config counts are preparation targets; an explicitly supplied validated
evaluation manifest is authoritative at runtime, and actual split IDs are
checkpointed. Hybrid replay has bounded slots/reuse; no speculative worker
prefetch is enabled. Logical cursor/pool/usage and journal audits distinguish
generated-ahead, consumed, regenerated and incomplete calls.

Checkpoint training_ids retains its legacy transform-provenance meaning: for
new providers it contains calibration IDs, not the count of training examples.
Actual consumed IDs/counters are in provider_state. Frozen transforms and
optimizer/source/time/device RNG states restore exact CPU sequences. SIGINT
defers to completed updates and atomic save, with finally cleanup.

chemistry/ provides a positive Strang/exact-reaction reference, square-root
representation and differentiable multispecies physical residual. Its known
kinetic encoder is separate from the shared spatial backbone. Current-source
operator recognition supports the explicit coupled system/rate definition,
including equation reordering; wrong family/roles/kinetics/layouts are rejected
by API validation independently of the UI.

Conventional Darcy inversion is in darcy/inverse.py, with a checked implicit
adjoint and physical forward solve per candidate. Shared reconstruction workers
export baselines reference-free; truth attachment updates their evaluation only.
The bounded study is darcy/study.py. Frontend FieldDiagnostics and TrainingDataCard
extend results/config export, rather than duplicating the workbench or creating
a remote training service.

## Verified process feedback boundary (25 October 2026)

sampling.sample dispatches optional inference.controller to process.run; the
original ODE/guidance implementation remains _sample_once. ProcessConfig owns
limits/thresholds, ProcessState owns recent diagnostics/actions/global budgets,
and process.diagnostics.report evaluates decoded physical components without
references. SamplingState exposes the model tensor, generation position, RNG
and latest physical endpoint estimate. Final tensor state is exported in NPZ;
JSON carries grid/checkpoint/particle/seed/RNG metadata and report histories.

refinement.refine_request owns the narrow 2× nodal heat spatial transfer and
calls compatibility validation again. A new fine bundle starts a fresh source;
there is no fabricated noisy-state bridge. CLI --fine-checkpoint and API
fine_model_id resolve this through trusted checkpoint ownership. Workers persist
live decisions and final reports. Frontend ProcessControls/ProcessPanel only
configure/display server logic. Legacy controller=None retains fixed schedule
semantics; the workbench enables bounded feedback with unconfigured thresholds
by default, which cannot claim criteria met. See process-audit.md for every
box/arrow and process-verification.md for actual execution evidence and limits.

## Implemented FK contracts

`fk.FKConfig` validates dynamics/stepper/noise/potential/resampler/ESS/chunk/budgets.
`fk.ParticleState` owns components[B,P,...], CPUfloat64 normalized log weights,
previous potentials, root ancestry, generation coordinate and full numerical
history. `fk.simulate` owns source/noise/resampling RNG banks per problem; it is
PDE-independent. Velocity and named physical-energy callbacks preserve B/P axes;
chunks never define separate populations. Every associated component is jointly
gathered/cloned when parents resample. `fk.pde.PDEEnergy` decodes checkpoint state,
uses actual measurement masks/noise and specialist residual quadrature/scales.
The current PDE adapter handles one request B1, while the core tests independent B2.

`sampling.sample` dispatches explicit fk/importance modes; the existing gradient
controller remains separate. The FK adapter reuses process budgets/final reports,
records legal scheduled RESAMPLE events, and rejects retries/refinement/proposal
changes. Every step's numerical trace persists; diagnostic_every throttles live
notifications with mandatory resampling/terminal notices. Cancellation/budget
exhaustion preserve nonterminal status and available state, without inference
resume. `ReconstructionResult.weights` flows into metrics, figures, NPZ and API
slices. Reference attachment preserves saved FK arrays/RNG metadata/actions.
The UI exposes controls/trace but implements no probability mathematics.

## Everyday orchestration contracts

`local` is a filesystem orchestration layer over Config, providers.prepare,
training.train, cli.reconstruct and the existing evaluators. It contains no PDE,
model, guidance or particle algorithms. Packaged presets resolve real specialist
configs and validated overrides. Low-level config/path CLI calls remain intact.
The default workspace is project-local outputs/local; explicit persistent roots
work outside the source cwd, including the wheel and notebook.

Each runs/<id>/run.json records schema/kind, ID/name, preset, resolved config,
mode/device/dtype, source revision/dirty-state/hash, timestamps, state/progress,
metric/log/evaluation/checkpoint paths. Filesystem scanning rebuilds discovery;
external output directories have additive reference entries. OS advisory locks
serialize training/resume/evaluation per run, automatically releasing after a
crash. Stale PID text cannot hold a lock. Multi-host orchestration is not claimed.

The trainer's optional event callback reports actual stages/checkpoints while
keeping optimizer/random-number order unchanged. Latest inference is always
published atomically; an optional distinct best export minimizes the fixed
validation CFM objective, never test errors. Explicit continuation permits only
increased total budget and runtime device changes while preserving model,
transforms/provider, precision and constant Adam state. Legacy default resume
still requires identical config. Full implementation-source identity remains
strict: old-source training requires its recorded source; inference stays portable.

API discovery reads managed manifests plus optional external links, validates
safe complete inference exports and deduplicates content IDs/roles. Explicit
model-root configuration remains available. Current equation/request matching
still gates every execution. The queue starts/closes only inside the app lifespan
under a job-root owner lock; Uvicorn reload does not create a duplicate queue.
Port preflight matches Uvicorn's SO_REUSEADDR convention to permit restart after
TIME_WAIT while rejecting live listeners. Frontend handles unavailable backend
explicitly and never launches scientific training from startup.

## Persisted reference experiments and suites — 26 October 2026

`reference/solvers.py` selects declared capabilities from the compiled **current**
mathematical request; reference-only `inputs` are separate. `reference/fields.py`
is a bounded AST field parser. Immutable ReferenceSolution directories retain
arrays and provenance; preview endpoints return planes. `reference/observations.py`
composes the shared PointOperator/ObservationSet, independent sensor/noise streams
and explicit disclosure into the existing inverse ProblemSpec and mathematical
request. It persists clean samples separately for evaluation only.
`reference/experiment.py` orchestrates these contracts and existing sampling;
reference arrays are loaded only after reconstruction. No truth enters the worker,
controller, normalization or stopping. `evaluation/comparison.py` builds physical
quadrature/masks and validates coordinates/transfer before EvaluationReport.
Existing ReconstructionResult/export paths, FK arrays and diagnostics remain.

`suites.py` plans a finite ordered collection and invokes only managed
`local.start_training/resume_training`. A suite manifest owns child identities,
source/config hashes, progress/failure/reuse and sequential scheduling. Existing
child locks/checkpoints/provider state and model discovery remain authoritative.
Reference and suite API/schema details: workbench-api.md, reference-experiments.md
and training-suites.md. No second neural solver or trainer is introduced.

## Explicit model selection, native U-Net and passive generation capture

ReconstructionRequest now accepts explicit model_selection mode auto/explicit;
legacy valid model_id remains strict. The server resolves compatible registered
content before creating a worker request and saves that resolved ID/rationale.
Current equation/full layout is authoritative. Request errors separate human
messages from structured technical details. See model-selection-regression.md.

models/multiscale.py adds one periodic multiscale spatial U-Net to make_model,
independent of LinearCFM. ModelConfig records all capacity/coordinate/temporal/
checkpointing choices; architecture metadata/parameter counts are validated at
loading. Original CNN checkpoints retain their exact architecture. Native inputs/
outputs and fixed saved-time channels remain explicit; no latent/tiling or arbitrary
resolution generalization is introduced. capacity-resolution.md defines tested
profiles and component-level1024² scope.

sampling/snapshots.py passively bounds/captures real current model/decoded/endpoint
states. ODE diagnostic calls are accounted separately without scientific-budget or
RNG changes; FK composes actual parent maps. Separate generation.npz/json survive
post-run reference evaluation. workbench/inspection.py caches actual saved fields,
serves bounded coordinated planes and computes native local equation terms. The
frontend lazy-loads linked visual components without a large new rendering library.

A separate job evaluation-context.json associates a synthetic reference outside
worker request.json. Parent result retrieval applies comparison after success and
preserves it when reopening. Clearing local draft remounts components after removing
only PDEFlow draft/selection keys; callbacks from old requests cannot repopulate it.
Saved data/jobs/checkpoints remain untouched. flow-workspace-guide.md documents the
three visual modes and independent physical/generation coordinates.

## Versioned training catalogue and status/storage views

catalogue.py owns the finite advertised targets and capability/distribution
identities, separately from training effort and strict resume source identity.
suites.py freezes planned targets and reuses only verified successful budgets.
The full/default suite has 11 targets, including heat512² and1024²; execution is
sequential, explicit and never triggered by planning or UI load. Old snapshots
retain their original completion and show current targets absent from that snapshot.

local._execute retains process ownership and atomic checkpoints, adding a small
heartbeat file and throttled progress.json. The existing React Models & Training
surface reads those same CLI manifests; it does not own a second process manager.
TrainingOverview adds coverage, plan-only inspection, live timing and cached storage.
storage.py inventories files without following symlinks; scientific/unknown outputs
remain protected. Review-digest plans can quarantine registered stale disposable
caches, with recovery manifests and no permanent deletion.

reference/analytical.py provides editable domain-scaled heat examples and bounded
independent candidate checks. Inputs remain in the reference/evaluation layer.
learning_diagnostics.py separates fixed-pair optimization, prior distribution
statistics, tractable Gaussian flow and repeated validation inverse probes.

## Optional learned latent-coordinate heat branch — 27 October2026

`coordinate/contracts.py` declares one physical capability and independent query
coordinates/observation sets. model.py implements set reference encoding, latent
transformer/measurement cross-attention velocity and periodic coordinate decoding.
training.py supplies two explicit stages using existing physical sampling/solvers,
runtime, atomic writes and kernel run locks. It freezes the supervised AE and its
training-only transforms before latent CFM, keeping target construction separate
from numerical reference solving and inference. There is no browser training
manager. Resume restores optimizer/RNG/data identities; complete flow exports embed
the frozen AE so portable inference needs no training dataset/dependency path.

`coordinate/inference.py` consumes only physical problem, noisy scalar observations,
query coordinates and sampling controls. It reuses the existing workbench JobManager
through a kind-specific worker dispatch, so cancellation/shutdown still have one
owner. Evaluation loads references only after successful generation. Separate
opaque coordinate model IDs/routes preserve the original fixed-grid checkpoint
schema and compatibility rules. registry.py passively validates stage metadata and
kernel ownership; explicit coordinate-import links standalone completed exports.
Storage recognizes these runs/dependencies and protects active/scientific files.

ReferenceWorkflow now exposes analytical/manufactured, numerical and measurements-
only modes, with separate solver/query/display discretizations, actual observation
totals/noise and field previews. Strict full-field JSON import/export lives in
reference/imports.py; expressions continue through the bounded parser. Per-scalar
sigma extends the backward-compatible ObservationSet and is preserved in baseline
weights, diagnostics and refinement. CoordinateModel adds stage status, explicit
new-backbone reconstruction, query controls, persistence and post-run comparison;
existing fixed-grid visual/workflow components remain.

## Integrated training workflow — 27 October 2026

training_workflow.py adds filesystem-backed orchestration around local.start_training/
resume_training/evaluate_run and coordinate.training.train/evaluation/registration.
It has no optimizer loop. workbench/training.py and the training CLI namespace use
the same plan/start/status/stop/resume/use functions. TrainModel is the ordinary UI;
the previous stage/catalogue/storage interfaces remain behind Advanced tools.

A detached worker owns each workflow using the existing run_lock. Manifests,
submission tokens and cooperative stop requests persist independently of HTTP or
browser lifetime. Matching active requests deduplicate; a released owner lock
cannot imply success. Existing trainers accept an optional should_stop callback
at update boundaries, leaving lower-level command semantics unchanged. Atomic
checkpoint writes and strict coordinate source/data/dependency identities remain.

The representation gate compares validation distortion with a training-fitted
constant baseline, pauses on failure and records an explicit override. Pipeline
checks have an execution-only exemption. Fixed validation previews are post-run
comparisons, outside reference-free inference. No test tuning or automatic model
promotion is added. Fixed-grid continuation snapshots the previous budget before
advancing existing trainer state; latent budget/target changes are unsupported.

known_workspaces discovers managed manifests inside project outputs while pruning
data/cache trees; external workspaces remain explicit. Existing model/run links
and coordinate registration are reused, not a new checkpoint format. Use selects
the content fingerprint, checks current compatibility and explicitly adopts a
training problem only when requested. A new CLI choice preserves the UI draft.

## Continuous heat time and placement — 28 October2026

heat_continuous.py owns bounded finite-Fourier space-time reference evaluation.
coordinate/temporal.py owns physical-time measures/capability declarations,
separately from generative flow time. The procedural dataset owns initial fields
and reproducible encoder sets, not dense animation tensors; engine consumes fresh
query/sensor coordinates and retains one optimizer/resume implementation. Temporal
mode/interval/sampling/placement are in config, data identity and checkpoints.
Legacy configs default to their snapshot capability; new main heat2D profiles are
explicitly continuous. Existing1D/3D/fixed-grid paths are retained.

heat_placement.py owns uniform/equal-area lattice and contrast-region stratified
sampling, with no observation noise or neural model. The playground facade caches
reference/feature analysis, previews designs, separately evaluates noisy values and
persists clean comparison values outside inference. Bounded analysis32² is distinct
from measurement accuracy. Fixed/adaptive timing and policy provenance are explicit.

A successful coordinate job saves its generated decoded latent with checkpoint
fingerprint. Arbitrary-time views load a bounded cached decoder and decode that
state; they do not rerun the generative ODE. Reference at-time views use Fourier
evolution, while cached animation interpolation remains a display policy. Frontend
abort/epoch guards coordinate edits/requests. temporal_validation.py supplies
uniform/time-stratified/weighted metrics and a matched-policy comparison command.
Representation gates compare matching uniform quadratures. All actual evidence
and preserved artifact locations are in continuous-heat-verification.md.

## Field reconstruction loss boundary —29 October2026

field_losses.py owns a validated configuration, reference-only periodic segmentation,
bounded analysis cache, grouped/stratified physical supervision, differentiable FP32
reductions and physical metrics. The existing coordinate engine composes these only
at representation decoding; its latent CFM velocity update remains independent.
Config/checkpoint/run identities retain objective and estimator settings. Local/UI
plans and updated Colab training_preset explicitly recommend α.5 for new runs;
historical missing config remains point-sampled MSE. Workbench previews use actual
training realizations/calibration; comparison endpoints are reference/evaluation
only and cannot enter the inverse worker. FieldLossControls and the heat metric/map
controls remain compact UI adapters. See field-reconstruction-losses.md.

## Historical joint-v1 generator —30 October2026

The previous playground used Physics-Conditioned Joint Latent Flow v1. The old
stacked-frame and solution-only branches described above are historical research
records. New heat2D runs use coupled solution/initial/scalar-D groups, known-status
masks, typed physics/set conditioning and a frozen coordinate decoder. Full tensor,
anchor, loss, export and compatibility contracts are in joint-latent-model.md.
The modular numerical 1D–3D and chemistry/steering goal remains unchanged.

## Active operator-aware physical-state v2 —30 October2026

The current request replaces joint-v1 trajectory generation with factorized initial
u / initial v (wave only) / typed scalar D or c, followed by a learned direct-time
complete-state evolution and a spatial-only decoder. There is one shared heat/wave
observation encoder, edge-aware directed PDE graph encoder, coupled flow processor,
evolution backbone and decoding infrastructure; typed heads and masks handle schemas.
A1 representation → A2 frozen-representation evolution → B frozen-physics CFM is the
shared local/UI/Colab path. Strong residual derivatives pass through evolution.
No conventional reference solver constructs neural dynamics. Static/weak branches
are explicitly unsupported. The broader1D–3D chemistry/steering specification is
preserved, with this bounded2D dynamic slice operational. Exact contracts, counts,
normalization, topology and limitations: operator-wave-model.md. Evidence:
operator-wave-verification.md. Old numerical/source/history remains attributable.

### Batched operator training and preparation

coordinate/batching.py defines per-realization physical-time/query plans and real
A1/A2 neural batches, with wave-only velocity decoding and per-example reductions.
coordinate/frozen.py resolves immutable exact-time encoder requests with bounded
memory/disk caches and one next CPU-reference prefetch; accelerator model calls
remain on the trainer thread. The large identity, source/config and dependency
hashes are setup metadata. Measurements distinguish per-step lookup/reference/
encoding/serialization/IO and cumulative counters. The accumulated execution path
is retained as a numerical control. See docs/batched-training.md; this accelerates
the existing scalar2D heat/wave slice without reducing the modular1D–3D roadmap.

### Authoritative training settings and disposable experiments

training_settings owns batch validation, per-stage resolution, budget conversion,
explicit continuation authorization and immutable Session replacement. The Colab
binding and UI request consume that returned configuration. batch_experiments runs
selected-stage disposable subprocesses and persists identities, results/partial
progress and stable recommendations; it never advances the actual run. Optimizer
successful/attempted/skip/example counters are separate. Workbench batch_settings
is a thin adapter to these shared APIs. No scientific architecture/objective change.

### Reconstruction diagnosis boundaries —1 November2026

coordinate/diagnostic_cases captures immutable initial u/v, physical problem,
observations/noise, independent held-out sensors and explicit flow noise. These
reference-bearing artifacts are evaluation-only. conditioning.py is the shared
production measurement packing path; no reference argument is accepted.
diagnostics.py compares oracle initial encoding, oracle learned evolution and
production observation-only generation; missing matching encoders are unavailable,
not randomly initialized. diagnostic_inventory enforces exact frozen dependencies
and tensors, while diagnostic_metrics owns physical field/sensor statistics.

A separate diagnostic_baseline freezes existing encoders/physics and trains only
a small deterministic observation-to-initial-latent head. diagnostic_correction
instead updates active latents through frozen learned evolution/decoding using
observations and an explicit quadratic trust penalty. Neither replaces the main
model, imports hidden full-field truth into production or establishes posterior
calibration. Existing spatial dimension/PDE module boundaries are unchanged.

### Objective alignment and complete artifacts (1 November)

coordinate/observation_objective.py adds no model parameters: it evaluates a
straight-path clean-endpoint estimate through frozen T/D at exact sensor x/y/t
and quantity. Mixed-family B forwards remain batched; bounded sensor chunks and
per-case reductions preserve accumulation. New likelihood configuration is
separate from legacy auxiliaries, field losses and optional A2 physics.
coordinate/evolution_objective.py supplies independent decoded strong residuals
and opt-in structured phase-sensitive spectral supervision. A scoped existing
math-attention path supports higher derivatives without architecture changes.

persistence.py owns verified content-addressed snapshots with completion markers,
sequence/parent metadata, a last-written advisory pointer and scan recovery.
colab.Session owns first/cadence/barrier saves, unique identities, explicit failure
policy and source capture. training_artifacts.py defines inference/diagnostic/
resumable manifest views over canonical files, including the exact E, dependencies,
all available optimizer/scheduler/scaler/RNG/cursor state and source. Downloads
are optional hash-keyed archives, excluded from routine snapshots. No background
queue or per-update Drive dataset access. See observation-aligned-training.md.

### Standard-MSE new-run selection

field_losses.new_run_default owns the explicit Standard MSE baseline, shared by
CoordinateConfig/presets and UI/CLI/Colab resolution. training_summary records
applicable settings, A1/A2 scope and actual domain/regional query costs. Uniform
domain losses bypass regional reference analysis/cache without changing domain
sampling. Recorded checkpoint resolution is version-aware and separate from new
configuration defaults; resume uses recorded loss and rejects ambiguity/changes.
Architecture and B/likelihood/physics/evaluation paths are unchanged. See
standard-mse-workflow.md for controls and evidence.
