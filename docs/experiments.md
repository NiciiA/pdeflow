# PDEFlow: experiments and acceptance criteria

## 1. Questions

Primary: how do generative priors and inference-time physical constraints affect reconstruction from sparse measurements across selected PDE families in up to three spatial dimensions?

Secondary comparisons, introduced when implementations exist:

1. Flow matching versus a specified diffusion formulation.
2. Observation-only versus observation-plus-physics guidance.
3. Gradient guidance versus FK steering at comparable measured computational budgets.
4. Independent specialists versus a shared model conditioned on equation structure.
5. Fixed resolution versus validated refinement.

Do not require a full Cartesian product of every family, backbone, dimension, sampler and steering method. Establish the main reconstruction task, then add targeted comparisons. Claims about unseen equations require an explicit held-out structural test, not just new random seeds.

## 2. Experimental hierarchy

### A. Mathematical and numerical verification

- Derivatives on analytic functions in 1D, 2D and 3D, checking convergence as the grid is refined.
- Periodic endpoint and axis conventions; vector divergence and Laplacian in multiple dimensions.
- Exact heat Fourier modes: amplitudes decay as exp(-D |k|^2 tau) with the correct physical-domain wave numbers.
- Periodic Poisson: mean-zero forcing, zero-mean gauge, manufactured solution and convergence. A source with incompatible mean must be rejected or explicitly projected with that action recorded.
- Observation indexing/interpolation and differentiability; missing values distinct from observed zeros.
- Encode/decode roundtrip; training-only statistics; derivative/residual scaling in physical coordinates.
- ODE steppers against analytic ODEs; verify observed convergence order and model evaluation counts.
- Gradient steering: gradient against finite differences on a small smooth case and evidence of observation-loss reduction for a controlled small update.
- Checkpoint reload, resume semantics, config compatibility errors and RNG ownership.

These tests are not evidence of learned scientific performance.

### B. First end-to-end integration

Use positive-diffusivity periodic heat trajectories with randomized smooth initial profiles, fixed known physical time coordinates, and explicit diffusivity conditioning. Keep initial fields signed in the numerical baseline if using Gaussian coefficients; do not call them nonnegative chemical concentrations.

Profiles below are initial engineering defaults, adjustable after measuring hardware. Record actual values in configs and results; sample counts are not assertions of adequacy.

| Profile | Geometry and state | Data / training intent | Acceptance |
|---|---|---|---|
| CPU smoke | 2D periodic 8x8 or 16x16, 4–8 physical times | Tens of trajectories, tens of optimizer steps | Actual generate/train/save/reload/reconstruct/evaluate execution |
| 3D smoke | True 3D periodic 8x8x8, 4 physical times | Tiny network/data and short optimizer run | Correct 3D operators, backbone, training and reconstruction; not a performance benchmark |
| Local pilot | 2D 32x32, roughly 8–16 times | Hundreds/thousands of trajectories as resources permit | Useful training curves and reproducible held-out figures |
| Colab pilot | 2D 64x64 or 3D 16^3/32^3 with explicit time window | Hardware-sized batches/model, resumable training | Launchable notebook; hardware-tested status reported honestly |

The source noise is a tensor of matching model-state shape. Flow matching uses the documented independent Gaussian linear path. Use explicit source/time/sensor/data RNG streams. Store exact split IDs and model transforms.

Observation cases: spatial sensors observed over all saved physical times; scattered space–time observations; and partial initial/final observations where defined. Start with one case and keep APIs extensible. Physical time endpoints available to the evaluator are not automatically observed initial data. Test reference values must not enter generation except through declared observations.

Required initial comparisons: unguided flow, observation-guided flow, observation-plus-PDE-guided flow. Do not require every guided run to improve on an undertrained smoke model; require correctness and transparent reporting. Use validation data to tune guidance.

### C. Broader family coverage

| Family | Principal experiment | Scientific checks |
|---|---|---|
| Variable-coefficient Darcy | 2D/3D joint coefficient-pressure reconstruction | Positive permeability, flux consistency, gauge/boundaries |
| Coupled A+B <-> C | 2D/3D sparse multispecies reconstruction | Nonnegativity and a+c / b+c invariants for closed boundaries |
| Waves | 2D/3D displacement reconstruction | Initial velocity treatment, propagation/energy checks |
| Navier–Stokes | Dedicated 2D then selected 3D experiment | Incompressibility, boundary and pressure treatment |
| Fokker–Planck | Density reconstruction in harmonic/double-well potentials | Normalization, nonnegativity, equilibrium and flux checks |

Use real multidimensional simulations. Tiling a 1D solution along new axes is useful for an analytic operator test but is not a substantive 2D/3D model benchmark.

## 3. Baselines

Use interpolation as a simple lower reference, not the only comparator. Implement an appropriate conventional inverse baseline: parameterize the unknown initial/coefficient field and optimize a stated regularized objective using the forward solver. For the first linear heat task, a small Fourier-basis inverse problem provides a transparent baseline; report its basis/truncation/prior and noise assumptions.

When the baseline uses a different admissible family from the generator, state the mismatch. Never choose baseline hyperparameters on the test set. Ground truth numerical solver cost is reported separately; a forward solve with complete hidden inputs is an oracle reference, not a fair inverse baseline.

## 4. Measurements

- Relative L2 error on unobserved values with volume/time weights and a defined near-zero denominator convention.
- Per-field absolute error; ensemble mean and individual-sample errors.
- Observation RMSE / standardized misfit using the specified measurement model.
- PDE and boundary residuals, preferably on an independent discretization or finer grid as well as the steering grid.
- Family-specific invariants and admissibility violations before and after any correction.
- Ensemble spread, interval coverage and width on synthetic held-out cases; calibration cannot be concluded from a single reconstruction.
- Training/data generation cost; inference wall time, network evaluations, gradient evaluations, particle count, measured peak memory where available.
- FK ESS over generation steps, resampling frequency, ancestry/unique ancestors, and sample diversity.

A zero field can have a small homogeneous PDE residual and still be wrong. Verify observation fit and reconstruction simultaneously.

## 5. FK validation before PDE claims

Reproduce a tractable toy distribution with a known energy-tilted target. Verify the unsteered base model, incremental potential construction, log-weight normalization, resampling, stochastic dynamics and particle diversity. Compare to direct importance weighting and an available reference target. Repeat with particle counts/seeds and report finite-population error.

For PDE steering, state the intended tilted target, endpoint-estimation approximation, noise schedule, resampling and ESS thresholds, and all computational costs. Do not describe heuristic resampling as exact posterior inference.

## 6. Fairness, splits and generalization

- Split complete physical problems/trajectories before crops, masks, normalization or augmentation.
- Preserve independent train/validation/test seeds and a fixed test measurement set.
- Use several independent training or sampling seeds for reported comparisons; state which sources of randomness were repeated.
- Match backbone size/training effort where sensible and separately plot quality versus actual inference cost.
- Include offline prior training and data cost when claiming practical acceleration.
- Generalization levels: unseen realization; unseen parameter values; unseen combinations of supported operators; unseen family; unseen geometry; unseen grid. Label each separately.
- A successful code path at a new resolution is not validated resolution generalization.

## 7. Figures and report

For each central experiment: reference, observations, samples/ensemble summary, reconstruction error, empirical spread, and physical diagnostics. In 3D include spatial slices/isosurfaces with axis and physical-time labels, not a flattened heatmap without explanation. Share colour limits where comparisons require it.

The report should cover task definition, PDE/data generation, generative formulations, conditioning, steering derivations/approximations, baselines, validation, results, compute, limitations and reproducibility. Maintain a capability matrix with states: planned, implemented, numerically verified, smoke-tested, trained, evaluated.

## 8. First implementation acceptance checklist

Completed in the recorded first slice and continuation; see verification.md
and handoff.md for execution evidence and backend limits.

- [x] Editable install and documented CLI entry points work.
- [x] Numeric derivative/reference checks include genuine 1D–3D arrays.
- [x] 2D flow pipeline executes real optimizer updates, checkpoint save/load and guided reconstruction.
- [x] Small genuine 3D flow pipeline executes, not merely a model forward pass.
- [x] Euler/Heun and generation schedules are interchangeable without editing physics code.
- [x] Observations and PDE evaluations use physical coordinates and correct transforms.
- [x] Invalid component combinations fail before expensive work.
- [x] A conventional inverse baseline and numeric metric export exist for the first task.
- [x] Colab notebook contains package calls, configured persistent paths and resume/export instructions.
- [x] CPU tests run; CUDA/MPS status says tested or untested with actual evidence.
- [x] Actual commands, outputs, limitations and remaining roadmap are recorded.

## Executed continuation pilot

The initial acceptance slice is implemented; see verification.md for the
historical 37 tests and the continuation record. A separate bounded local
profile, configs/experiments/local_budget.json, uses 120 complete trajectories,
16x16 nodes, six physical times, width 16, two blocks and budgets 0,100,500.
One training trajectory supplies all exports. Two fixed validation problems
select among four guidance settings and two Fourier ridge values at budget
500; selected settings are frozen across two fixed test problems at every
budget. All observation sets, model/source/data/sensor seeds and split IDs are
retained. Nothing is selected using test reference error.

Executed via python -m pdeflow.experiments. Exported metrics include individual
and mean errors, observation and independent spectral-space residuals, mass,
spread, timing/calls and process RSS; figures include objective, budget/cost
quality, field comparisons and spread. Results are in docs/pilot.md. The
500-update model has lower validation loss but worse mean test error than the
100-update model. The Fourier inverse is better. These are pilot measurements,
not proof of convergence, posterior calibration or generic reconstruction.

A separate genuine 3D CPU regression and six-update MPS regression were
executed. The latter includes guidance and export/CPU reload. CUDA and Colab
remain untested. Next comparisons should repeat independent training/sampling
seeds and use more validation/test cases before interpreting causes.

## Symbolic editor and expanded heat dataset continuation

The new configurations mps_2d_pilot, mps_2d_standard and mps_3d_pilot define
2000/200/200, 10000/1000/1000 and 1000/100/100 independent train/validation/test
instances respectively. They are data-engineering targets, not completed large
studies. See local-training.md for exact 32²/16³ grids, six physical times,
float32 shards, diverse initial distributions, measured small-probe storage/cost
and user-launched generation/training/resume commands. No full large dataset or
long training was launched. Smoke profiles and accepted scientific arrays,
weights, manifests, metrics and prior budget study are preserved.

Bounded intended-layout MPS preflights used two actual updates on a separate
12-instance dataset, resume, fixed-noise validation, guidance and CPU reload.
Those are execution/resource checks, not evidence that the larger profiles
improve reconstruction. The earlier Fourier inverse advantage and non-monotonic
test-budget results stand. New quality comparisons must freeze choices on
validation and retain immutable test identities. Equation templates add editing
coverage, not new generated families or trained specialists.

## Executed Darcy integration, distinct from scientific evaluation

The multi-specialist slice ran real bounded training in
outputs/multispecialist-20261021/cpu-2d and cpu-3d before recovery. Recovery
inspected the saved history/checkpoints/metrics and did not repeat those runs.
The 2D data split is 16/4/4, grid8², width12/two blocks, batch4,20 updates.
True3D uses 8/2/2, grid8³, width8/one block, batch2,12 updates. Both resumed from
an intermediate checkpoint and ran all three guidance modes, exported figures
for a/u and reloaded for evaluation. These are integration-only smoke models.

| Layout / mode | Observation RMSE | Flux residual RMSE | Unobserved relative L2, a | Unobserved relative L2, u |
|---|---:|---:|---:|---:|
| 8² unguided | 0.448780 | 6.432866 | 0.316764 | 0.945747 |
| 8² observation | 0.115604 | 6.488939 | 0.296495 | 0.964356 |
| 8² observation+PDE | 0.115605 | 6.487851 | 0.296487 | 0.964320 |
| true8³ unguided | 0.227403 | 7.270693 | 0.366931 | 1.253283 |
| true8³ observation | 0.049893 | 7.241772 | 0.364039 | 1.251730 |
| true8³ observation+PDE | 0.049893 | 7.241595 | 0.364037 | 1.251722 |

Observation RMSE averages candidate errors at measured physical a/u points.
Residual RMSE is the cell-volume-weighted harmonic FV div(q)-f diagnostic.
Per-field relative L2 is the error of the candidate mean relative to held-out
truth on unobserved cells, with denominator floor1e-12. Metrics are in each
reconstruction/{mode}/metrics.json; reference is evaluation-only. Pressure
errors are poor; observation fit and tiny residual changes do not establish
scientific success. The aggregate two-field relative L2 hides the smaller
pressure amplitude and must be read with the per-field results.

At that historical stage the Darcy inverse benchmark was unfinished; the
22 October continuation below implements and evaluates it. No oracle forward
solve is compared as an inverse baseline. The retained heat pilot's Fourier inverse advantage and
non-monotonic test quality across0/100/500 updates remain unchanged; test
performance was not used to select a budget.

Saved MPS preflights at32² and true16³ use12 instances split8/2/2, width32/four
blocks, batch8/2, two updates each, resume, guidance, export and CPU reload.
Their preflight.json files record actual float32 MPS/backend/memory information.
These were executed before recovery and reused, not rerun. Full larger MPS or
Colab runs remain unexecuted. See multispecialist.md for exact generation,
training/resume/registration/reconstruction commands and the next chemistry
slice. CUDA/actual Colab remain untested.

## Online data / conventional inverse / chemistry task (2026-10-21–22)

The current common-provider experiments are documented in online-training.md,
darcy-study.md and chemistry.md. All outputs use the new
outputs/online-chemistry-20261021 root; earlier accepted heat/Darcy arrays,
weights, studies, screenshots and archives were preserved.

Matched four-update CPU mode trials use the same physical prior/model/batch
and fixed held-outs per family. Eight training presentations are distinct from
unique solved problems: offline prepares8 (six consumed under replacement
sampling), fresh online consumes8, hybrid solves4 and reuses4 presentations.
All use two independent calibration problems. Preparation/solve/load/transfer/
train/validation/checkpoint times are saved; common optimizer initialization
and held-out preparation are reported separately. No broad throughput claim
follows from tiny warm-process single trials; replay changes unique examples.
Earlier timing probes include cold optimizer initialization and are retained
separately; isolated final comparisons are the primary mode-cost evidence.

The Darcy study uses one0/20/80 online path and0/40/160 unique consumed problems,
four calibration examples, two independent validation and two independent test
problems, four observation scenarios per problem, noise.005, validation-only
selection and all three guidance modes versus conventional inversion. The
inverse is better at every reported budget. Permeability errors are non-monotonic;
pressure improves but remains worse. Six/eight test inverse optimizations are
budget-capped, with zero forward-solver failures. Separate a/log(a)/u errors,
per-field observations, flux/gauge/positivity, spread and calls are retained.
See darcy-study.md for exact values, hyperparameters and cost/identifiability
limits. No winning training budget was selected using test data.

Chemistry online smoke uses8 updates in2D8² and4 in true3D8³, four physical
frames, three species and the shared trainer. Real reference generation,
calibration, save/resume/export, all guidance modes, CLI/API and browser flows
execute. Reference invariant drift is at most1.35e-14/2.03e-14 with zero clipping;
model invariant errors are reported separately and are not assumed small.
Two-update MPS preflights verify16² and true8³ chemistry plus heat/Darcy online
execution, guidance and CPU reload. Full larger profiles remain user-launched;
actual CUDA/Colab remain untested. The notebook's online launch/resume cells
were executed locally without a saved training directory.

## FK analytic and learned-prior experiments — 25 October 2026

No training was needed or repeated. Primary results under outputs/fk-20261025/:
gaussian-verified/ and mixture-verified/ use four seeds and Cartesian P512/4096 ×
N16/128; heat-comparison-verified/ uses one fixed validation/test problem, three
sampling seeds and validation-only energy/gradient-scale selection. Analytic
FK at4096/128 has Gaussian mean/covariance errors.01287/.00810 and mixture
.03414/.02850, with all individual records retained. Importance can outperform
FK on easy metrics. The learned heat result is negative on unobserved error:
unguided.99853, gradient.92866, terminal importance1.27789, FK1.22522. FK's better
observation fit does not imply improved reconstruction. Root ancestry collapses
to6/2/3 of16 despite high final ESS. No test-based training-budget selection.

FK configs define a specific numerical SDE terminal tilt, not an analytic heat
posterior under a different Fourier prior. The larger P64/N64 comparison/config
is supplied for explicit user launch, not executed. Actual MPS integration also
covers Darcy2D and chemistry3D, with separate gradient heat refinement. See
fk-mathematics.md/fk-verification.md for target, costs, metrics, backend scope,
exact commands, failures and artifact paths.

## Local orchestration verification — 25 October 2026

New evidence is outputs/local-workflow-20261025/. This is a usability study,
not another scientific training-budget comparison. Actual CPU quickstart20-update
runs exercise CLI SIGINT/resume with unchanged40 consumed IDs and frozen held-outs/
calibration. All six MacBook preset layouts execute two actual MPS optimizer
updates; explicit MPS→CPU and CPU→MPS continuations retain constant Adam state.
No3000/1500/10000-update pilot was launched. A wheel executes one update outside
the checkout. Validation/default and explicit test/full reconstruction record
all gradient modes, conventional baseline, errors/costs/stop reasons. No test
result selects a hyperparameter or training budget. Existing accepted numerical
studies, model weights, screenshots and archives remain in their original roots.
See everyday-verification.md for test/browser/service records and exact limits.

### Workbench redesign continuation

The UI verification adds no scientific training-budget study. A final-source
20-update CPU quickstart stopped at1, resumed to20 and evaluated one fixed
validation case. Conventional Fourier inversion remains better. Actual metrics,
definitions and paths are in workbench-verification.md. The earlier0/100/500
test-quality non-monotonicity and fixed validation selection remain unchanged.
MPS evidence is explicitly carried forward from bounded execution; CUDA and
actual Colab remain untested.

### Reference/suite continuation — 26 October 2026

New evidence in outputs/reference-suite-20261026 exercises immutable physical
references, exact-grid noisy observation disclosure, isolated actual inverse
sampling and post-run quadrature comparison. A six-layout CPU suite executes
2 updates each, and final-source heat2D/Darcy3D execute2 MPS updates. Actual SIGINT
in the second child resumes without repeating the completed first child or
calibration/held-outs. These are bounded execution probes only. No3000/1500-update
suite was launched. The saved finite-Fourier CLI probe remains strongly negative
for the learned reconstruction versus Fourier inverse (unobserved RMSE0.71129
versus0.01514). Prior test-budget non-monotonicity remains unchanged and no test
performance chooses a budget. Definitions/artifacts/limits: reference-verification.md.

### Flow/capacity continuation

outputs/flow-workspace-20261026/ contains a real learned Gaussian CFM toy,
analytic ODE/path tests, two matched20-update32² capacity runs, one-update native
1024² and large128² MPS operation probes, real Heun candidates/snapshots and
medium64² managed MPS interruption/resume/CPU loading. These are bounded execution/
method checks, not a full training study. Medium was not better than small in fixed
validation (CFM1.53007 vs1.51396; obs+PDE unobserved relL2 1.55722 vs1.47368).
Fourier inverse remains better (.52520). Both variants and all metrics are retained;
no held-out test score chooses a winner. No3000/10000-update pilot or high-resolution
FK study was launched. Actual resource/layout/backend limits are in
capacity-resolution.md; mathematical objective/inference audit in flow-matching-audit.md.


### Catalogue/native512 bounded diagnostics — 26 October 2026

outputs/catalogue-512-20261026/ records actual bounded native512 MPS update/
guidance/sampling, managed resume and CPU inference portability. This establishes
execution only; no10000-update target or expanded full suite was trained. Existing
user six-target3000/1500-update runs and earlier studies remain unchanged.

A copy of the user's3000-update heat32 model (34372 parameters) reduced fixed-pair
velocity MSE0.0910508 to0.00903343 in100 diagnostic updates; its original checkpoint
was unchanged. On two fixed validation problems, three seeds and four particles,
mean unobserved relativeL2 with observation+PDE guidance was1.24281 at4 Heun steps
and1.28308 at8, versus Fourier inverse0.83737. Fourier remains better. No test set
or winning budget was selected. Prior distribution statistics and a tractable
Gaussian CFM probe are reported separately from paired reconstruction errors;
the earlier matched-capacity comparison is retained, not pooled. Definitions,
costs, full guidance/seed settings and limitations: [catalogue verification](catalogue-verification.md) and learning-diagnostic/diagnostics.json in that root.

### Optional latent-coordinate heat workflow — 27 October2026

outputs/coordinate-20261027/ preserves real CPU AE4/flow4, MPS AE2/flow2,
resume/portable export and full input→inverse→comparison browser execution.
This is a new optional representation; existing3000-update user heat32 and the
six-target pilot were preserved. A deterministic supervised AE is frozen before
latent CFM; no VAE or new PDE family was added.

The4-update AE's two held-out RMSEs are0.917771/0.738686, relative spectral-power
errors0.969511/0.924995. Its bottleneck is poor; finer query execution does not
establish accuracy. comparison-sensitivity retains24 records on one validation
realization, counts2/5, sigma0/.03, seeds91/92, ensembles2 and2 Heun steps at
8²/12²/32². Query32² is direct coordinate decoding; fixed-grid output at8²/12² is
explicitly interpolated. Both use exactly the same noisy measurements. Training
budgets/conditioning differ, so no architecture winner is selected. Per-sample,
mean, sensor, physical/time and cost metadata remain machine-readable. Existing
Fourier-inverse superiority and earlier non-monotonic budget results are unchanged;
no test-based budget selection or long study was performed. Detailed evidence:
coordinate-verification.md; runnable commands: coordinate-specialist.md.


### Integrated training execution checks — 27 October 2026

Only tiny fixed-grid2, coordinate AE4/flow4 and AE80/flow4 stop/resume probes
were executed. Development500 and extended budgets remain configurations,
not demonstrated adequate training. Validation is separate from test; no new
scientific budget study or accuracy advantage is claimed. See
training-workflow-verification.md for exact evidence and retained failures.

## 27 October 2026 — Colab workflow execution checks

Only bounded CPU checks were performed; no Medium/Large/XL training. Large meta
counts are AE3,235,329 and flow18,675,200 parameters; requested data would be
4,563,402,752 bytes as a materialized float32 tensor. Actual on-demand six-family
64² references were compared against128² at common nodes in float64. The six
RMSEs range 3.31e-17–8.77e-17; these unusually small errors reflect the smooth,
well-resolved generated fields and shared spectral reference, not an independent
solver validation or learned accuracy. No discretization claim for arbitrary
painted fields follows. JSON/gallery: outputs/colab-training-20261027/large-reference-gallery.*.

The sequential notebook integration runs actual setup, disposable preflight,
AE2/flow2, explicit failed-gate override where needed, persistence and portable
export. These are pipeline checks with poor scientific quality. CPU tests cover
interruption/restore/resume, extension, partial export and independent reload;
a real browser imports the notebook export and runs numerical reconstruction
without full-reference conditioning. Source and exact logs are listed in
colab-verification.md. Retained previous scientific studies remain unchanged;
conventional Fourier inversion remains better, and no budget was selected using
test performance. Next bounded experiment: first establish acceptable AE physical,
initial and spectral validation quality on a representative held-out family
mixture, then compare flow budgets using validation only. Profile on actual GPU
before spending substantial compute; an AE gate pass alone is insufficient.

## 28 October2026 — continuous heat execution and matched sampling

Only AE2/flow2 CPU and actual MPS pipeline checks were run for the new continuous
physical-time path, plus bounded test fixtures/profiling. New Large/XL capacities
and budgets remain selectable but untrained. Current notebook uniform AE MSE is
0.1089510841 versus training-fitted constant baseline0.1189131892 (ratio0.9162237157)
on six validation families. The gate now compares matching uniform quadratures;
this correction does not improve the weights or establish good initial-field
reconstruction. Mean-subtracted, initial/early/intermediate/late/final and spectral
metrics accompany whole-interval errors.

outputs/continuous-time-20261028/placement-comparison.json contains36 matched
validation reconstructions (six families, random/even/heat policies, seeds91/92,
N25, sigma.01, design/sensor time.237, four Heun steps, two particles). Policies
use the same field/noise/settings budgets; heat focus is reference-assisted.
No ranking/selection using test performance or accuracy improvement is claimed.
The existing conventional Fourier inverse results remain better. The native64²
versus128² six-family reference gallery has common-node RMSE3.69e-17–1.13e-16;
this verifies those smooth spectral representations, not arbitrary drawings or
learned accuracy. Detailed settings/evidence/failures: continuous-heat-verification.md.

## Experimental region-balanced representation objective —29 October2026

The new loss is implemented and smoke-tested, not evaluated for scientific benefit.
scripts/compare_field_losses.py plans matching MSE/WMSE/RB(α.5)/MAE representations
with unchanged architecture, physical recipes/splits, seed/time/observation policies,
optimizer and update budgets. Only --run executes; fresh roots protect existing
outputs. RB's additional region queries/analysis cost are explicit, so equal update
counts are not equal compute. Report per-region size/error, whole-field/background,
initial/time-stratum and spectral metrics on validation; keep test untouched. Saved
comparison-plan-final/ configs are unexecuted. No winning loss/α/budget selected.
Earlier Fourier inverse advantage and non-monotonic test-budget conclusions remain.
Method/evidence: field-reconstruction-losses.md and field-loss-verification.md.

## 30 October2026 — joint latent architecture execution

Only tiny2-update representation/flow runs (plus one decoded auxiliary update)
were executed for implementation. Known and unknown scalar-D inference use the
same jointly trained architecture, without a hidden initial field or coefficient.
CPU and actual MPS stop/resume, export and CPU reload passed; identical explicit
latent noise gives maximum CPU/MPS output differences2.38e-7 knownD and2.09e-7
unknownD for the retained MPS artifact. This is not accuracy/identifiability or
uncertainty calibration evidence. CUDA and actual Colab execution remain untested.

outputs/joint-latent-20261030/ contains manifests, counts, initial/final failures,
repairs, real browser workflows, portable tiny checkpoints and preservation hashes.
Large/XL are configuration/meta-count evidence only. Do not select budgets using
test performance. Historical conventional Fourier inverse superiority and
non-monotonic test-budget findings remain; no matched new-architecture accuracy
comparison was run. The next bounded experiment is representation validation
before a substantial conditional-flow training allocation.

##30 October2026 — operator-aware shared heat/wave execution

A1/A2/B were actually run with two updates each, including optimizer-boundary stop/
resume, frozen dependency checks, portable export, one checkpoint for heat/wave and
known/unknown D/c. Wave reference checks cover independent u0/v0, travelling/standing/
velocity-only/zero/Nyquist modes, energy, arbitrary coordinates and domain lengths.
The final tiny checkpoints and all evidence live in outputs/operator-wave-20261030/.

These are pipeline checks. The representation is worse than the comparable constant
baseline and reconstruction is poor; no scientific accuracy improvement is claimed.
The historical conventional Fourier inverse remains better in its recorded studies;
those studies also showed non-monotonic test quality across training budgets. They
have not been repeated/ranked for this architecture. No budget was chosen using test
performance. Region balance alpha=.5 is an experimental starting point and may trade
whole-field/background accuracy for distinct-feature emphasis. Actual CPU and MPS
execution are separate from local notebook-cell execution; CUDA/Colab remain untested.
Next: bounded controlled A1 validation, then A2 dynamics validation, before larger B.

## 31 October 2026 — bounded performance controls

No substantial training. Identical smoke-config original-source/optimized CPU
controls (7 updates each A1/A2/B, warmup2/measured5) and independent large-preset
reference-only cold/warm control (8 repeats) are recorded under
outputs/performance-20261031/. Reference-only batch medians4.083/.070 s cold,
4.023/.062 s warm;44 compared FP32 arrays identical. This is CPU preprocessing,
not requested-large-network throughput or a T4 speedup. Actual model control
parameter differences <=1.59e-6 and loss/gradient-norm differences <=2.38e-7.
Region-balanced alpha=.5 remains experimental. No model-quality improvement,
new optimum, or gate pass is inferred. See docs/training-performance.md for exact
workload, update/all-in timing differences, source hashes and T4 rerun procedure.

## Actual representation-batch sweep —31 October2026

outputs/batching-20261031/ records strict CPU gradient/loss comparisons and a
smoke-capacity FP32 cold/replay-warm sweep at effective batch32, actual8/16/32
microbatches, and an accumulated1 control. All variants keep architecture, points,
objective, distributions and frozen dependencies fixed. Seven updates/trial,
warmup2/measured5; common2-update random anchors are pipeline fixtures only.
Cold median end-to-end A1/A2/B seconds: accumulated .1545/.5611/.1434;
batched8 .0569/.0922/.0434; batched16 .0511/.0713/.0387;
batched32 .0448/.0586/.0364. Warm is a replay with valid initial-target cache files,
not a full continuous-time/dataset cache. Full raw data, setup/IO/cache/host-memory
and variability are retained. Actual MPS A1/A2/B passed; actual Medium/T4 sweep
remains unexecuted here. No scientific quality improvement or optimal batch is
inferred. docs/batched-training.md includes exact Colab source reload and commands.

## Configure → benchmark → apply → actual training —31 October2026

outputs/batch-settings-20261031/ records actual local CPU/IPython notebook execution,
bounded efficiency and learning experiments, and a real rendered MPS workbench chain.
Application changed the actual Session/request, preserved science/name/storage, and
invalidated stale preflight. Browser worker effective4 used A1/B [2,2] and A2
[1,1,1,1],2 successful updates each. The learning check held exposure at28 examples
for effective2 versus4; wall/update budget semantics and mocked OOM/overflow are
separate tests. No test-based model selection or accuracy improvement is claimed.
These are pipeline-check models only, no substantial training or CUDA/Colab run.

## Saved Medium reconstruction diagnosis —1 November2026

outputs/reconstruction-diagnosis-20261101/ records actual MPS inference for the
imported80.23M-inference-parameter Medium model (B2,500 successful updates, no skips),
three saved UI jobs, three clean-sensor validation-field controls and three exact-training-observation-policy validation cases. No main
model is retrained. Saved UI predictions replay with max difference0. Matching
A1/A2 files/gate are missing; configured A1 budget is not counted as executed.

The periodic initial-field RMSE.47558 and exact-sensor RMSE.65328 establish failure
before physical evolution. Its150 noiseless t=0 measurements have rank81 in the
81-coefficient training-band Fourier subspace and can recover that initial field;
D remains unidentifiable at t=0. The hotspot50-row system is underdetermined and
its unconstrained Fourier interpolant is unstable. More Heun steps do not repair
the model on the bounded4/8/16 probe; no step budget is selected by truth errors.

A predeclared20-step observation-only active-latent correction with trust penalty
reduces periodic sensor RMSE.65328→.09236, held-out.29059→.22481, initial-field
.47558→.17635 and domain/time.29352→.21689. All model tensor hashes and inferred D
are unchanged; conservation still fails. This is a single-case inference-time
optimization result, not retrained quality, an exact posterior or calibrated
uncertainty. No hidden references drive the optimization/stopping.

Actual Medium deterministic-head training is unexecuted without E. Its bounded
20-update tiny-fixture control lowers latent error but worsens decoded field error;
it cannot establish real conditioning quality. The noiseless input-column audit
finds exactly initialized weights, but feature ablation hardly improves error.
An explicit new-run noiseless mixture is implemented/tested, not scientifically
selected or retrospectively applied. All substantial/main-model training and new
CUDA/actual Colab diagnostics remain unexecuted. Historical Fourier-inverse
superiority and non-monotonic test-budget findings are preserved, not reranked.

### 1 November — new objective/persistence implementation, not a learning study

Retain physical-state v2 capacities and the factorized heat/wave architecture.
Add separately configurable batched clean-endpoint observation likelihood in B
(defaultbeta0), optional normalized decoded A2 residuals (defaultweight0), and
opt-in structured phase-sensitive spectral field supervision. Existing new-run
RB alpha.5 and ordinary-MSE compatibility remain. No Stage C, decoder/attention
replacement or substantial training occurred. One-factor future ablations and
held-out/ensemble criteria are specified in observation-aligned-training.md.

Actual local notebook A1=2 failed its minimum baseline gate (ratio4.0453);
no override or B progression. Partial complete artifacts restore correctly.
Separate low-level objective fixtures run A1/A2/B=2/2/2, both families per update,
with exact frozen dependencies, then fresh-process B extension2→3 for state recovery.
These are pipeline evidence, not accurate models. CPU/MPS compute-only endpoint
cost is3.55×/6.01× in tiny always-active fixed fixtures; no T4 extrapolation or
accuracy claim. Current Medium still has B2500 and missing matching E/A2.
Historical Fourier-inverse superiority/non-monotonic test budgets remain unchanged.
Full results/failures/identities: observation-aligned-verification.md.

### Standard-MSE baseline-selection continuation

The latest new-run default is Standard MSE, with explicit experimental RB alpha.5.
Only tiny CPU notebook A1 and browser A1/A2/B pipeline executions verified that
both choices reach persisted objectives; no learning comparison was launched.
Existing RB source configurations in accepted output runs/checkpoints remain
unchanged. Query measures/normalization/reductions and B objectives are preserved;
standard mode omits regional work instead of reallocating its budget. No evidence
that this default repairs poor reconstruction. See standard-mse-workflow.md.

### 1 November: A2 debugging and known sensor times

A2 before/after unchanged-weight control confirms stochastic validation/reporting
bugs, with identical training losses and weights. Fixed two-trajectory learning
experiments (120/200 and explicitly fixed-A1 200/200 updates) demonstrate nonzero
operator updates and train learned/held-initial normalized MSE ratios .98849 and
.82271. Independent validation ratios .98788/.97717 are weak and some late wave-v
errors worsen. Poor A1 and smoke quality remain; no current-Colab-run diagnosis or
accuracy claim. No model-size increase or gate bypass. Details: a2-investigation.md,
outputs/a2-investigation-20261101. A2 default sampling already has continuous whole
support plus early and exact identity; optional nonzero-conditioned policy is a
NEW experiment only.

Known sensor-time implementation adds seeded coverage/value/gradient/resume tests
and real CPU browser inference for heat random-continuous and wave-v arbitrary
explicit times. New B default differs from historical15/35/45/5 mixture. Existing
checkpoints remain unchanged. No substantial new training, CUDA/Colab execution or
scientific comparison of sensor distributions was launched. The tiny A1/A2/B=2/2/2
execution is pipeline evidence only. See measurement-times.md and verification.md.
