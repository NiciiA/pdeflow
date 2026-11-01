> Historical implementation record. The active heat/wave physical-state v2 workflow is documented in [operator-wave-model.md](operator-wave-model.md) and the current README. Previous whole-trajectory checkpoints are preserved and incompatible.

# Resolution-flexible heat specialist and input workflow

New heat2D defaults use continuous physical[0,1] and direct arbitrary-time
references; see [continuous-heat.md](continuous-heat.md) for current settings and
legacy migration. Earlier snapshot examples below remain legacy contracts.

This optional architecture adds learned latent CFM beside the existing fixed-grid
CNN/U-Net specialists. It supports scalar unforced periodic heat, one declared
spatial dimension1/2/3 per checkpoint, one exact Cartesian domain/origin, known
constant diffusivity within the recorded range and physical times within the
trained interval. It does not add Darcy/chemistry latent models, arbitrary forcing,
coefficient fields, geometry/dimension/equation transfer, exact densities or a
calibrated posterior. Fixed-grid checkpoint checks remain strict.

## Local operation

From the repository root:

```sh
source .venv/bin/activate
pdeflow serve
```

Second terminal, with supported Node22 on PATH: `cd frontend` then
`npm run dev -- --port 5173 --strictPort`. Open http://127.0.0.1:5173;
API http://127.0.0.1:8000. Startup and refresh never train. Use the already-created
verification workspace instead with:

```sh
pdeflow serve --workspace outputs/coordinate-20261027/workspace
```

It contains CPU4-update AE/4-update flow and MPS2/2 smoke stages and links the
unchanged user3000-update heat32 baseline. These checkpoints establish execution,
not scientific model quality. The actual stages/metrics are visible in Models &
Training → Latent coordinate heat specialist. Original fixed-grid run views remain.

## Three explicit input modes

In Add information choose:

1. **Analytical / manufactured reference.** Heat's initial u(x,0) and complete
   reference u_ref(x,t) are separate editors. x/y/z are physical coordinates; t is
   physical time; D is known diffusivity. Review a template, edit, verify, solve,
   and inspect the field before generating observations. Known analytical mode
   rejects inconsistent candidates. Manufactured mode accepts finite reference
   functions with residual/initial/periodic diagnostics and an unverified label.
   No forcing is invented: forced heat is outside this specialist's capabilities.
   Safe allowlisted parsing never evaluates arbitrary Python. Other families use
   their genuinely implemented numerical reference solvers.
2. **Numerical reference.** Supply initial fields, parameters and supported forcing
   separately. Heat uses exact mode evolution/numerical spectral interpolants, so
   it has output times rather than an artificial solver time-step setting. Darcy
   exposes CG tolerance/iteration controls; chemistry exposes max_step/iterations.
   Reference grid/output times are separate from model/query settings. Plotting
   strides to at most128 nodes per visible axis without changing numerical arrays.
3. **Measurements only.** Enter/import JSON measurement rows, including measured
   zeros. Missing rows mean unmeasured. Adopt without a reference: reconstruction
   and supported physics diagnostics remain; truth error is unavailable. Existing
   fixed-grid row imports require exact nodes/saved times. The coordinate panel
   separately accepts scattered scalar sensors with explicit absolute sigma.

Reference numerical JSON has exactly `problem`, `coordinates` (named axes),
`physical_times` (null for static), ordered `fields`, `values` in
`[field,time,*spatial]`, `units="nondimensional"`, and `provenance`. Coordinates
must be unique, increasing, uniform and exactly consistent with ProblemSpec.
Axis inference/transposition, duplicate endpoints and silent extrapolation are
rejected. UI uploads are bounded by the existing2MB request limit; CLI handles
larger files within the explicit field budget. Export reimportable field JSON from
the reference UI, or:

```sh
pdeflow reference --preset heat-2d --output outputs/my-reference
pdeflow reference-export --reference outputs/my-reference --output outputs/my-reference-field.json
pdeflow reference-import --input outputs/my-reference-field.json --output outputs/my-imported-reference
pdeflow observe --reference outputs/my-reference --sampling configs/coordinate/observations.json --output outputs/my-observations
```

Generate observations displays its total before execution. Fixed/regular/random
spatial count means locations per selected field and saved physical time; fixed
sensors reuse one spatial population. Space-time count means total scalar points.
Positions/reuse count means explicit row count. Random discrete selection is
without replacement within the declared population and rejects oversubscription.
Separate seeded sensor/noise streams preserve reproducibility. Advance the seed
with Resample, then explicitly generate. Field and time selection, noisy-value
preview/table, overlays and CSV/JSON exports retain coordinates and actual sigma.

Noise is independent Gaussian, either absolute sigma or magnitude times full clean
reference field RMS at each selected physical time. Zero reference RMS gives
sigma0. Clean values remain a separate evaluation-only artifact. Per-observation
absolute sigma is authoritative through import, guidance, diagnostics, Fourier
weighted inverse and refinement; scalar noise metadata is an absolute fallback
summary, not a relative factor. No correlated-noise or calibration claim is made.

## Two resumable training stages

Small CPU development commands (4 updates per stage):

```sh
pdeflow coordinate-train --config configs/coordinate/heat-smoke.json --stage autoencoder --output outputs/local/coordinate/heat-ae-smoke
pdeflow coordinate-evaluate-ae --checkpoint outputs/local/coordinate/heat-ae-smoke/autoencoder.pt --output outputs/ae-smoke-validation.json
pdeflow coordinate-train --config configs/coordinate/heat-smoke.json --stage flow --autoencoder outputs/local/coordinate/heat-ae-smoke/autoencoder.pt --output outputs/local/coordinate/heat-flow-smoke
```

Substantive user-controlled MPS configuration (not launched by Codex):

```sh
pdeflow coordinate-train --config configs/coordinate/heat-full.json --stage autoencoder --output outputs/local/coordinate/heat-ae-full
pdeflow coordinate-evaluate-ae --checkpoint outputs/local/coordinate/heat-ae-full/autoencoder.pt --output outputs/ae-full-validation.json --device mps
pdeflow coordinate-train --config configs/coordinate/heat-full.json --stage flow --autoencoder outputs/local/coordinate/heat-ae-full/autoencoder.pt --output outputs/local/coordinate/heat-flow-full
```

Evaluate the bottleneck before spending the latent-flow budget. The full profile
uses32²/T4,1024 training/64 validation/64 test physical realizations,64-dimensional
32-token latents and10000 updates per stage. Its resource fit/full quality is
unverified; the actual MPS probe used8² and two updates per stage. Change config
runtime explicitly for CPU/CUDA; no hidden fallback. Notebooks can install the
same package and call these commands with configurable persistent output paths.
Actual Colab/CUDA execution remains untested.

Resume either interrupted stage with its original config/output and `--resume`;
flow dependencies are recovered from its checkpoint:

```sh
pdeflow coordinate-train --config configs/coordinate/heat-full.json --stage autoencoder --output outputs/local/coordinate/heat-ae-full --resume
pdeflow coordinate-train --config configs/coordinate/heat-full.json --stage flow --output outputs/local/coordinate/heat-flow-full --resume
pdeflow coordinate-runs
```

Resume restores Adam, CPU/source RNG, progress and histories, and requires the
recorded source/config/data/normalization/dependency identities. SIGINT saves at a
safe update boundary; `--stop-after N` supplies a bounded per-session check.
Inference does not impose this source restriction or require training data.

The reference encoder is trainable set-to-latent attention; the coordinate decoder
is trainable cross-attention plus MLP. Stage1 minimizes mean normalized point MSE
per physical realization. Stage2 freezes both, fixes training-only latent affine
scaling, then trains a transformer velocity with observation-set cross-attention.
No unconstrained moving target or latent-collapse shortcut is added. Physical
normalization and latent scaling use training data only. Underlying physical
realizations are split before any count/noise observation variants. Validation
noise/source/time probes are fixed. First flow update exposes the zero-observation
path, second a nonzero count, then the configured count/noise/dropout distribution.
There is one reference discretization per checkpoint; multi-reference-resolution
training is not claimed. Losses average per problem, rather than weighting by
number of query points.

`dataset.json/npz`, `config.json`, `run.json`, `history.json`, `resume.pt` and
completed `autoencoder.pt`/`coordinate.pt` retain data/split identities, seeds,
source, architecture, stages, normalization and dependencies. Full history remains
in history.json/checkpoints; run metadata shows its latest200 rows with total count.
Live metadata is throttled to one second plus checkpoints, and liveness uses the
existing kernel lock. Preparing/failed/interrupted states are distinct. No browser
owns the training process. Completed budget is not scientific evaluation.

`train-all` continues the explicitly documented eleven-target fixed-grid catalogue.
These two optional architecture stages are explicitly selected with coordinate-train;
no large extra experiment is silently scheduled.

## Reconstruction, queries and portable registration

Managed stages under `WORKSPACE/coordinate/<run>` are discovered automatically.
A completed portable export elsewhere can be linked without moving weights:

```sh
pdeflow coordinate-import --checkpoint /path/to/coordinate.pt
pdeflow coordinate-runs
```

Import records its content fingerprint and capability/stage metadata. Changed or
invalid exports reject with an actionable error. Inference exports embed the
frozen encoder/decoder; standalone CPU inference needs neither the original AE
path nor training data. Optimizer resume/AE evaluation still needs its original
stage/data. Storage inventory protects stage dependencies and active runs; existing
`pdeflow storage --refresh`, `pdeflow cleanup --preview` and pin/review-digest
quarantine controls apply. No existing outputs were deleted or relocated.

In Reconstruction select the separate coordinate specialist checkpoint and query
grid, or supply ordered scattered `[x,(y,z),physical_time]` queries. Explicit sensor
rows may be off-grid. `[]` selects the trained zero-observation prior. Multiple
reconstructions are an ensemble, not extra measurements. Query chunking preserves
order and numerical output within floating-point tolerance. The decoder evaluates
the same learned function; a finer plot does not establish learned fine-scale
physics. Coarse/native/fine/scattered queries were actually executed.

```sh
pdeflow coordinate-reconstruct --checkpoint outputs/local/coordinate/heat-flow-smoke/coordinate.pt --problem outputs/my-observations/problem.json --observations outputs/my-observations/rows.json --query-shape 12,12 --query-times 0,.025,.1,.15 --output outputs/my-coordinate-result --steps 12 --particles 4 --chunk-size 256 --device cpu
```

For scattered queries replace the regular query flags with `--queries queries.json`.
Reference values never enter this command's schema or worker. Attach a reference
only after success in the UI. Reference/reconstruction share physical axes and
color limits, with separate error/spread views and per-time/3D slice controls.
Visible/off-slice observations are distinguished. Saved jobs reopen through Runs;
source/settings and job IDs persist across refresh. Error metrics are absent when
no reference is attached.

## Reproducible comparison

Use fresh output paths. The following is the actually executed bounded sensitivity
configuration, comparing the saved smoke with the preserved user baseline:

```sh
pdeflow coordinate-compare --coordinate outputs/coordinate-20261027/workspace/coordinate/heat-flow/coordinate.pt --baseline outputs/local/runs/20261026T171753-local-pilot-281503ff-heat-2d/training/inference.pt --output outputs/my-coordinate-comparison --counts 2,5 --noise 0,.03 --resolutions 8,12,32 --seeds 91,92 --cases 1 --particles 2 --steps 2 --device cpu
```

For substantial validation, supply your trained exports and deliberately matched
training distributions/data/compute; increase cases/seeds/flow steps explicitly.
This command does not train. It uses the same held-out physical problems,
coefficients, actual noisy measurements and seeds for both methods, records
parameter counts/checkpoint/data/training identities and exports samples/queries,
machine-readable comparison.json and report.md. Unmatched training budgets and
conditioning methods are explicit; these smoke results do not rank architectures.

Metrics include per-sample RMSE/MAE/relativeL2, separate ensemble-mean error/spread,
per-time/initial error, observation RMSE against noisy values at actual sensor
coordinates, clean-reference sensor error after inference and common query-grid FD
heat residuals. RelativeL2 is null when reference RMS<=1e-10. Absolute error and
reference norms remain. Fixed-grid output at other resolutions is explicitly
**baseline plus periodic multilinear interpolation**. Reference transfer away
from its original nodes is also approximate and labelled; neither is fine-grid
truth. No extrapolation is allowed. Periodic endpoint equality is imposed by
coordinate features/baseline periodic interpolation, not learned boundary accuracy.
Unknown initial conditions are never enforced through truth during generation.

Memory records process-lifetime RSS high-water mark, CUDA allocated peak when reset
and measured, and MPS allocation after execution. MPS values are not peaks; shared
process RSS cannot isolate architecture cost. No calibrated coverage metric is
invented. The saved smoke AE has large held-out errors/spectral distortion, so
query execution is not evidence of an accurate learned solution manifold.

The coordinate panel has its own explicit float32 device selector. It defaults to
CPU; MPS/CUDA requests validate availability and never silently fall back. Auto
selection is explicit and the resolved device is recorded in metrics. Its settings
are separate from the fixed-grid generation panel below. New training checkpoints
also accumulate actual host training/setup/reference costs across resume; missing
older cost fields remain null in comparisons rather than fabricated estimates.
