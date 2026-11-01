> Historical implementation record. The active heat/wave physical-state v2 workflow is documented in [operator-wave-model.md](operator-wave-model.md) and the current README. Previous whole-trajectory checkpoints are preserved and incompatible.

# Integrated training and model use

The ordinary workflow is **choose problem → choose model → choose budget → Train
model → inspect validation → Use this model**. Models & Training owns all data,
stage dependencies, paths and registration. Architecture, budgets, seeds,
discretizations, Adam settings, observation/noise distributions and checkpoint/
validation frequency remain expandable controls. Resolved configuration is shown
before submission and retained with the run. Physical time, generation steps and
optimizer updates have distinct labels. No training starts on page load.

## Local use

Activate the existing environment, run `pdeflow serve`, and in a second terminal
run `cd frontend` then `npm run dev -- --port 5173 --strictPort` using supported
Node 22.23.3 (or another supported Node version). Open http://127.0.0.1:5173.
The API is http://127.0.0.1:8000. The default workspace is outputs/local.

1. Open Models & Training. Choose the actual specialist/dimension.
2. Choose Existing fixed-grid model or Resolution-flexible model (heat only).
3. Choose Pipeline check, Development run or Extended run and a device. Inspect
   the training distribution and budget details, then click Train model once.
4. Inspect stage progress, actual updates, loss history, representation distortion,
   fixed validation reconstruction errors and reference/reconstruction/error plots.
5. Click Use this model. If necessary, its physical problem is loaded explicitly
   and incompatible measurements cleared; the previous mathematical draft remains
   available through Undo. Supply observations in Add information and reconstruct.

The model list includes both types and historical managed exports within project
outputs, without relocating weights. Finishing a budget does not select a model.
Imported exports do not acquire invented histories or optimizer state. Historical
short runs are conservatively marked execution checks; absent validation is unknown.
External workspaces can still be selected explicitly with --workspace.

## Budgets and physical contracts

These are execution/experiment configurations, not evidence of sufficient training:

| Model / preset | Actual default budget | Examples / resolution |
|---|---|---|
| Fixed-grid pipeline check | 2 updates, batch 1 | Existing provider; 4 calibration, 2 validation, 2 untouched test; 8 nodes/axis in 1D/2D, 4 in 3D |
| Fixed-grid development | 500 updates | Existing specialist pilot grid/distribution and provider counts; inspect resolved settings |
| Fixed-grid extended | Existing named preset budget: 3000 in 2D, 1500 in 3D | Original pilot settings; resource fit/accuracy not guaranteed |
| Resolution-flexible pipeline check | 4 representation + 4 flow updates | 8 training, 2 validation, 2 untouched test; 8 nodes/axis, four saved physical times |
| Resolution-flexible development | 500 + 500 updates | 64/8/8 physical realizations, 8 nodes/axis; width 32, 256 encoder/query points |
| Resolution-flexible extended | 10000 + 10000 updates | 1024/64/64; 32 nodes/axis in 1D/2D, 16 in 3D; full fit/quality unverified |

Pipeline and extended coordinate settings reuse heat-smoke.json/heat-full.json,
also packaged under coordinate/profiles. The 500-update development budget is a
bounded follow-up to the existing 500-update learning profile and the inadequate
four-update AE; it is not a scientifically validated prescription. Fixed-grid
pipeline budgets follow the existing two-update suite convention. Full expanded
train-all behaviour remains unchanged.

The form reports equation, domain, dimension, boundary, physical times and parameter
ranges. These describe a population, not the displayed reference. Heat varies
smooth initial fields and known D. Darcy varies positive permeability and compatible
forcing with a zero-mean pressure gauge. Chemistry varies positive initial species
and the recorded physical parameter ranges. Generic symbolic editing still does
not create neural support for an unimplemented family.

Auto selects CUDA if available, then MPS, then CPU. CPU, Apple GPU and NVIDIA GPU
choices are labelled with actual availability; explicit unavailable requests fail
before run allocation. Requested and resolved devices are recorded separately.
The coordinate branch is float32. CUDA/Colab execution is not verified by this task.

## Stages and the representation gate

Resolution-flexible: prepare fixed splits → train field representation (AE) →
validate representation → train conditional latent flow with AE frozen → validate
observation-only reconstruction → register completed inference artifact.
Fixed-grid: prepare/calibrate → train field-space flow → validate → register.
Both use the existing trainers and evaluation routines, not a second optimizer.

The representation gate compares held-out validation MSE against a constant
predictor fitted to the **training-field mean**. Default continuation requires a
ratio below 1. This minimal sanity check has an interpretable comparator; it is
not a universal adequacy threshold. The user can require a smaller ratio in
Advanced settings. A failed gate pauses as needs_review, displays actual distortion
and never launches flow blindly. Explicit override is recorded, including its time
when resuming. Pipeline checks have a recorded execution-only exemption so both
stages can be verified; poor representation metrics remain visible.

Validation reconstruction uses fixed held-out case IDs and sensor/sampling seeds.
Coordinate previews use two generated samples on at most two validation cases;
fixed-grid validation uses the existing quick validation case. Reference arrays
are attached after generation. No automatic test evaluation or test-based setting
selection is performed. Loss, physical RMSE, unobserved relative L2, sensor fit and
uncalibrated sample spread have distinct meanings. No new accuracy advantage is
claimed; retained conventional-inverse superiority remains unchanged.

## Simple CLI

```sh
pdeflow training start --problem heat-2d --model resolution-flexible --budget pipeline-check --device cpu
pdeflow training list
pdeflow training status latest
pdeflow training stop latest
pdeflow training resume latest
pdeflow training use latest
```

Use fixed-grid for the original model, development/extended for larger presets,
and auto/cpu/mps/cuda for device selection. Start --plan resolves settings without
launching. Start/resume --detach returns immediately; otherwise the terminal
follows progress, and Ctrl-C requests a safe stop. Status --json includes logs,
configuration, validation and the exact run ID. Use accepts a run ID or a model ID;
the workbench consumes a new explicit selection on its next load. No weights or
paths need copying. Low-level coordinate-train, train, resume, train-all and import
commands remain available.

Optional advanced overrides include --updates for fixed-grid and separate
--representation-updates/--flow-updates for the coordinate model. A poor
representation can be continued explicitly with resume --allow-poor-representation.
The UI exposes the same explicit override beside a paused run.

Fixed-grid completed runs support Train further in their model row, or
`pdeflow training extend latest --additional-updates 100`. The previous checkpoint,
inference exports, history and validation are retained as an immutable budget
snapshot; the old model remains selectable. This continues the same distribution,
not a new test-selected model. Changing coordinate stage budgets/targets is not a
supported resume operation; start a new experiment. Completed inference exports
remain portable after source changes; optimizer resume/continuation requires the
recorded source. Historical lower-level continuation commands retain their contracts.

## Persistence and ownership

Workflows are stored beneath WORKSPACE/training-workflows/RUN_ID, with a resolved
configuration, workflow.json, worker.log, child stages and validation artifacts.
This is implementation detail, not an ordinary user input. Existing managed fixed
run/model links and fingerprinted coordinate registration are reused.

One detached process owns the workflow through the existing kernel advisory lock.
It survives browser disconnects and API shutdown/restart. A submission token is
idempotent; an identical active request reuses the current run. The UI disables
duplicate submissions and ignores stale polls racing a mutation. Polls never
overlap. Initial loading is distinguished from an empty registry.

Stop writes a cooperative request file; it never signals an unverified/reused PID.
Preparation/evaluation may finish before stopping. Trainers check the request at
update boundaries and save resume state, including a zero-update checkpoint when
appropriate. A stop before a stage starts requires no checkpoint and restarts that
stage on resume. Matching owned preparation can restart after an early failure.
Stage failures stop orchestration; lock release after a crash is reported as
interrupted rather than success. No browser-lifetime job or silent retry exists.

ETA is explicitly the current learning stage only and unavailable until enough
current-session measurements exist (at least 20 timed updates and 10 seconds).
Pause time is excluded from accumulated worker elapsed time. Logs and actual
history remain inspectable after a restart.

Evidence, corrected failures and backend limits are recorded in
[training-workflow-verification.md](training-workflow-verification.md).
