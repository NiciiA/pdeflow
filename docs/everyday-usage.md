# Everyday PDEFlow

Use the short commands here for local work. The original explicit-config/data/
checkpoint CLI remains available in advanced-workflows.md, local-training.md,
online-training.md and fk-verification.md. The package still owns every numerical
algorithm; the local layer prepares paths, manifests, locks and calls it.

## Install once

Use Python3.12 and the Node version in `.nvmrc` (22.23.3). Activate a usable
existing virtual environment; do not recreate it. On a first installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[workbench]"
pdeflow setup
pdeflow doctor
```

Setup creates project-local directories and runs `npm ci` with the committed
lockfile. Rerunning preserves settings, checkpoints and runs; npm ci replaces
only generated frontend dependencies. It never trains or downloads scientific
datasets. Node/Python/Homebrew are not installed globally. If you use nvm,
`nvm install` in the repository reads .nvmrc; otherwise install a supported Node
through your usual installer and ensure node/npm are on PATH. Installed Vite,
Vitest and ESLint engine constraints intersect at **^22.13 || ^24 || >=26**;
the package engines, Python check and frontend preflight agree. Node22.12 meets
Vite but misses the installed ESLint requirement, so it is rejected early.

For headless Python/Colab use `pdeflow setup --headless`; missing Node or frontend
dependencies do not block training. `pdeflow doctor --json` gives support data.
Missing CUDA on a Mac is informational, not a general failure. Full precision
float32 is the normal training setting; explicit unavailable MPS/CUDA fails.

## Train, stop and resume

```bash
pdeflow presets
pdeflow train --preset heat-2d --device mps
# Ctrl-C, then in this or a later activated terminal:
pdeflow resume latest
pdeflow status latest
```

The preset command prepares deterministic fixed validation/test problems, fits
frozen transforms from a separate training-only calibration namespace, then
starts the same online trainer. No full saved training dataset or manual export
is required. CPU reference generation is explicit even when the model uses MPS.
Progress prints completed updates, losses, validation update, measured data wait,
throughput including preparation and actual checkpoint path. No unmeasured ETA
is promised. Structured events and training history persist separately.

A short complete check is `pdeflow train --preset quickstart-heat-2d --device cpu`.
Other normal choices:

```bash
pdeflow train --preset darcy-2d --device mps
pdeflow train --preset chemistry-2d --device mps
pdeflow train --preset chemistry-3d --device mps
pdeflow train --preset heat-2d --device mps --dry-run
```

Available overrides: --device, --steps (total budget), --batch-size, --seed,
--name, --workspace, --output, --data-mode online/offline/hybrid. `--plain` uses
plain interval summaries, also the default. --stop-after caps a single session
without changing its recorded total. A dry run validates the resolved setup and
prints stages without creating a run or claiming measured duration. No setting
silently reduces a physical grid or changes an equation after memory failure.

Ctrl-C during training is deferred to an update boundary, followed by an atomic
resumable checkpoint and provider cleanup. It prints the actual saved update.
A hard kill loses work since the last complete checkpoint; completed logical
sample IDs and RNG/provider cursor are restored. No background training producer
exists. A preparation interruption can be retried with resume: completed physical
records and a completed frozen calibration are reused. A partial calibration
before its frozen file exists may repeat deterministic calibration solves.

Use an ID or path instead of latest to select explicitly. `latest` means the
most recently **started** managed training run in the selected workspace, with
lexical ID breaking a timestamp tie. It is not the most recently updated or
best run and does not skip a completed/failed run for an older one. Resolution
prints the exact specialist, ID and path once. `--specialist heat|darcy|chemistry`
can restrict status/resume/evaluate selection; an explicit mismatched run fails.

```bash
pdeflow runs
pdeflow resume RUN_ID
pdeflow resume RUN_ID --device mps
pdeflow resume RUN_ID --additional-steps 1000
```

Additional steps add to the **recorded total**, not the current progress. The
existing constant Adam learning rate and optimizer state continue; there is no
scheduler to reset or rescale. All other scientific/model/provider settings and
precision stay fixed. Explicit device remapping is supported, but cross-device
bitwise equality is not promised. Resume checks source identity: old-source
checkpoints remain loadable for inference, but training with changed source is
rejected. Keep the source recorded by a long run. No force-resume flag rewrites
an old architecture or normalization.

## Presets and evidence

All profiles use actually implemented periodic specialists and online physical
reference generation. Transient times below are0,.05,.1,.15; Darcy has one static
frame with no physical-time evolution. Every 2D/3D choice has distinct spatial
axes. Full default pilot budgets were not launched for orchestration testing.
See everyday-verification.md for actual bounded backend execution.

| Preset | Specialist | Grid / saved frames | Mode | Width / blocks | Batch | Updates | Save interval | Resource purpose |
|---|---|---|---|---|---:|---:|---:|---|
| quickstart-heat-2d | Heat |8² /4|online|8 /1|2|20|5|Quick check, CPU or MPS|
| heat-2d | Heat |32² /4|online|24 /3|2|3000|100|MacBook pilot|
| darcy-2d | Darcy a,u |32² /static1|online|24 /3|2|3000|100|MacBook pilot|
| chemistry-2d | A,B,C |16² /4|online|24 /3|2|3000|100|MacBook pilot|
| heat-3d | Heat |16³ /4|online|16 /3|2|1500|100|Explicit small3D pilot|
| darcy-3d | Darcy a,u |16³ /static1|online|16 /3|2|1500|100|Explicit small3D pilot|
| chemistry-3d | A,B,C |8³ /4|online|16 /3|2|1500|100|Explicit small3D pilot|
| colab-heat-2d | Heat |64² /4|online|48 /4|16|10000|250|Larger CUDA profile, untested|

Defaults resolve auto to CUDA, then MPS, then CPU and print the choice, except
the explicit colab profile defaults to CUDA. Use --device cpu/mps for deliberate
selection. Online calibration/validation/test counts are4/2/2 for quickstart,
8/8/8 for MacBook pilots and16/16/16 for Colab. Offline mode automatically
prepares the configured finite training set; hybrid uses bounded replay and
records reuse. Mode changes are allowed at fresh launch, not mid-resume.
No preset promises useful scientific accuracy or a measured full-run duration.

## Backend and frontend separately

```bash
# Activated Python terminal:
pdeflow serve
# Separate terminal, from the repository root:
cd frontend
npm run dev
```

API http://127.0.0.1:8000; UI http://127.0.0.1:5173. The backend never starts the
frontend or training. Vite proxies /api to the ordinary backend. Advanced
`VITE_API_URL=http://127.0.0.1:8001 npm run dev` selects another backend and updates
the displayed expected URL. Without a backend, the UI says “Backend unavailable.”,
shows the actual expected URL and `pdeflow serve`; refresh reconnects and discovers
new checkpoints without replacing the current mathematical draft.

Serve accepts --host, --port, --reload and --workspace. Keep localhost for normal
use; there is no remote authentication/deployment layer. Direct invocation:

```bash
python -m uvicorn pdeflow.workbench.api:create_app --factory --host 127.0.0.1 --port 8000
```

For an explicit workspace with direct uvicorn, set PDEFLOW_WORKSPACE for that
advanced invocation. `pdeflow serve --workspace PATH` needs no environment setup.
The optional `python scripts/dev.py` still starts/stops both components. Uvicorn
reload watches source only; model creation and the numerical queue begin inside
the worker lifespan. A kernel lock on the job root prevents duplicate queue
owners. Graceful shutdown closes the queue and joins workers; unfinished durable
jobs become interrupted. A crash is reconciled on the next server startup.

## Discovery and the workspace

The default is **outputs/local/** beneath the source project root. It contains
runs/, cache/ and workbench/{jobs,data}/. Generated files stay ignored by Git.
A different workspace is explicit with --workspace PATH or PDEFLOW_WORKSPACE;
use the same choice for train/runs/resume/evaluate/serve. Source-installed
commands find the project independently of cwd; a standalone wheel uses the
selected workspace (or cwd when no source project is discoverable).

Each run has run.json, config.json, fixed evaluation, training logs/history,
checkpoint.pt and portable inference exports. run.json is durable filesystem
truth; runs/model discovery is rebuilt from it, not a central database.
An external --output directory is referenced in the workspace's run-links.json;
the external run manifest is still authoritative and can be selected directly
if that reference file is lost. Never move accepted QA outputs automatically.

A checkpoint save publishes **inference.pt = latest saved update** and
**best-inference.pt = minimum fixed validation CFM loss** (ties keep the earlier
update). The best export is unavailable until a validation occurs. Test error
never selects it. API model labels show run/name, specialist/layout, update,
latest/validation-selected role and quick-check/pilot status. Identical content
IDs are deduplicated. Partially written/invalid exports are excluded. A completed
budget is still scientifically unevaluated. Equation/full-request compatibility
continues to gate execution independently of discovery.

For old QA outputs or a checkpoint transferred from Colab:

```bash
pdeflow models --import PATH_TO_INFERENCE_PT_OR_DIRECTORY
pdeflow models
```

This stores references without copying/moving or modifying external artifacts.
Advanced PDEFLOW_MODEL_ROOTS/PDEFLOW_JOB_ROOT/PDEFLOW_DATA_ROOT overrides remain
supported. Explicit model roots take precedence over managed model discovery.

## Evaluate without remembering internal paths

```bash
pdeflow evaluate --run latest
pdeflow evaluate --run RUN_ID --export best
pdeflow evaluate --run RUN_ID --split test
pdeflow evaluate --run RUN_ID --split test --full
```

Default is **one validation case** with the run's recorded inference settings,
all existing gradient comparison modes and implemented conventional inverse.
It does not tune settings. --full uses every case in the explicitly selected
split; it can cost substantially more. --split test is explicit and records
its use; repeatedly inspecting test scores must not become hidden tuning.
Every evaluation gets a new directory, selection record, per-field metrics,
physical residuals, reference errors, cost/stop reasons and baseline diagnostics.
Reconstruction finishes before truth is opened. FK/controller/refinement controls
remain available in matching advanced configs/workbench; no mathematics is
reimplemented in the local layer. The old `evaluate --result PATH [--reference]`
continues to evaluate an already exported result. Legacy explicit datasets have
only saved test observation artifacts, so managed evaluation requires --split test.

## Diagnose and recover

| Symptom | Action |
|---|---|
| MPS explicitly unavailable | Inspect `pdeflow doctor`; choose --device cpu explicitly or use a host with Metal access |
| Frontend Node rejected | Use .nvmrc Node22.23.3, then `pdeflow setup`; headless training remains available |
| API offline | `pdeflow serve`, then Refresh checkpoints |
| Port in use | Stop its owner or `pdeflow serve --port 8001` with matching VITE_API_URL |
| No latest run | `pdeflow runs --workspace PATH`, or start a named preset |
| Preparation failed | `pdeflow status RUN`; `pdeflow resume RUN` reuses completed work |
| Another process owns run | Wait/stop the reported owner; do not delete a live lock |
| Stale lock metadata | OS releases the lock after process exit; resume reacquires it and reports the real state |
| Incomplete checkpoint | Keep the last valid atomic checkpoint; inspect status/log and recover with recorded source |
| Source/config mismatch | Use the recorded source or start a new run; no silent normalization/architecture migration |
| No discovered model | Wait for a complete checkpoint, refresh, check workspace or use explicit models --import |

Run/queue locks are advisory filesystem locks intended for a single host; shared
multi-host/network-filesystem orchestration is outside this local workflow.
CUDA and actual Colab execution remain untested. The notebook's preferred final
section uses these same train/resume/evaluate commands with a persistent workspace.

## Workbench navigation

Workspace exposes Define problem, Add information and Reconstruct & inspect,
with a persistent mathematical request map. Runs holds immutable results and
compatible comparisons. Models & Training shows discovered exports, actual
managed optimizer/validation histories and CLI commands. Drafts survive browser
refresh. Input edits retain previous results with an explicit stale-input notice.
See [the workbench journey](workbench.md#everyday-workspace-october-25-redesign).

## Complete references, measurements and training collections

[Reference experiments](reference-experiments.md) gives the shortest compatible
checkpoint → reference → disclosure → reconstruction → comparison workflow and
all low-level CLI commands. [Training suites](training-suites.md) defines the
finite train-all collection, sequential resource control and durable resume.
These extend the existing managed workflow and do not replace online/offline/
hybrid training, current symbolic routing, FK/controller or refinement.
A suite quickstart retains its pilot's grids; use `quickstart-heat-2d` (8²) for
the bundled reference examples, or explicitly edit the reference grid to the
trained layout. Fixed-grid checkpoints never silently change resolution.

## Current linked flow workspace and larger capacity

See flow-workspace-handoff.md for exact medium/large/native1024 profiles, resume,
matching reference experiments, suite plans, Colab export/transfer and verification.
The new visual centre links point/field/equation and shows actual generation
snapshots separately from physical evolution. Synthetic comparisons associate
automatically after inference. Clear workspace resets only the current persisted
editor/selection, with explicit running-job choices; saved data is retained.
No compatible export is ever replaced by an empty/random ID or unrelated model.

## Current expanded training and native 512 workflow

The current default train-all covers the full finite 11-target catalogue. Use
--suite local-pilot for the original six. Native MacBook heat: `pdeflow train
--preset heat-2d-512 --device mps`; planning: `pdeflow train-all --plan`; missing
coverage: `pdeflow train-all --missing --plan`; actual status: `pdeflow coverage`.
Models & Training now reads live CLI progress, frozen suites, current coverage,
configs/logs and cached storage. It never starts long jobs merely by opening.
Reference experiment contains separate editable analytical initial/reference
expressions and verification. See training-catalogue.md for exact workflows and
the explicit native-grid feedback-budget review. Automatic routing now prefers
more optimizer updates, then completed-budget/export provenance, within exact
compatible candidates. This is effort priority, never truth-selected quality.


## Optional coordinate model / source modes — 27 October2026

The three source modes, explicit observation counts/noise, full-field import,
AE→latent-flow training/resume, checkpoint registration, query resolution and
comparison commands are in [coordinate-specialist.md](coordinate-specialist.md).
The fixed-grid commands above retain their meaning. No long training is started
by selecting a source, opening Models & Training or refreshing discovery.
