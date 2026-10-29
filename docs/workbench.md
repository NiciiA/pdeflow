# Local scientific workbench

The React application calls a localhost FastAPI service. Algorithms remain in
pdeflow; neither browser nor API reimplements training, PDE residuals or
numerical sampling. Opening the UI never starts training. This is a local
research tool, without accounts, hosting, credentials or remote services.

## Install and launch

Use the first-time instructions in [README](../README.md): Python3.12,
`pip install -e ".[workbench]"`, then `pdeflow setup`. Node22.23.3 is selected in
.nvmrc; frontend engines support ^22.13, ^24 or >=26. Headless training does not
need Node.

```bash
source .venv/bin/activate
pdeflow train --preset quickstart-heat-2d --device cpu
pdeflow serve
# In a separate terminal, from the repository root:
cd frontend
npm run dev
```

Open http://127.0.0.1:5173; API http://127.0.0.1:8000. Locally trained exports
are discovered from outputs/local automatically, with run/name, grid/time,
latest/validation-selected role, update and scientific-status labels. Refresh
checkpoints without replacing the draft. No model/data/job environment variables
are needed. Select the compatible equation, review its checkpoint layout,
generate/import observations and reconstruct. See everyday-usage.md.

`pdeflow serve --workspace PATH` selects another persistent workspace. Direct
Uvicorn is `python -m uvicorn pdeflow.workbench.api:create_app --factory --host
127.0.0.1 --port 8000`. The optional `python scripts/dev.py` still starts both
components and stops them on Ctrl-C. The frontend displays an actionable
Backend unavailable message with the expected API URL and `pdeflow serve`.

Existing accepted QA checkpoints remain accessible through an explicit link:

```bash
pdeflow models --import outputs/accepted_cpu_2d/training/inference.pt
```

Advanced root overrides are preserved, without moving old outputs:

```bash
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training"
export PDEFLOW_JOB_ROOT="outputs/my-workbench/jobs"
export PDEFLOW_DATA_ROOT="outputs/my-workbench/data"
pdeflow serve
```

The separator is the host's path separator (colon on macOS/Linux). All paths
are server configuration, never browser-controlled. Model discovery reads
server-owned inference.pt files with weights_only=True on CPU and exposes
opaque content IDs. Invalid checkpoints are skipped; the UI presents an empty
state if no model is registered. A browser cannot upload a pickle or supply
an arbitrary model/output path. No server deployment or access control is
provided; keep binding to localhost.

## Workflow and exact semantics

The primary Problem UI is now the symbolic editor documented in
[symbolic-editor.md](symbolic-editor.md). Begin with an empty equation or optional
preset, not a mandatory family selector. Edit 𝒞=(𝓔,λ,Ω,ℬ,ℐ,y,M); the backend
parses current source and exposes an executable ProblemSpec only for supported
heat/Darcy/chemistry requests. The complete-request JSON/export/import replaces editable legacy
operator JSON; parsed nodes and ProblemSpec are separate read-only views. Preset
and checkpoint-layout conflicts require a visible review and preserve existing
measurements, boundary notes and common parameter values. Add a Heat equation
before using a checkpoint layout. Dataset/MPS commands are in local-training.md.

1. Problem: insert/edit equations, then route automatically or choose an explicit compatible override and
   explicitly review/apply its layout if desired. Edit declarations, diffusivity, dimension, domain, grid, saved physical
   times and boundary information. Structured fields and advanced JSON update
   the same draft. The readable equation renders the allowlisted operator
   tree. MathLive exports LaTeX through the documented bounded PDE parser;
   unrestricted LaTeX execution is unavailable. Invalid source/JSON remains
   editable and blocks launch.
2. Validate problem syntax/residual support. A valid scalar expression can be
   unsupported by the heat checkpoint. Darcy requires explicit known forcing and a compatible static two-field
   checkpoint; chemistry uses its real three-species specialist. Full request validation checks exact family,
   operator, layout, grid, domain, boundary, conditioning, formulation,
   stepper/schedule, device and dtype. Recorded diffusivity range comes from
   the saved data configuration; out-of-range values produce an explicit
   unvalidated-generalization warning.
3. Observations: add/edit/delete manual rows or import CSV/JSON. Required
   columns are field,x[,y,z],time,value for heat, and
   field,x,y[,z],value for static Darcy (physical a/u). JSON is an array of objects using these
   names. CSV uses the same header. Export in either format. Noise is a stated
   independent Gaussian standard deviation; the positive guidance noise floor
   tempers its energy as documented in mathematics.md.
4. Coordinates must match saved periodic nodes and saved physical times.
   Endpoint duplicates, off-grid/time interpolation, unknown fields, missing
   or extra columns, duplicates/conflicts, nonfinite values and empty sets are
   rejected. Off-grid interpolation remains unavailable. Values and coordinate
   names are distinct; measured zero is not missing.
5. The explicit synthetic action generates a numerical reference and noisy
   sensors. Only known metadata and measurement rows are sent to inference.
   Detach the reference before running to exercise standalone reconstruction.
   After success it may be attached solely for evaluation, or removed again.
6. Configure seed, independent sample count, backend, stepper, generation
   schedule/steps and guidance. Advanced weights/clipping/noise-floor controls
   are available. Compatible current requests also expose FK/terminal importance
   and their explicit population controls. Validate, review warnings/storage
   and run. Duplicate submission is prevented in the UI and by a server
   idempotency token. Export the resolved request.
7. Results: select physical time and sample/mean/empirical standard deviation.
   1D uses curves; 2D uses labelled heatmaps with observation dots; 3D uses
   orthogonal slices with normal-axis/index/coordinate controls. Reference and
   reconstruction share a field colour scale. Error is unavailable without an
   explicitly attached reference; spread is not calibrated uncertainty.
8. Download samples.npz, metrics.json and configuration.json. Provenance shows
   actual seed, backend, model/gradient calls, stop reason and settings. Runs
   persist on disk and can be reopened after browser refresh.

## Job lifecycle and resource limits

One spawned compute process runs at a time. API request handling remains
responsive. Persistent atomic JSON records move through queued, running,
succeeded/failed/cancelled/interrupted. Progress records actual completed
sampling steps; loading/export are named stages. Sampling covers 0–90%, export
95%, success 100%. The percentages are stage fractions, not an ETA. Cancellation
terminates and joins the worker before marking the job cancelled. A crashed
worker becomes failed with an error. Server startup/shutdown marks unfinished
jobs interrupted rather than leaving them running indefinitely. No job resume
is claimed; training resume remains a separate CLI capability.

Requests are limited to 2 MB, 20000 measurements, 16 samples, 512 generation
steps, eight CPU threads and 128 MiB of raw sample storage. These bounds do not
predict full activation/GPU memory. Preview slices contain at most 65536 nodes;
the API does not serialize full 3D ensembles as JSON. Files remain under the
configured roots with UUID IDs and an explicit artifact-name allowlist.
Do not expose this development service to an untrusted network.

## API map

- GET /api/mathematics/presets; POST /api/mathematics/parse
- Browser reconstruction includes mathematical_request; validation and submit
  reparse it and verify ProblemSpec/measurement/noise agreement. Existing direct
  ProblemSpec clients remain supported.

- GET /api/health, /api/capabilities, /api/models
- POST /api/problems/validate
- POST /api/observations/validate (rows, or text plus csv/json format)
- POST /api/demos (explicit synthetic reference-data action)
- POST /api/reconstruction/validate
- POST /api/jobs; GET /api/jobs and /api/jobs/{id}
- POST /api/jobs/{id}/cancel
- GET /api/jobs/{id}/result and /api/jobs/{id}/slice
- POST /api/jobs/{id}/reference (post-inference evaluation attachment/detachment)
- GET /api/jobs/{id}/artifacts/{samples.npz,metrics.json,configuration.json,figure.png}

Interactive schemas are at http://127.0.0.1:8000/docs. ReconstructionRequest
forbids reference fields and arbitrary extra top-level keys. The worker entry
point has no reference argument. The CLI also completes sampling before
opening evaluation truth. Job roots retain resolved requests, checkpoint IDs,
seeds, worker logs, output arrays, metrics, problem/observation metadata and
environment. A process lifetime RSS high-water mark is labelled as such.

## Verification commands

```bash
.venv/bin/python -m pytest -q
cd frontend
npm run typecheck
npm run build
npm run lint
npm test
# With API and frontend running:
PLAYWRIGHT_BROWSERS_PATH=../cache/playwright npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=../cache/playwright npm run e2e
```

Actual managed-worker and MPS tests required normal host access: the execution
sandbox blocked OpenMP shared memory and hid Metal. This is an environment
restriction, not a CPU fallback. Browser tests launch Chromium with background
render throttling disabled on this Mac. Generated shadcn source produces three
nonfatal fast-refresh export warnings; application lint has no errors. Registry
source attribution and MIT notice are in frontend/THIRD_PARTY.md.

For a fresh checkout without existing accepted outputs, create demo checkpoints
explicitly before opening the UI:

```bash
pdeflow smoke --config configs/experiments/cpu_2d.json --output outputs/demo-2d
pdeflow smoke --config configs/experiments/cpu_3d.json --output outputs/demo-3d
pdeflow smoke --config configs/experiments/cpu_1d.json --output outputs/demo-1d
export PDEFLOW_MODEL_ROOTS="outputs/demo-1d/training:outputs/demo-2d/training:outputs/demo-3d/training"
python scripts/dev.py
```

The Python/API tests generate their own temporary tiny checkpoints and do not
require accepted data, weights or output directories from this development
session. Browser tests require the three real registered dimension profiles
above (or the recorded continuation profiles), and exercise the actual API.

The current real browser suite additionally needs both Darcy CPU smoke exports
registered (multispecialist.md), and uses the existing repository browser cache:
`PLAYWRIGHT_BROWSERS_PATH=../cache/playwright npm run e2e` from frontend.
All current screenshots write under outputs/multispecialist-20261021; historical
accepted screenshots are preserved. Darcy results select physical a/u and true
3D slice normal/index; static views have no fictitious physical-time controls.
Routing reports expose the deterministic rationale, compatible IDs and
exclusion reasons, and invalid current mathematics cannot launch a stale draft.

## Online/chemistry continuation

Heat, Darcy and chemistry now execute. For current launch roots and offline/
online/hybrid command export see online-training.md; chemistry.md describes
species/time/3D controls and parameter/kinetics restrictions. Field error tables
are reference-only; positivity/invariant diagnostics remain available without
truth. Workers run conventional Darcy inversion from declared observations
and known forcing, with baseline evaluation updated only after attachment.
Latest browser evidence contains ten real workflows. Screenshots now write
only outputs/online-chemistry-20261021, preserving earlier task evidence.

## Process feedback and final grid

Feedback controller and budgets exposes the server's inference.controller
configuration. It is enabled with unconfigured acceptance thresholds by default;
this yields explicit unresolved/budget-exhausted completion, not verification
success. The advanced JSON accepts per-component bounds, check interval,
patience, model/step/guidance/attempt/grid/population/wall limits and optional
heat refinement/fine_model_id. Registered server content IDs select fine models;
API validation rejects incompatible grids/checkpoints before queuing. A missing
fine export produces an unavailable action if refinement is requested.

Jobs persist latest controller reports in process-live.json and job.json. Final
metrics.process retains decisions/history/budgets/stop reason, and verification
retains component units/status/counts. ProcessPanel shows these on results and
history reopen. The API result's request.problem is the resolved final physical
problem (fine if refined); original_request retains the submitted coarse request for reproducing the
process. The final-state view aligns its checkpoint ID, observations, symbolic
domain and seed with returned fields; it is not a new editable draft or a
replacement for the original process input.
The result NPZ also stores final_model_state for sampler inspection. A history
reopen displays results; it does not silently replace the current editable
mathematical draft. To submit a new request, create/review the draft explicitly.

Refinement keeps observation rows in identical physical coordinates; the saved
fine observations contain remapped indices. Coarse evaluation references are
not fine-grid truth; attachment is rejected if the final layout differs. API
cancellation terminates the bounded worker and reports STOP_CANCELLED, retaining
any saved live trace but never pretending final verification ran after a kill.
FK resampling now executes through the separately specified particle method;
FK plus refinement remains unsupported. See fk-verification.md and the historical
process-verification.md.

## FK weighted populations

For a compatible current request/checkpoint, select FK or Terminal importance.
Particles are candidate complete solutions, not physical molecules. Configure
particle count (up to256 within memory/budget guards), fixed Euler–Maruyama/
vanishing-noise parameters, Gaussian-sum or explicitly tempered-mean observation
energy, PDE weight/scales, systematic ESS/interval, chunk size and optional final
resampling. Unsupported paths, missing FK config, invalid ESS, retries and
FK-plus-refinement reject independently on the API. Current equation/layout
compatibility determines availability; a stale preset/model label cannot enable it.

The result shows real ESS over generation position, before/after values, maximum
weight, root ancestors, energy components, model/particle/energy counts, weights
and actual controller actions. The weighted mean/spread and slice payloads use
normalized terminal weights. Terminal systematic resampling is labelled; high
ESS after resampling does not imply ancestry diversity. Cancellation/budgets
before1 are incomplete probability paths. Terminal completion is separate from
physical criteria met. Reopening or attaching/removing evaluation truth does
not change inference fields/weights/ancestry/actions. See fk-verification.md for
exact launch, shortest workflow and saved1D/2D/true3D browser evidence.

## Everyday workspace (October 25 redesign)

Start `pdeflow serve` in the activated Python environment and `npm run dev`
inside `frontend/`. Open http://127.0.0.1:5173. Use the same managed workspace
for training and serving. The API runs at http://127.0.0.1:8000.

Workspace has three freely accessible stages: **Define problem**, **Add
information**, **Reconstruct & inspect**. Runs and Models & Training are top
navigation destinations. The persistent seven-part request summary links directly
to the relevant editor; preset/default values require review. The mathematical
draft is stored locally under a versioned key, including observations and
settings. Invalid advanced JSON is also retained and blocks execution. Use Reset
draft or Undo last problem edit deliberately; export JSON for durable transfer
or drafts exceeding browser storage.

For the shortest real demo:

1. Create a compatible model with
   `pdeflow train --preset quickstart-heat-2d --device cpu` (20 small updates).
2. Open Guided example, Load supported heat example, then Dismiss example.
   Inspect MathLive's editable equation and Symbols in this problem. Check the
   automatic model summary. Change model / review layout exposes alternatives;
   adopting a layout always opens a change review.
3. Add information → Generate synthetic observations. Inspect actual positions
   and values on the preview and table. Selecting a row changes the field,
   physical time and 3D slice to that observation. Clicking a plotted marker
   selects its table row. Synthetic truth stays separate from inference.
4. Configure reconstruction; choose sample count, method and device. Numerical
   sampling settings, guidance, FK and controller controls are disclosures.
   Validate complete request, then Run reconstruction.
5. Inspect reconstruction, empirical population spread and saved inputs.
   Reference/error appear only after explicitly attaching evaluation truth.
   A finished execution does not certify physical criteria or accuracy.
6. Runs reopens immutable outputs; edited draft inputs remain separate with
   **Inputs changed since this run**. Compare 2–4 finished runs only when their
   actual problem/measurement identities match.

Heat's physical time is separate from generation coordinate and optimizer update.
Darcy is static: preview **f** as the known forcing and reconstruct **a/u** as
permeability/pressure, without physical-time controls. Chemistry's a/b/c are
species concentrations, with independently selectable field and saved time.
3D previews use actual orthogonal slices and normal coordinates. Only observations
on the selected saved plane are shown; this is not a full-volume projection.

Hatched input regions mean unknown. No interpolation or hidden truth fills them.
Known arrays are shown at their supplied coordinates. Measurement noise is the
independent Gaussian standard deviation, not a physical parameter or guidance
weight. Empirical spread is not calibrated uncertainty. Weighted methods use
weighted means/spreads and weighted observation predictions.

Models & Training reads actual managed history, validation checkpoints and saved
checkpoint updates. Linked external exports without managed history have no
invented curve. Named preset budgets come from package metadata; opening this
page never starts training. Current pilots remain scientifically unevaluated.

The preview/chart API is bounded; full numerical ensembles remain NPZ downloads.
Large input grids may need smaller explicit previews. Current geometry is
periodic Cartesian; unsupported boundary choices remain editable drafts and are
labelled execution unavailable. See [design and tradeoffs](workbench-design.md)
and [verification](workbench-verification.md).

## Complete reference experiments

The Reference experiment card extends the existing Problem/Add information views.
Its complete inputs and truth-visibility toggle are separate from the saved inverse
request. Explicit Solve reference → Sample measurements → Reconstruct from
measurements → Compare actions reuse the canonical editor, routing and actual
numerical jobs. Manual observations and the previous explicit synthetic shortcut
remain available. No hidden full fields enter the worker. Coordinate policy,
per-field/per-time metrics, original physical/FK/controller diagnostics and a full
experiment export are available in Results. Models & Training now includes durable
sequential-suite summaries and links to real child history. See
[reference-experiments.md](reference-experiments.md) and
[training-suites.md](training-suites.md) for exact workflows/commands.

## Linked mathematical interaction

The current central experience is documented in flow-workspace-guide.md and
flow-workspace-design.md: linked heatmap/surface/native point, actual local terms,
orthogonal3D slices, explicit physical/generation/spatial modes and real bounded
sampler playback with FK ancestry. Synthetic reference context is evaluated
automatically after successful sampling and retained on reopening; truth stays
outside the worker. Clear workspace confirms and invalidates late draft callbacks
while preserving saved artifacts. Selection and schema failures are actionable,
with technical details expandable. Current screenshots/evidence:
flow-workspace-verification.md.

## Default application changed (27 October 2026)

The default is now the [2D heat playground](heat-playground.md). This document
continues to describe the broader **Legacy** workbench, reachable from the header.
Existing models, outputs and lower-level workflows remain preserved.
