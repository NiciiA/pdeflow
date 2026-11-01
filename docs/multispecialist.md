# Executable heat and Darcy specialists

Historical two-specialist handoff. The subsequent online/chemistry task adds a
third executable specialist and the Darcy inverse baseline. Current workflows
are in online-training.md, chemistry.md and darcy-study.md. Earlier measurements
below are preserved as historical evidence.

Heat and Darcy now execute through the same trainer, CFM formulation, spatial
CNN, Euler/Heun sampler, gradient steering and local job queue. A specialist is
the physical implementation and its assumptions. The residual CNN is a backbone;
independent Gaussian linear CFM is a generative formulation. Each inference.pt
is an individual trained checkpoint, with its own layout, transforms and
conditioning. None of these distinctions imply scientific accuracy.

## Routing and restrictions

The bounded parser produces rank-checked operator trees from the current source.
The registry in src/pdeflow/specialists recognizes only these explicit patterns:

- Heat: u_τ=D lap(u), or u_τ-D lap(u)=0. Supported LaTeX derivative/Laplacian
  notation normalizes to the same trees. Unknown scalar u, known positive scalar
  D; periodic nodal Cartesian trajectories with at least three saved times
  starting at zero; unknown initial information.
- Darcy: -div(a grad(u))=f, or -div(a grad(u))-f=0. Unknown scalar spatial a and
  pressure u, known cell field f; periodic Cartesian 2D/3D cell-centred static
  fields, compatible mean-zero forcing and zero-mean pressure. No initial data.

No preset label, display name, substring or unrestricted symbolic equivalence
selects a specialist. Reordered factors, renamed symbols, extra terms, other
boundaries and other unknown/known roles require separate explicit support.
Syntactically valid unsupported equations remain editable/exportable.

After recognition the complete request validates roles/ranks/units, domain,
boundary, initial information, forcing and observation operator/noise/rows.
The API checks every discovered checkpoint's specialist, exact grid/time layout,
conditioning, formulation and stepper/schedule. Empty observation drafts can
route, but execution requires valid nonempty measurements. No old valid source
can execute when the current mathematical source is invalid or different.

The routing report contains canonical formulation, candidate/selected specialist,
compatible content IDs, exclusions with reasons, unknown ranges, warnings and
selection rationale. Automatic selection takes the lexicographically first
compatible content ID; this is deterministic and is **not an accuracy ranking**.
A manual incompatible override selects no checkpoint and is rejected by the API
before queuing. Select Automatic to route again. An overwritten/deleted export
retires its old ID; queued workers also check identity before sampling.

The trainer and sampler call the small specialist contract for validation,
physical fields/transforms, data, conditions, residuals, diagnostics and baseline
integration. They contain no heat/Darcy equation branches. Measurements are
inference-time steering, separate from known physical conditioning. Checkpoint
load validates capability/field/transform/schema metadata before execution.

## Start with existing checkpoints on this MacBook

From the repository root, use the supported Node installation on PATH
(`node --version`; tested here with Homebrew Node 22.23.3, npm 10.9.9).
Do not insert a private Codex runtime path. Dependencies are already installed
in this workspace; for a new checkout run `cd frontend && npm ci`, then return
to the root and install the Python package as described in README.md.

Terminal 1:

```bash
cd /Users/nicolasackermann/Documents/ChatGPT/pdeflow
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training:outputs/workbench-20261021/cpu-1d-regression/training:outputs/multispecialist-20261021/cpu-2d/training:outputs/multispecialist-20261021/cpu-3d/training"
export PDEFLOW_JOB_ROOT="outputs/my-specialist-workbench/jobs"
export PDEFLOW_DATA_ROOT="outputs/my-specialist-workbench/data"
.venv/bin/python -m pdeflow.workbench.api
```

Terminal 2:

```bash
cd /Users/nicolasackermann/Documents/ChatGPT/pdeflow/frontend
node --version
npm run dev -- --strictPort
```

Open http://127.0.0.1:5173 (API http://127.0.0.1:8000).
Ctrl-C in each terminal stops that service. `scripts/dev.py` remains the combined
launcher, with the same environment variables and Node on PATH. Default roots
include only the historical heat folders; set the roots above for Darcy.

For a source archive without weights, generate explicit demo checkpoints first
with the four existing cpu_2d/cpu_3d/darcy_cpu_2d/darcy_cpu_3d smoke configs,
each in a fresh output folder. Register their training folders in
PDEFLOW_MODEL_ROOTS. Weights and datasets are deliberately omitted from source.

## Shortest workbench workflow

Insert Heat or Darcy in the empty equation editor. The checkpoint selector
stays Automatic; the routing report shows the selected specialist/content ID.
For Darcy, click **Set known mean-zero sinusoidal forcing** explicitly. It only
supplies f, never hidden a/u. The forcing can instead be supplied under
known_fields.f in full request JSON. Use Spatial dimension=3 for the tiny 3D
checkpoint; its grid remains 8³. Other layouts require a compatible export.
Checkpoint layout suggestions require review; they do not silently change the
request or observations. Nonempty preset changes also require review.

Continue to Observations, click Generate synthetic observations, then Configure
reconstruction. Validate the complete request and Run reconstruction. Synthetic
truth is held separately and is evaluation-only. Static rows are
field,x,y[,z],value; transient heat rows also require time. Either physical a or
u, or both, may be sparsely observed; measured zero remains observed. Point
locations must coincide with cells/nodes; no off-grid interpolation is implied.

Results provide a/u field selection for Darcy, ensemble views and orthogonal
true-3D slices. Static Darcy has no physical-time slider; its internal T=1 zero
index is storage only. Attach/remove reference after inference for errors.
Per-field relative L2/MAE, permeability positivity and pressure gauge diagnostics
are in metrics.json; a small ensemble's spread/coverage is not calibrated
uncertainty. The figure reconstruction.png shows a, reconstruction_u.png shows u.

## Larger Darcy generation, MPS training and resume

These are provided commands, **not long runs executed during recovery**. The 2D
profile is 32², 240 problems split 160/40/40, width32/four blocks/batch8, 5000
updates. The 3D profile is true 16³, 120 split 80/20/20, width32/four blocks/batch2,
3000 updates. Both use restartable compressed shards, train-only streaming
statistics and configurable paths. Generation uses CPU NumPy even for MPS/CUDA
profiles. Choose a new root for a new experiment.

```bash
cd /Users/nicolasackermann/Documents/ChatGPT/pdeflow
.venv/bin/python -m pdeflow.cli generate \
  --config configs/experiments/darcy_mps_2d.json --output outputs/darcy-mps-2d/data
.venv/bin/python -m pdeflow.cli train \
  --config configs/experiments/darcy_mps_2d.json --device mps \
  --data outputs/darcy-mps-2d/data --output outputs/darcy-mps-2d/training --stop-after 100
.venv/bin/python -m pdeflow.cli train \
  --config configs/experiments/darcy_mps_2d.json --device mps \
  --data outputs/darcy-mps-2d/data --output outputs/darcy-mps-2d/training \
  --resume outputs/darcy-mps-2d/training/checkpoint.pt
```

For 3D use darcy_mps_3d.json and a new outputs/darcy-mps-3d root throughout.
Resume requires the same source, resolved config/device, dataset, splits and
transforms. Checkpoints retain optimizer and random streams; no scheduler/scaler
is used in the full-precision profiles. To discover the new export, append its
training folder to PDEFLOW_MODEL_ROOTS before starting the API. Discovery scans
inference.pt files, exposes content IDs and does not load training data. The
edited request must match its exact grid; explicitly review a layout suggestion
or edit the grid. Old IDs retire when a running trainer replaces the export.

## Reconstruction and evaluation with existing artifacts

These commands were exercised during recovery and do not retrain. Choose a fresh
reconstruction output path if the command has already completed:

```bash
.venv/bin/python -m pdeflow.cli reconstruct \
  --config configs/experiments/darcy_cpu_2d.json --device cpu \
  --checkpoint outputs/multispecialist-20261021/cpu-2d/training/inference.pt \
  --problem outputs/multispecialist-20261021/cpu-2d/data/test_problem.json \
  --observations outputs/multispecialist-20261021/cpu-2d/data/test_observations.json \
  --reference outputs/multispecialist-20261021/cpu-2d/data/test_reference.npz \
  --output outputs/darcy-cli-demo
.venv/bin/python -m pdeflow.cli evaluate --result outputs/darcy-cli-demo/observation_pde
```

Use darcy_cpu_3d.json and cpu-3d paths for the existing true-3D case.
For your larger MPS run use its config, checkpoint and data paths; --device cpu
remaps inference without training data. Reference is optional evaluation input,
opened only after all three reconstructions. Darcy's conventional inverse
baseline is explicitly **unfinished**; no oracle forward solution is reported
as an inverse reconstruction.

## Colab and transfer

Use notebooks/colab.ipynb with the packaged source ZIP (or a source commit), an
explicit persistent output directory and config
configs/experiments/darcy_colab_2d.json or darcy_colab_3d.json. It calls the same
package generate/train/resume/reconstruct commands; set --device cuda in Colab.
Copy/export training/inference.pt and the matching resolved config back to a
configured Mac folder. Register that folder through PDEFLOW_MODEL_ROOTS.
For standalone CLI inference also copy test_problem.json and
test_observations.json (and optional test_reference.npz for evaluation).
Training data is not needed for inference. Resume in Colab additionally needs
checkpoint.pt, the identical source/config and persistent data shards/manifest.
The notebook validates archive paths/cache identity and installs pinned package
dependencies; it assumes no repository URL, account or Drive mount. CUDA and
actual Colab execution remain untested. Existing CPU/MPS execution does not
verify either of them.

## Scientific status and next slice

The 8² 2D Darcy checkpoint has 20 updates; true 8³ has 12. They executed all
three guidance modes and exported metrics/figures. Intended-layout 32² and 16³
MPS preflights performed two updates each, resume, guidance and CPU reload.
These are smoke-trained checkpoints, not scientifically evaluated models.
Permeability generation is bounded smooth log structure; identifiability,
robust generalization and posterior calibration are not established. Larger
configs are available but their full datasets/training have not run.

The previous heat 0/100/500 study remains unchanged: the conventional Fourier
inverse is better, and test quality was non-monotonic across budgets. No budget
was selected using test performance. Those results do not benchmark Darcy.
Darcy still needs a transparent conventional inverse benchmark and a proper
held-out scientific comparison. The next physical specialist is A+B ↔ C
reaction–diffusion, with nonnegative species, genuine reaction/diffusion
numerics, stoichiometric invariants of a+c and b+c, true 2D/3D tests and the same
routing/observation/checkpoint integration. The sum a+b+c is not conserved.
