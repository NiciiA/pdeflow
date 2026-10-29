# PDEFlow

Train specialist flow models and reconstruct PDE fields from sparse measurements.
Heat (1D–3D), Darcy and A+B reversible C chemistry (2D/3D) are implemented.
Models, guidance and FK remain research tools: a finished run does not establish
scientific accuracy or calibrated uncertainty.

## Play with heat

The default app is now a focused **2D heat playground**. Choose a starting field
or paint it, press **Play**, take N measurements, then **Run model**. Simulation
needs no checkpoint. Reference/model/error views share physical time and a common
heat scale. Existing smoke checkpoints require explicit experimental use and have
poor accuracy; more output pixels do not establish more learned detail.

**Heat along y** (left) and **Heat along x** (below) integrate the current field
across the other direction. Profiles default on; choose **Magnitude** for |u|
integrals. Signed values may cancel. Reference/model profiles share scales.
See [profile conventions and checks](docs/spatial-profiles.md).

**Train model** opens a separate view, defaulting to heat-2D,
resolution-flexible, Development run and Auto device. The existing orchestration
manages both stages and validation. **Storage** offers reviewed permanent removal
of selected/all recognized inference, resume and stage weights, with optional
associated generated data/results. No existing artifacts were deleted to build
this feature. Other PDEs, dimensions and fixed-grid tools remain behind **Legacy**.

See the [playground guide](docs/heat-playground.md) for play/sample/run, safe
expressions, measurement import, deletion scopes and numerical limits, and
[actual verification/screenshots](docs/heat-playground-verification.md).
[Training](docs/training-workflow.md), [coordinate-model CLI/Colab/comparison](docs/coordinate-specialist.md)
and [legacy workbench](docs/workbench.md) documentation remain available.

## Continuous physical time and sensor placement

New heat2D training uses continuously sampled physical times over **[0,1]**,
with unchanged D. Animation frames control display/export only. Scrub to0.237
and choose **Random**, **Evenly spaced**, **Heat-focused** or **Click points**.
Heat focus balances retained hot/cold regions with background coverage and uses
reference information. A generated reconstruction keeps one latent across time.
Old snapshot checkpoints retain their actual temporal capability; they are not
automatically continuous models. See [the workflow/migration guide](docs/continuous-heat.md)
and [actual verification/screenshots](docs/continuous-heat-verification.md).

## Field reconstruction loss and error metrics

New heat2D representation runs select experimental **Region-balanced MSE**, α=0.5,
with actual encoder/decoder gradient updates. **Train model → Field reconstruction
loss** also offers ordinary MSE, contrast-weighted MSE and MAE. Existing runs keep
their recorded loss; the latent flow velocity objective stays unchanged. The updated
Colab notebook uses the same new-run default and shared trainer. Large budgets are
preserved. Region balancing can trade whole-field/background accuracy for distinct
feature preservation; no accuracy improvement or optimal α is established.

The comparison's **Error metric** selects a scalar; **Error map** independently
selects a labelled pointwise discrepancy. **Show detected regions** adds boundaries.
These display choices reuse predictions and never retrain or rerun inference.
See [mathematics/configuration/controlled comparison](docs/field-reconstruction-losses.md)
and [actual tests/screenshots/limitations](docs/field-loss-verification.md).

## Substantial Colab training

Upload [notebooks/colab.ipynb](notebooks/colab.ipynb) to Google Colab and select a
GPU runtime. Follow its six cells: source ZIP or exact checkout, persistent
storage, settings, actual data preview/profile, **Run training**, and **Download
model for PDEFlow**. The **Large** preset preserves the requested 65,536 training
solutions, width512 / tokens128 / layers8 / heads8 and 100,000 updates per stage.
**Extra large** expands capacity and budget; neither is promised to fit every GPU.

Set `RUN_TRAINING=True` only after inspecting preflight. Resume with the same
source/storage/run name and `RESUME_MODE="resume"`; stage dependencies, gate,
checkpoint integrity and export are managed automatically. Import the downloaded
`coordinate.pt` through the local playground's **Import model** button.
See [the operating guide](docs/colab-training.md) and
[actual verification](docs/colab-verification.md). Only bounded CPU training,
sequential notebook cell execution and real browser import were run here;
CUDA, actual Colab and mounted Drive execution remain unverified.

Create a source ZIP without datasets, weights or private files:

```bash
python scripts/package_source.py --output outputs/pdeflow-colab-source.zip
```

The original multi-specialist notebook remains at
[notebooks/colab-legacy.ipynb](notebooks/colab-legacy.ipynb).

## Start on a MacBook

Use Python **3.12** and Node **22.23.3** (`.nvmrc`; `nvm install` if you use nvm).
Supported frontend tools require Node ^22.13, ^24 or >=26. No global software is
installed automatically. If the existing `.venv` works, keep it and activate it.
For first-time setup, from this source directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[workbench]"
pdeflow setup
pdeflow doctor
```

Start the backend and frontend in separate terminals:

```bash
# Terminal 1, activated Python environment:
pdeflow serve
```

```bash
# Terminal 2, repository root:
cd frontend
npm run dev -- --port 5173 --strictPort
```

Open **http://127.0.0.1:5173**. The API is **http://127.0.0.1:8000**. Trained local
exports appear automatically. Use the header **Train model**; after
validation **Use this model** selects the exact artifact for reconstruction.
Neither startup command trains a model.

Inspect or resume the most recent integrated training run:

```bash
pdeflow training list
pdeflow training status latest
pdeflow training resume latest
pdeflow training use latest
```

“Latest” is the most recently created integrated run, never the best accuracy.
The default persistent workspace is **outputs/local/**. Historical project-output
workspaces are discovered consistently. External workspaces remain explicit.
The original commands below remain available as advanced tools; they are not
needed to manage the normal two-stage model workflow.

| Need | Command |
|---|---|
| List training choices | `pdeflow presets` |
| Check without training | `pdeflow train --preset heat-2d --device mps --dry-run` |
| Darcy / chemistry | `pdeflow train --preset darcy-2d --device mps` / `--preset chemistry-2d` |
| Small true 3D | `pdeflow train --preset chemistry-3d --device mps` |
| Extend recorded budget | `pdeflow resume latest --additional-steps 1000` |
| Headless setup | `pdeflow setup --headless` |
| Link an existing export | `pdeflow models --import PATH` |
| Diagnose environment | `pdeflow doctor --json` |

See [everyday commands, presets and troubleshooting](docs/everyday-usage.md),
[actual user-journey verification](docs/everyday-verification.md),
[advanced configurations and original command examples](docs/advanced-workflows.md),
[Colab notebook](notebooks/colab.ipynb), and [source-only setup](SOURCE_PACKAGE.md).
Scientific details remain in the [capability matrix](docs/capabilities.md),
[mathematics](docs/mathematics.md), [FK derivation](docs/fk-mathematics.md) and
[research specification](docs/project.md). Conventional inverses remain better
in the retained heat/Darcy studies; heat test quality was non-monotonic across
training budgets. No test-based winning budget was selected.

The [workbench guide](docs/workbench.md#everyday-workspace-october-25-redesign)
explains the three-stage workspace, actual input previews, model discovery,
training-history charts and compatible-run comparisons. Start with the bounded
quickstart checkpoint above, then use Guided example → Load supported heat example.
See [UI verification and screenshots](docs/workbench-verification.md) and
[visual API schemas](docs/workbench-api.md).

## Reference experiments and specialist suites (26 October 2026)

The workbench now separates complete reference solving, explicit inverse
disclosure, noisy observation sampling, reconstruction and post-run comparison.
Heat supports an explicit finite-Fourier analytic case; Darcy/chemistry use the
verified numerical solvers. Reference solving needs no neural weights. Hidden
initial states/permeability and clean sampled values stay outside inference.
See [reference-experiments.md](docs/reference-experiments.md) for editable inputs,
exact-grid patterns, coordinate-aware metrics, exports and a full UI walkthrough.

With the venv active, this creates a compatible small checkpoint and a real
8² synthetic experiment (choose fresh output paths for subsequent runs):

```sh
pdeflow train --preset quickstart-heat-2d --device cpu --steps 2
pdeflow experiment --preset heat-2d --model auto --output outputs/my-reference-experiment --device cpu
pdeflow serve
```

In another terminal run `cd frontend` then `npm run dev -- --port 5173 --strictPort`;
open http://127.0.0.1:5173. Use Node from .nvmrc/on your PATH. Insert heat, Load
visible reference defaults, Preview supplied field, Solve reference, Sample
measurements, Reconstruct from measurements, Validate/Run, then Compare.

For the finite versioned training catalogue:

```sh
pdeflow train-all --plan
pdeflow train-all --suite quickstart --device cpu
pdeflow train-all --device mps
pdeflow resume-suite latest
pdeflow suite-status latest
```

Default `train-all` now covers the complete 11-target catalogue, including medium,
large, native 512²/1024² heat and supported 3D targets. `--suite local-pilot` retains
the original six layouts; quickstart uses their explicit grids with two updates.
Long jobs remain user launched; inspect `pdeflow train-all --plan` first. See
[training-suites.md](docs/training-suites.md) for grids, filters, budgets, provenance,
reuse, locks and Ctrl-C. Serve the same --workspace to discover suite exports.
[reference-verification.md](docs/reference-verification.md) records actual tests,
CPU/MPS execution, artifact locations and limits; CUDA/actual Colab are untested.

## Current flow workspace and capacity continuation

The original empty model-ID failure is reproduced and fixed with explicit server
Automatic/registered selection and readable errors. The linked mathematical
workspace now coordinates heatmaps/surfaces, native point diagnostics, reference/
reconstruction/error, physical τ and actual captured generation s. Associated
references appear automatically after inference. Clear workspace resets the
persisted editor with explicit running-job choices while preserving saved artifacts.

Training/inference CFM formulas and executed evidence:
[flow-matching-audit.md](docs/flow-matching-audit.md). Named small/medium/large
capacity and native-resolution policy:
[capacity-resolution.md](docs/capacity-resolution.md). Actual1024² heat MPS
forward/backward/update, physical guidance and two-step Heun operations passed;
this is a one-update execution probe, not validated high-resolution reconstruction.
The matched20-update comparison did not show a larger-model advantage.

```sh
pdeflow train --preset heat-2d-medium --device mps
pdeflow train --preset heat-2d-large --device mps
pdeflow train-all --suite capacity-local --device mps --plan
pdeflow capacity-probe --config configs/resolution/heat-1024.json --device mps --output outputs/my-native1024-probe
```

These pilot commands are explicit user launches; no long training starts from the
UI. Full commands, Colab transfer, screenshots, actual test results and scientific
limits: [flow-workspace-handoff.md](docs/flow-workspace-handoff.md) and
[flow-workspace-verification.md](docs/flow-workspace-verification.md).

## Native 512² heat, complete training coverage and storage

```sh
pdeflow train --preset heat-2d-512 --device mps
pdeflow train-all --plan
pdeflow train-all --missing --plan
pdeflow coverage
pdeflow status latest
pdeflow storage
pdeflow cleanup --preview
```

The new [training catalogue guide](docs/training-catalogue.md) gives exact full/
missing/retrain/resume commands, the matching 512² reference workflow, reviewed
layout adoption, real CLI monitoring and conservative storage/quarantine rules.
The editor includes distinct editable initial-condition and complete analytical
reference fields with verification. No long training or scientific output cleanup
starts when the UI opens. [Findings](docs/catalogue-findings.md),
[lecture mapping](docs/generative-foundations.md) and
[verification/handoff](docs/catalogue-verification.md) distinguish implemented,
configured, partial, completed, evaluated and scientifically established behavior.
