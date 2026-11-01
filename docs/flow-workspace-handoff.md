# Model selection, flow matching, capacity and linked workspace handoff

Use the repository root, existing Python3.12 venv and supported Node22.23.3/.nvmrc
on PATH. New evidence is outputs/flow-workspace-20261026/. Previous accepted
experiments/checkpoints were preserved.

## Start locally

```sh
source .venv/bin/activate
pdeflow serve
```

In a second terminal, from the repository root:

```sh
cd frontend
npm run dev -- --port 5173 --strictPort
```

Open http://127.0.0.1:5173; API http://127.0.0.1:8000. Combined alternative:
`python scripts/dev.py` with the venv active and Node/npm on PATH. To reproduce the
new task's exact saved jobs/screenshots, serve with
`--workspace outputs/flow-workspace-20261026/workspace`. This workspace links the
existing compatible8² heat/Darcy/chemistry and16-node1D exports; no new long training
is needed. On a fresh checkout:

```sh
pdeflow train --preset quickstart-heat-2d --device cpu --steps 2
```

That creates an8² heat checkpoint, not a high-resolution model. Automatic selection
is explicit in the API and resolves server-side; empty/stale/missing/wrong-layout
selection is rejected clearly. See model-selection-regression.md for the captured
original model_id="" request, cause, fix and browser evidence.

## Use the actual visual workflow

Insert heat → review the domain/coefficient → Load visible reference defaults →
Preview/Solve reference → Sample measurements → Reconstruct from measurements →
Validate/Run. The already-associated reference/error now appears automatically
without a manual reattachment step. Reopen/refresh retains it. Spatial inspection
links field/surface/point and actual local equation terms; Physical evolution uses
saved τ; Generative evolution plays captured model/current/endpoint states at s.
Static Darcy has no physical-time animation. FK ancestry changes/gaps are explicit.
Individual samples/spread and saved-input numerical views remain available. Clear
workspace confirms and leaves saved files intact; running jobs offer keep/cancel/
return choices. Refresh/late responses cannot restore the cleared draft.

Current checkpoint fields are not scientifically accurate merely because these
workflows execute. The actual interface/screenshots: flow-workspace-guide.md and
outputs/flow-workspace-20261026/browser/.

## Medium local pilot and matching reference experiment

These are real user-launchable pilots; the full3000-update jobs were not run here:

```sh
pdeflow train --preset heat-2d-medium --device mps --workspace outputs/my-medium
pdeflow resume latest --workspace outputs/my-medium
pdeflow models --workspace outputs/my-medium
pdeflow experiment --problem configs/reference/heat-64.json --model auto --workspace outputs/my-medium --output outputs/my-medium-experiment --device mps
pdeflow compare --result outputs/my-medium-experiment/reconstruction --reference outputs/my-medium-experiment/reference --output outputs/my-medium-comparison.json
```

The medium preset is native64²,T4,width16/3 levels/1 block,condition32,132,436
parameters,batch1. Matching explicit larger local profile:

```sh
pdeflow train --preset heat-2d-large --device mps --workspace outputs/my-large
pdeflow experiment --problem configs/reference/heat-128.json --model auto --workspace outputs/my-large --output outputs/my-large-experiment --device mps
```

It is native128²,T4,width24/4 levels/2 blocks,1,257,436 parameters. Its bounded
MPS forward/backward/guidance/Heun probe passed. Scientific quality is unestablished.
For a short execution check add `--steps 2 --stop-after 1`, then resume to the
recorded total2. Do not edit Python source between interruption and resume:
source/config compatibility is strict. Old inference exports remain loadable.

## CUDA/Colab-oriented native1024² path

The actual1024² MPS operation probe passed; CUDA/actual Colab remain untested.
The full profile is explicitly user launched:

```sh
pdeflow train --preset heat-2d-1024-cuda --device cuda --workspace /content/pdeflow-runs
pdeflow resume latest --workspace /content/pdeflow-runs
pdeflow experiment --problem configs/reference/heat-1024.json --model auto --workspace /content/pdeflow-runs --output /content/pdeflow-native1024-experiment --device cuda
```

This is heat1024²,T3=[0,.05,.1],width8/4 levels/1 block,73,939 parameters,
batch1,10000 configured updates. It does not fit arbitrary existing8²/32²
checkpoints. The existing notebook installs the same package; use configurable
persistent storage (e.g. replace /content/pdeflow-runs with your persisted directory),
then call these commands or resume and export. The disposable runtime is not a
persistence guarantee. Copy the resulting training/inference.pt to the Mac and
`pdeflow models --import /path/to/inference.pt --workspace outputs/my-imports`;
serve that same workspace. CPU remapping is supported; cross-device bitwise
identity is not promised. No CUDA/Colab execution is claimed.

A bounded real operation probe, not a long training run:

```sh
pdeflow capacity-probe --config configs/resolution/heat-1024.json --device mps --output outputs/my-native1024-probe
```

It executes a native reference, one real CFM update, endpoint guidance and production
Heun source-to-candidate sampling with genuine bounded snapshots. Native returned
shape is[1,1,1,3,1024,1024]; no output upscaling or tiling. See capacity-resolution.md
for per-component support, measured memory/time and limits. A1024² forward/update
is not a validated trained1024² inverse capability. No1024³ target is implied.

## Low-level reconstruction and evaluation

After the medium run and reference experiment above, use the exact exported
checkpoint path printed by `pdeflow models --workspace outputs/my-medium`:

```sh
CHECKPOINT="/path/printed/by/models/training/inference.pt"
pdeflow reconstruct --config configs/experiments/heat-2d-medium.json --device mps --checkpoint "$CHECKPOINT" --problem outputs/my-medium-experiment/observations/problem.json --observations outputs/my-medium-experiment/observations/observations.json --mode observation_pde --output outputs/my-medium-cli
pdeflow evaluate --result outputs/my-medium-cli/observation_pde --reference outputs/my-medium-experiment/reference/reference.npz
```

Use a fresh output directory for each reconstruction. `--device cpu` is supported
with the same export. Evaluation truth is opened only after sampling; it is not
needed by the reconstruction command. The exact current export/observations were
also exercised through these low-level commands on CPU; cli-reconstruct.log and
cli-evaluate.log record this separate delivery check.

## Suites and verification

```sh
pdeflow train-all --plan
pdeflow train-all --suite capacity-local --device mps --plan
pdeflow train-all --suite capacity-local --device mps
pdeflow resume-suite latest
pdeflow suite-status latest
```

The subsequent catalogue continuation changes the default to full/all (11 targets).
Use --suite local-pilot for the original six. capacity-local still explicitly
contains medium64² then large128². See training-catalogue.md for current coverage.
Wrong-suite budget overrides are rejected, not silently ignored. No full suite
pilot was launched for this task.

```sh
python -m pytest -q
cd frontend
npm test
npm run build
npm run lint
npx playwright test e2e/flow-workspace.spec.ts
```

The real-registry browser suite needs the compatible exports noted above; the
separate empty-registry test uses `PDEFLOW_EMPTY_REGISTRY=1` against an intentionally
empty API workspace. Component mocks do not count as numerical jobs.

## What the measurements mean

Training and inference use the actual independent Gaussian linear CFM path,
production loss and Euler/Heun steppers; the complete audit and mathematical tests
are in flow-matching-audit.md. A learned Gaussian toy reports mean/std errors rather
than claiming exact moments. A matched20-update comparison used the same physical
and calibration IDs: small CNN CFM1.51396 versus medium U-Net1.53007; validation
unobserved relL2(obs+PDE)1.47368 versus1.55722. Fourier inversion was0.52520.
Larger was not better. Neither validation nor test results silently choose a model,
particle or training budget. Earlier test-budget non-monotonicity remains unchanged.

Remaining limits: periodic Cartesian specialists, fixed declared resolution/time
layouts, finite receptive field and uncalibrated empirical spread; no general
symbolic solver, arbitrary boundaries or validated cross-resolution generalization.
High-resolution FK populations, long high-resolution training quality, CUDA and
actual Colab are unverified. Next scientific slice: repeated-seed validation-designed
capacity/training experiments with native-resolution baseline/reference convergence,
then untouched held-out testing; no test-selected budget claims.
