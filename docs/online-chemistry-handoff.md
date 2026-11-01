# Online data, Darcy inverse and chemistry handoff — 22 October 2026

The shared trainer executes legacy offline datasets and common physical offline,
fresh online and bounded hybrid replay providers. Heat (1D–3D), Darcy (2D/3D)
and A+B reversible C (2D/3D) retain separate physics/encoders/transforms behind
one trainer and sampler. Online removes the need to save a full training set;
clean numerical reference generation and fixed held-out evaluation remain necessary.

All source/config/command paths below are relative to the repository root. Use
Python3.12 and the user's Node>=22.12 on PATH (tested Homebrew22.23.3). The package
and frontend dependencies are installed in this workspace. Fresh installations:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-tested.txt -e '.[test,workbench]'
npm --prefix frontend ci
```

## Start the workbench with existing checkpoints

First terminal:

```bash
cd /Users/nicolasackermann/Documents/ChatGPT/pdeflow
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training:outputs/workbench-20261021/cpu-1d-regression/training:outputs/multispecialist-20261021/cpu-2d/training:outputs/multispecialist-20261021/cpu-3d/training:outputs/online-chemistry-20261021/chemistry-2d/training:outputs/online-chemistry-20261021/chemistry-3d/training"
export PDEFLOW_JOB_ROOT="outputs/my-current-workbench/jobs"
export PDEFLOW_DATA_ROOT="outputs/my-current-workbench/data"
.venv/bin/python -m pdeflow.workbench.api
```

Second terminal:

```bash
cd /Users/nicolasackermann/Documents/ChatGPT/pdeflow/frontend
node --version
npm run dev -- --strictPort
```

Open http://127.0.0.1:5173; API is http://127.0.0.1:8000. Alternatively use
`python scripts/dev.py` with the same environment. The direct API/Vite startup
above was used successfully for the final actual browser run. Stop both with
Ctrl-C. Verification servers have already shut down.

Shortest chemistry workflow: insert A+B reversible C; leave2D8² defaults and
known positive parameters; confirm checkpoint:yes and selected specialist:
chemistry in the routing report. Continue to observations → Generate synthetic
observations → Configure reconstruction → Validate complete request → Run
reconstruction. Results initially have no evaluation errors. Select a/b/c and
physical time; attach synthetic reference only to evaluate afterward. Select
spatial dimension3 for the real8³ model and orthogonal slice controls. Heat and
Darcy presets similarly route from current parsed equations/full request.
Darcy has a/u static slices, not fictitious physical time. Editing kinetics or
selecting a wrong-family checkpoint blocks execution independently at the API.
Opening/selecting training mode exports config/commands; it does not train.

Archives omit all weights/data. To obtain new compatible models from an
extracted source ZIP, explicitly run the real bounded smoke, then register:

```bash
.venv/bin/python -m pdeflow.cli smoke --config configs/experiments/online_chemistry_2d.json --output outputs/my-chemistry-demo
export PDEFLOW_MODEL_ROOTS="outputs/my-chemistry-demo/training"
```

Use online_heat_2d.json or online_darcy_2d.json for the other families; use
*_3d.json for true3D. Each needs a fresh root. New provider smoke observations
are in evaluation/test/0/ rather than the legacy data/ directory.

## Prepare fixed held-outs; train on MacBook MPS; resume

Choose one of these profiles and a distinct output root. Every2D profile is
3000 updates, every3D profile1500; these larger runs were not executed here.

| Family | 2D MPS profile/grid | True3D MPS profile/grid |
|---|---|---|
| Heat | online_mps_heat_2d.json,32² | online_mps_heat_3d.json,16³ |
| Darcy | online_mps_darcy_2d.json,32² | online_mps_darcy_3d.json,16³ |
| Chemistry | online_mps_chemistry_2d.json,16² | online_mps_chemistry_3d.json,8³ |

Copy exactly one assignment pair (default below chemistry; substitutes shown):

```bash
PDEFLOW_CONFIG=configs/experiments/online_mps_chemistry_2d.json
PDEFLOW_RUN=outputs/my-online-chemistry-mps
# Heat: PDEFLOW_CONFIG=configs/experiments/online_mps_heat_2d.json
#       PDEFLOW_RUN=outputs/my-online-heat-mps
# Darcy: PDEFLOW_CONFIG=configs/experiments/online_mps_darcy_2d.json
#        PDEFLOW_RUN=outputs/my-online-darcy-mps
.venv/bin/python -m pdeflow.cli prepare-evaluation --config "$PDEFLOW_CONFIG" --output "$PDEFLOW_RUN/evaluation"
.venv/bin/python -m pdeflow.cli train --config "$PDEFLOW_CONFIG" --device mps --evaluation "$PDEFLOW_RUN/evaluation" --output "$PDEFLOW_RUN/training" --stop-after 100
.venv/bin/python -m pdeflow.cli train --config "$PDEFLOW_CONFIG" --device mps --evaluation "$PDEFLOW_RUN/evaluation" --output "$PDEFLOW_RUN/training" --resume "$PDEFLOW_RUN/training/checkpoint.pt"
```

The first training command stops at100; the resume continues to the configured
budget. Keep source/config/manifest unchanged. Calibration is frozen and not
regenerated on resume; hybrid regenerates its bounded replay from saved specs,
charging the additional solves. CPU deterministic online/hybrid IDs and weights
match uninterrupted runs; no cross-device bitwise guarantee is made.

Optional preflight (fresh directory):

```bash
.venv/bin/python -m pdeflow.cli preflight --config "$PDEFLOW_CONFIG" --device mps --output outputs/my-new-preflight
```

No full training dataset generation is required. Hybrid example with fixed8²
held-outs, synchronous capacity-bounded replay and the actual CPU config:

```bash
.venv/bin/python -m pdeflow.cli prepare-evaluation --config configs/experiments/hybrid_darcy_2d.json --output outputs/my-hybrid/evaluation
.venv/bin/python -m pdeflow.cli train --config configs/experiments/hybrid_darcy_2d.json --evaluation outputs/my-hybrid/evaluation --output outputs/my-hybrid/training
```

Replace darcy by heat or chemistry for their hybrid profiles. Offline common
provider commands, matched-mode cost probe and explicit data contract are in
online-training.md. Legacy NPZ/sharded commands remain supported.

## Register, reconstruct and evaluate

```bash
.venv/bin/python -m pdeflow.cli inspect-checkpoint --checkpoint "$PDEFLOW_RUN/training/inference.pt"
export PDEFLOW_MODEL_ROOTS="$PDEFLOW_RUN/training"
.venv/bin/python -m pdeflow.workbench.api
```

Restart the API with all desired roots, colon-separated. Discovery uses exported
content identity; no separate mutable registry file/registration service exists.
Use another terminal for CLI reconstruction, or stop the API first:

```bash
.venv/bin/python -m pdeflow.cli reconstruct --config "$PDEFLOW_CONFIG" --device cpu \
  --checkpoint "$PDEFLOW_RUN/training/inference.pt" \
  --problem "$PDEFLOW_RUN/evaluation/test/0/problem.json" \
  --observations "$PDEFLOW_RUN/evaluation/test/0/observations.json" \
  --reference "$PDEFLOW_RUN/evaluation/test/0/reference.npz" \
  --output "$PDEFLOW_RUN/reconstruction"
.venv/bin/python -m pdeflow.cli evaluate --result "$PDEFLOW_RUN/reconstruction/observation_pde"
```

Reference is optional and is opened only after generative sampling and the
Darcy conventional inverse. No hidden field is an inference input. CLI outputs
unguided, observation, observation_pde and (Darcy) darcy_inverse folders.

Existing chemistry smoke can be reconstructed immediately using config
configs/experiments/online_chemistry_2d.json and run root
outputs/online-chemistry-20261021/chemistry-2d; choose a fresh reconstruction
output path. Its saved reconstruction/ already contains all three modes.

## Colab and transfer back

Open notebooks/colab.ipynb. Upload pdeflow-online-chemistry-source.zip and set
SOURCE_ARCHIVE to its actual uploaded path; leave repository URL empty, since
this checkout has no commit/remote. Choose persistent OUTPUT_ROOT explicitly.
Set PROFILE to online_colab_heat_2d, online_colab_darcy_2d or
online_colab_chemistry_2d (or *_3d), DEVICE="cuda", and a fresh RUN_NAME.
Run installation/configuration/training cells. Online/hybrid skip full training
arrays; fixed evaluation is persisted/restored across disposable runtime loss.
Rerun the train cell with unchanged source/config to resume checkpoint.pt.
Actual CUDA/Colab execution is untested; local CPU notebook cells, cache deletion
and persistent evaluation restoration were verified.

Transfer portable-export/ (inference.pt,config.json,test_problem.json,
test_observations.json) back to the Mac. No training arrays are needed:

```bash
PDEFLOW_EXPORT=outputs/from-colab/portable-export
.venv/bin/python -m pdeflow.cli reconstruct --config "$PDEFLOW_EXPORT/config.json" --device mps \
  --checkpoint "$PDEFLOW_EXPORT/inference.pt" --problem "$PDEFLOW_EXPORT/test_problem.json" \
  --observations "$PDEFLOW_EXPORT/test_observations.json" --output outputs/from-colab/reconstruction
```

Use --device cpu for CPU reload. CUDA-to-MPS transfer is an intended portable
path, not a tested cross-device experiment. Full training resume also needs
checkpoint.pt, same source/config and fixed evaluation; copy journals to retain
complete historical costs. Do not change device/config midway and claim exact
training continuation.

## Evidence and scientific limits

114 Python tests passed;16 frontend tests, TypeScript/build/lint passed (3
existing lint warnings and large bundle warning);10 actual browser workflows
passed. CPU chemistry performed8/4 updates at8²/8³. Actual MPS online heat/Darcy
8² and chemistry16²/8³ each performed2 updates/resume/guidance/CPU reload.
Earlier heat/Darcy accepted models remain smoke-trained. Darcy has a bounded
one-path empirical pilot; no model here is scientifically validated.

Matched four-update CPU warm totals (offline/online/hybrid seconds):
heat .2549/.0681/.0732; Darcy .1153/.0706/.0597; chemistry .4664/.4249/.2777.
Offline solved8/consumed6 unique train problems with2 reuses; online8/8 with0;
hybrid4/4 with4. Every mode used2 calibration references. Common initialization
and fixed-evaluation costs are separate; these tiny single trials do not rank
large-scale throughput. See timing-summary.json and online-training.md.

Darcy study budgets0/20/80 consumed0/40/160 unique training examples on one path.
Conventional inverse beat every generative method: mean unobserved relative L2
(a,log(a),u) .16895/.76222/.09630 vs80-update observation+PDE
.27780/1.25194/.50603. Permeability quality was non-monotonic. Validation alone
selected weights/ridge; no test-selected budget. Six of8 inverse optimizations
hit their24-update cap, despite all299 forward/299 adjoint test calls meeting
CG tolerances. Two independent test problems are too few for broad conclusions.
The older heat Fourier inverse also remains better; its study was preserved.

Chemistry uses periodic Strang/exact reaction/positive diffusion, no concentration
clipping. Saved reference max a+c/b+c drift was1.35e-14/2.03e-14 (2D/3D), while
learned reconstruction drift and errors remain material. Square decoding ensures
nonnegativity, not invariants or calibrated posteriors. See chemistry.md for
scaling, reaction limits, fixed step error control and known-rate restrictions.

Workers=0/single producer/solver_threads=1 only; hybrid is replay rather than
parallel prefetch. Chemical boundaries are periodic only, rates known, species
names fixed; generic equation parsing is not generic neural physics. Grids and
physical-time layouts must match checkpoints. No full-size training, CUDA or
actual Colab run was performed. Next scientific slice: repeated seeds and more
independent held-out problems, stronger documented inverse stopping and finer
physical evaluation, then invariant-aware chemistry comparisons at matched cost.

## Deliverables and Git

New artifacts: outputs/online-chemistry-20261021/; raw tests/browser reports,
reference journals, MPS preflights, chemistry smoke results, timing-summary.json,
darcy-study/, verification-summary.json and delivery-checks.json. Source archive:
pdeflow-online-chemistry-source.zip under that root; it excludes accepted outputs,
weights/data, environments/caches, personal files and secrets. SOURCE_PACKAGE.md
explains generating or transferring omitted artifacts. Every archive manifest
hash and required file was checked. Existing accepted outputs were preserved.

Git is main with no HEAD/commit/remote. Ninety staged additions remain plus
unstaged/untracked work; no Git mutation was attempted. The current index byte
hash differs from the earlier recovery report, so that historical bytewise
preservation claim is not repeated. Actual entries/hash are in delivery-checks.json.
No action currently remains blocked by automatic approval review.
