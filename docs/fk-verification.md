# FK verification and runnable handoff — 25 October 2026

This continuation implements one specified FK method and verifies its target on
analytic problems before using existing smoke-trained PDE checkpoints. No new
training or training-budget study was run. Earlier accepted outputs and the Git
index are preserved. All executions described below belong to this FK task;
older 0/100/500 heat and 0/20/80 Darcy studies remain historical evidence.

## What executes and what it targets

`sampling.sample(..., mode='fk')` selects `fk.pde.run` and the PDE-independent
`fk.simulate`. Supported dynamics are independent-Gaussian linear CFM with the
derived score-corrected, linearly vanishing-noise SDE; fixed uniform/quadratic
Euler–Maruyama steps; annealed difference potentials; scheduled systematic
resampling triggered by ESS. `importance` uses identical SDE kernels and a single
terminal weight. Existing gradient modes retain their original ODE semantics.

The exact *algorithmic finite-step* target is the numerical SDE terminal law
Q_h,N tilted by exp[-E(decode(X_N))]. The intended learned-flow physical tilt is
p_theta(U|C) exp[-E(U;C)]/Z. Their difference includes numerical error and the fact
that a learned path-specific score need not be the actual learned ODE's density
score. Finite interacting populations add sampling error. This is not an exact
physical posterior. See [the derivation](fk-mathematics.md), including the initial
potential factor, actual terminal-state energy and analytic posterior formulas.

Observation energy is explicitly Gaussian **sum**, or a labelled tempered mean;
PDE penalties use named scaled physical residuals. Measurements are not network
conditioning in these checkpoints. No truth is read until inference finishes.
Darcy decodes positive permeability and zero-mean pressure; chemistry decodes
three nonnegative species with the actual reaction law. No invented boundary,
initial or invariant likelihood is supplied. Structural transforms can affect
the generated law and are part of the checkpoint's base decoder.

Weights, previous potentials and sensitive reductions use CPU float64, while
model/physical energy runs on the selected backend. CPU random streams are
explicit, independently owned by problem/source/noise/resampling; transfers and
their time are recorded. Full-population noise is drawn before chunking. Weighted
means, variances, quantiles, observation/residual metrics and exported slices
respect final weights. Incomplete paths carry an explicit nonterminal label.
Inference snapshots retain RNG/weights/ancestry for inspection; inference resume
is **not implemented**. Training resume is separate.

The outer controller records real RESAMPLE actions, ESS before/after, parent
indices, trigger, coordinate and consumed budget. It cannot modify the fixed
potential, reset weights arbitrarily, stop early as a terminal sample, or combine
gradient proposals with FK. FK retries and FK plus refinement are rejected.
Legacy gradient heat8²→16² refinement was exercised separately on MPS.

## Analytic evidence

Primary files: `outputs/fk-20261025/{gaussian-verified,mixture-verified}/results.json`.
Each uses four seeds11/23/37/53, P∈{512,4096}, N∈{16,128}, all four Cartesian
budgets and base/terminal importance/FK/annealed-no-resampling. All seed records,
ESS trajectories, ancestry, evaluation counts, runtime and population settings
are retained. No-resampling and terminal importance have the same final paths
and normalized weights for identical RNGs; the telescoping test also checks
nonzero initial beta. Zero energy recovers the unsteered SDE.

| Target / method | P | N | Mean Euclidean error | Covariance Frobenius error | Final ESS | Root ancestors |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian FK |512|16|.05852|.04007|335.8|335.8|
| Gaussian FK |4096|16|.02774|.01617|2678.5|2731.8|
| Gaussian FK |512|128|.05356|.03441|480.3|253.8|
| Gaussian FK |4096|128|.01287|.00810|3777.2|2051.5|
| Gaussian terminal importance |4096|128|.01098|.01508|1302.1|4096|
| Mixture FK |512|16|.12444|.11005|—|—|
| Mixture FK |4096|16|.09002|.03404|—|—|
| Mixture FK |512|128|.07985|.09789|—|—|
| Mixture FK |4096|128|.03414|.02850|3154.2|2930.3|
| Mixture terminal importance |4096|128|.03520|.04461|1658.4|4096|

These are means across seeds, not deterministic error bounds. Gaussian FK mean
error standard deviations and every per-seed metric are in the JSON. Increasing
both budgets improves these reported tilted moments; no claim of monotonic
sample error or proof of asymptotic convergence follows. The Gaussian base
covariance error at P4096 falls .18234→.05043 when N16→128; base mean error changes
.01207→.01456, illustrating Monte Carlo variation. The independent unit check
uses three seeds/P4096/N128 for base and tilted moments with statistical tolerances.

Mixture reference mean=(1.086388904,.062068966), covariance diagonal
(1.407706999,.248275862). Positive x occupancy is computed analytically from the
updated mixture: .8303830005, recorded as target_positive_occupancy. Measured FK occupancy
at4096/128 is .82309; terminal importance .82026. This is an analytic two-mode
prior, not learned-base evidence. Importance sampling can be better on easy
Gaussian metrics; FK is not required to win.

## Learned heat comparison, including the negative outcome

Primary: `outputs/fk-20261025/heat-comparison-verified/`:
`config.json`, `validation-selection.json`, `comparison.json`, `quality-cost.png`,
and each `seed-{11,23,37}/{mode}/{metrics.json,result.npz}`. Checkpoint is
`outputs/accepted_cpu_2d/training/inference.pt`, common fixed evaluation is
`outputs/online-chemistry-20261021/throughput-final-heat/evaluation/`.
One independent validation problem and one test problem, P16/N8, three sampling
seeds. FK observation weights{.0001,.001} and PDE weights{0,.0001} were selected
using validation observation RMSE+.001 FD residual, with no truth. Selected
weights are .001/.0001; gradient observation strength .3, physics .0001. Fixed
kappa.2, ESSfraction.7, resampling every2 were set before test use. No test result
selects a training budget. The soft tempered target is not the ordinary Gaussian
posterior because the observation coefficient is .001.

| Method | Observation RMSE | FD PDE RMSE | Unobserved relative L2 | RMS spread | Batch model calls | Particle forwards | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unguided ODE |1.23466|21.32424|.99853|.80837|19|304|.03637|
| Observation/PDE gradient |.66742|20.31941|.92866|.75912|26|416|.06786|
| Terminal importance SDE |1.00149|20.86677|1.27789|.47040|16|128|.02059|
| FK SDE |.95354|20.65644|1.22522|.48328|16|128|.03090|

The error is the relative L2 of the population mean on unobserved points, against
held-out truth opened only after inference. FK/IS summaries use weights; ODE
modes are equal-weight. FD residual is in physical units; saved metrics also use
spectral spatial derivatives, separate from the steering FD operator. Neither is
a continuum certificate. Small ensemble coverage is descriptive, not calibration.
Gradient evaluation counts, energy/endpoint calls and lifetime process peak RSS
are saved separately; particle forwards do not include the additional backward
cost. Wall times are tiny measured CPU trials, with possible host contention,
not a general throughput ranking. No isolated GPU peak-allocation measurement
was made; lifetime RSS must not be interpreted as per-phase accelerator memory.

**Negative result:** FK improves observation fit over unguided generation but
worsens unobserved reconstruction error. Gradient guidance is better here at
greater measured cost. Importance final ESS is1.59/1.45/1.41 out of16. FK ESS is
12.18/13.73/11.99, but only6/2/3 root ancestors survive: high post-resampling ESS
does not establish diversity. No failed or collapsed seed is removed. Successful
primary runs report zero nonfinite populations; controller criteria remain
unconfigured, so terminal completion does not imply criteria met. Undertrained
checkpoints and one held-out problem establish integration, not scientific
superiority. The older conventional heat Fourier inverse and Darcy inverse
remain better in their retained studies; heat test quality across0/100/500
updates was non-monotonic.

## Tests, browser and backends

| Level | Actual evidence | Practical limit |
|---|---|---|
| Analytic target | Gaussian and mixture moments, step/population variation, algebra, telescoping, offspring statistics | Finite Monte Carlo evidence, no universal convergence theorem |
| Learned prior | Explicit numerical SDE law and its physical tilt; portable checkpoint used | Learned score/ODE consistency is approximate |
| PDE integration | CPU heat1D/2D/true3D, Darcy2D, chemistry3D; API/reference isolation and returned-field verification | Darcy/chemistry are integration probes, no FK accuracy study |
| Scientific reconstruction | Fixed heat validation/test comparison with all seeds | Negative unobserved-error result; insufficient for robust ranking/calibration |
| CPU | Complete Python regression plus actual CLI/API/browser and analytic studies | Small fixed grids |
| MPS | Actual float32 model/energy on MPS, explicit CPUfloat64 weights, all four PDE probes below | Tiny populations, no large-grid speed claim |
| CUDA / actual Colab | **Untested** | No execution claim |

Regression logs are under `outputs/fk-20261025/`. `python-verified.log` is the
complete final numerical suite (143 passed in86.70s); the later figure-only
refactor was exercised by regenerating both saved figures without rerunning
simulations. Earlier `python-final.log` recorded143 passing tests before
the final max-subtraction stability regression and cadence change. Unit tests
cover huge equal finite log weights, nonfinite failure without uniform fallback,
B/P isolation, joint field/parameter ancestry, descendant noise, deterministic
chunking, zero energy, weighted metrics, cancellation/budgets and terminal output.
Integration tests include heat/Darcy/chemistry actual exports, true3D, exact
checkpoint/request validation, API weighted slice calculations and reference
attach/remove preserving fields, weights, ancestry, RNG metadata and actions.

`frontend-tests-verified.log`:20 tests. `frontend-build-verified.log`: TypeScript
and Vite build. `frontend-lint-verified.log`: zero errors, three existing shadcn
fast-refresh export warnings. Real Chromium `browser-initial.json`:15 workflows
passed (12 previous-family/process workflows plus FK1D/2D/3D). Final
`browser-fk-verified.json`:four FK workflows passed, adding incompatible settings
and cancellation. The final combined rerun passed **16 workflows in100.31s**, zero skips/failures/
flaky results: `browser-verified.json`. These tests call the real numerical API and checkpoints;
component mocks are not counted as browser numerical integration. Screenshots
and actual job histories are under `browser/`.

MPS primary: `mps-verified/backend.json`, P8/N4/chunk4, three actual systematic
resampling events per case,8 batch/32 particle model evaluations and32 particle
energy evaluations. CPU/MPS weighted-mean field RMSE:

| Case | Grid | RMSE | MPS inference seconds | Weight-transfer seconds |
|---|---|---:|---:|---:|
| Heat2D |8²|4.21e-8|.69475|.00308|
| Heat3D |8³|5.26e-8|.36924|.00408|
| Darcy2D |8²|2.31e-8|.33384|.00402|
| Chemistry3D |8³|2.78e-8|.44155|.01005|

The same probe executes gradient controller diagnostics, separate real MPS
heat8²→16² fresh refinement, final-field verification, cancellation marked
incomplete, NPZ/weights CPU reload and CPU checkpoint sampling. Actual model
device is mps:0; no hidden model fallback. `gaussian-mps-verified/results.json`
and `gaussian-cpu-crosscheck/results.json` contain four-seed P512/N32 distribution
checks; `cpu-mps-moment-check.json` records mean differences below.049 and
covariance differences below.078, inside stated Monte Carlo bounds. Cross-device
bitwise identity is not promised.

Failures retained: initial MPS combined CPU/double transfer and two toy summary
conversions attempted float64 on MPS; corrected to explicit `.cpu().double()`.
A browser assertion matched two alerts; the actual API rejection was correct,
and the selector was narrowed. Earlier implementation/JSX test failures are
retained in development logs, not counted as successful runs. Sandbox loopback/
OpenMP/MPS visibility required reviewed host execution. No approval control was
bypassed; no action remains blocked. Every successful primary result above is
from actual execution during this continuation, not inferred from an old claim.

## Copy-paste execution

Run from the repository root in the activated Python environment. Python3.12.3,
torch2.5.1, NumPy2.1.3 and Homebrew Node22.23.3 were used. Existing exports and
measurements stay in place. Choose fresh output directories if rerunning.

```bash
source .venv/bin/activate
python -m pdeflow.fk.toys --kind gaussian --output outputs/my-fk-gaussian
python -m pdeflow.fk.toys --kind mixture --output outputs/my-fk-mixture
python -m pdeflow.fk.toys --kind gaussian --device mps --quick --output outputs/my-fk-mps-gaussian

pdeflow reconstruct --config configs/experiments/fk_heat_2d.json --mode fk \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --problem outputs/accepted_cpu_2d/data/test_problem.json \
  --observations outputs/accepted_cpu_2d/data/test_observations.json \
  --output outputs/my-fk-heat
pdeflow evaluate --result outputs/my-fk-heat/fk \
  --reference outputs/accepted_cpu_2d/data/test_reference.npz

pdeflow reconstruct --config configs/experiments/fk_darcy_2d.json --mode fk \
  --checkpoint outputs/multispecialist-20261021/cpu-2d/training/inference.pt \
  --problem outputs/multispecialist-20261021/cpu-2d/data/test_problem.json \
  --observations outputs/multispecialist-20261021/cpu-2d/data/test_observations.json \
  --output outputs/my-fk-darcy
pdeflow evaluate --result outputs/my-fk-darcy/fk \
  --reference outputs/multispecialist-20261021/cpu-2d/data/test_reference.npz

pdeflow reconstruct --config configs/experiments/fk_heat_3d.json --mode fk \
  --checkpoint outputs/accepted_cpu_3d/training/inference.pt \
  --problem outputs/accepted_cpu_3d/data/test_problem.json \
  --observations outputs/accepted_cpu_3d/data/test_observations.json \
  --output outputs/my-fk-3d

python -m pdeflow.fk.experiments backend --device mps --output outputs/my-fk-mps-check
python -m pdeflow.fk.experiments comparison --device cpu --output outputs/my-fk-comparison
# Larger inference study; configured, deliberately NOT launched in this task:
python -m pdeflow.fk.experiments comparison --device mps --particles 64 --steps 64 \
  --checkpoint outputs/accepted_cpu_2d/training/inference.pt \
  --evaluation outputs/online-chemistry-20261021/throughput-final-heat/evaluation \
  --output outputs/my-fk-larger-comparison
```

The comparison CLI's `--checkpoint` and `--evaluation` are configurable. It does
not take `--config`. `configs/experiments/fk_larger_comparison.json` is instead a
valid complete package config for CLI reconstruct at P64/N64, with larger
budgets. MPS variants `fk_mps_*.json` select float32 MPS; any reconstruct config
also accepts explicit `--device mps`. Merely changing a budget does not make the
smoke checkpoint scientifically useful. `cli-heat-2d/`, `cli-heat-3d/` and `cli-darcy/` contain
executed CLI results, in addition to programmatic/backend/API execution.

Workbench launch with supported Node on PATH:

```bash
source .venv/bin/activate
node --version
export PDEFLOW_MODEL_ROOTS="outputs/accepted_cpu_2d/training:outputs/accepted_cpu_3d/training:outputs/multispecialist-20261021/cpu-2d/training:outputs/online-chemistry-20261021/chemistry-3d/training:outputs/process-20261025/heat-fine-smoke/training"
npm --prefix frontend ci
python scripts/dev.py
```

Open **http://127.0.0.1:5173** (API http://127.0.0.1:8000). Shortest workflow:
insert Heat → confirm compatible8² checkpoint → Continue to observations →
Generate synthetic observations → Configure reconstruction → select FK →
P8/N6, ESS1/every1 for visible resampling → Validate complete request → Run.
Inspect weighted result, actual ESS/actions/root ancestors and terminal status.
Reference stays detached; attach it only for evaluation. Reopen via Runs.
For3D select the matching8³ checkpoint and explicitly review/apply its layout;
normal/index controls select real spatial slices. Darcy/chemistry require their
own compatible full mathematical request and checkpoint; FK cannot override
routing or enable unsupported equations. A completed path with unconfigured
physical thresholds still reports FINISH_BUDGET_EXHAUSTED, not criteria met.

For separate terminals use `python -m pdeflow.workbench.api` with the same model
roots, then `npm --prefix frontend run dev -- --strictPort`. These entry points
were used for actual browser verification. The combined launcher's startup and
clean shutdown are preserved earlier process evidence. Ctrl-C stops it.

## Colab, packaging and next slice

Use `notebooks/colab.ipynb`: upload/extract the source ZIP, configure the chosen
persistent output/evaluation path, install the package, choose a compatible
online/offline/hybrid family profile, then launch/resume the package trainer.
The notebook contains training orchestration, not FK formulas. Export inference.pt
and its public config/problem/observations back to the MacBook. Training resume
uses checkpoint.pt with its optimizer/RNG/provider state, not inference.pt.
For source-only archives generate compatible demo artifacts as described in
SOURCE_PACKAGE.md; accepted local checkpoints are deliberately omitted.
With a Colab-trained different layout, derive the inference config from that
export and add an explicit inference.fk block; the supplied8-grid configs cannot
be assumed compatible with a larger checkpoint. Select `--device cpu`/mps on the
Mac and register its training folder in PDEFLOW_MODEL_ROOTS. The same CLI FK
path is available after loading a compatible independent-Gaussian linear-CFM
export. Actual CUDA and Colab FK execution remain untested.

Current source archive: `outputs/fk-20261025/pdeflow-fk-source.zip`, with internal
SOURCE_MANIFEST.json and SOURCE_PACKAGE.md; package source/frontend lockfile,
configs, tests, scripts, notebook and documentation are included. Runtime data,
weights, caches, accepted output trees, scientific reference copies, .git and
personal files are excluded. All earlier archives remain at their original
paths. Current evidence root is `outputs/fk-20261025/`; machine summary is
`verification-summary.json`. Git main still has no HEAD/commit or remote;
90 original staged files remain staged, with later changes unstaged/untracked.
No stage/reset/commit/push or permission changes were attempted.

Missing branches include coupled multilevel FK/refinement, adaptive SDE steps,
rejuvenation/proposal corrections, inference resume, diffusion/CFG backends,
new geometry/boundaries and other PDE families. Next scientific experiment:
first improve and independently validate the learned heat prior, then compare
FK/IS/gradient against multiple fixed held-out problems at matched particle
forward/backward and wall-time budgets, validation-only energy selection,
ancestry/diversity and calibrated uncertainty metrics. Retain the conventional
inverse and all negative outcomes. This architecture is not universally complete.

## Analytic variation and cost detail

At P4096/N128, mean-error ± population standard deviation across four seeds:

| Target / method | Mean error ± SD | Seconds | Batch calls | Particle forwards | Particle energy evaluations | Resampling events |
|---|---:|---:|---:|---:|---:|---:|
|gaussian / base|0.01456 ± 0.00674|0.1246|2048|524288|0|0|
|gaussian / importance|0.01098 ± 0.00240|0.1251|2048|524288|4096|0|
|gaussian / fk|0.01287 ± 0.00658|0.1828|2048|524288|524288|2|
|gaussian / no_resampling|0.01098 ± 0.00240|0.1800|2048|524288|524288|0|
|mixture / base|0.02589 ± 0.00959|0.2023|2048|524288|0|0|
|mixture / importance|0.03520 ± 0.00446|0.2023|2048|524288|4096|0|
|mixture / fk|0.03413 ± 0.00748|0.2536|2048|524288|524288|1|
|mixture / no_resampling|0.03520 ± 0.00446|0.2548|2048|524288|524288|0|

These are analytic callback calls, not neural training/solver calls. Full ESS
trajectories and max weights are in every cost.trace; no-resampling spends
intermediate energy evaluations despite matching terminal importance weights.
The regenerated mean-errors.png figures separate N16 and N128 panels and show
seed standard deviations, avoiding a line that confounds step count and P.
No new simulation or selection was performed to redraw them.
