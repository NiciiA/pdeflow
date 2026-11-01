# Restricted Darcy inverse and bounded online learning study

The conventional inverse is now implemented. It does not optimize unrelated
pressure/permeability arrays: each candidate permeability generates pressure
through the verified harmonic-face periodic CG solve, with compatible known f
and mean(u)=0. Hidden permeability/pressure is never an input to initialization,
optimization, stopping, restart or result selection.

## Objective and gradient

Let g be an additive real Fourier field: a constant plus sine/cosine modes up
to configured degree on each spatial axis, excluding cross-mode products.
Use a=exp(o+A tanh(g)), o=.2 tanh(z_o), A=.2+.4 sigmoid(z_A). This has the same
positive support and map structure as the benchmark coefficient generator,
but a truncated additive field with a quadratic smoothness prior rather than
the full smooth random spectrum. Offset/amplitude are inferred with weak
parameter ridge penalties. This restricted prior does not establish unique
recovery of arbitrary permeability from one forcing.

Minimize .5 mean_i[(M(a,u)-y)_i/rho_i]^2 plus Fourier ridge/smoothness and weak
offset/amplitude penalties. rho is the maximum of stated per-field noise and
tolerance floors, so this is the same explicitly tempered Gaussian potential
used for guidance, not an untempered full likelihood. Initialization is always
zero unconstrained coefficients (a=1); no truth-based warm start/restart.

For A(a)u=f, solve A(a) lambda=P_zero_mean(dJ/du). Then
 dJ/da=direct_observation_term-lambda^T(dA/da)u.
For a shared face H=2a_i a_j/(a_i+a_j), its sensitivity is
 -(lambda_j-lambda_i)(u_j-u_i)/h², with dH/da_i=2a_j²/(a_i+a_j)² and the symmetric
neighbor term. Chain through exp, tanh and sigmoid to Fourier parameters.
The gauge projection here acts on the adjoint derivative, not user forcing.
Finite-difference directional tests verify the full objective gradient in2D
and true3D. Every forward/adjoint solve checks actual CG residual (rtol1e-10);
no fixed unrolled iteration is assumed converged.

A small NumPy BFGS optimizer uses Armijo descent, at most16 backtracking trials,
a preassigned24-update cap, gradient norm1e-5 or relative objective-change1e-9.
It records stop reason, objective trace, forward/adjoint calls/iterations,
solver failures and wall time. max_iterations is an optimizer cap, not proof
of convergence. A zero-forcing pressure-only diagnostic is explicitly labelled
uninformative; it is excluded from the principal benchmark.

## Executed study

Command: `python -m pdeflow.darcy.study --config configs/experiments/online_darcy_2d.json --output outputs/my-darcy-study`.
The completed run is outputs/online-chemistry-20261021/darcy-study.

The8², width8/one-block, batch2 CPU model followed one online training path at
budgets0/20/80 after measuring solver/update costs. It consumed0/40/160 unique
training problems; four separate calibration problems froze its transform.
Two validation and two test physical problems were fixed before training.
Each has pressure-only or permeability+pressure sensors at densities4/12 and
nonzero Gaussian noise .005 (tempering floor .02). These eight scenarios per
split share only two independent physical problems; they are not eight
independent test examples. Forcing is nontrivial and mean-zero for every case.

Validation at the fixed final budget selected observation/PDE weights .5/.01
from four candidates and baseline ridge .1 from {.001,.1}; those choices were
frozen for all test budgets. No budget was selected using test performance.
The complete benchmark manifest contains exact IDs, seeds and sensor/noise
realizations; budget checkpoints are snapshots of the one training path.

Mean unobserved relative L2 errors:

For each case/field this is norm(candidate ensemble mean minus truth) divided
by max(norm(truth),1e-12) over that field's unobserved cells, then averaged across
the eight scenarios. log(a) uses log of the mean physical permeability, not mean
log-permeability. It has its own truth norm and is not averaged into a/u errors.

| Budget | Method | a | log(a) | u |
|---:|---|---:|---:|---:|
|0|Unguided|.30997|1.49943|1.23717|
|0|Observation|.30874|1.49385|1.23334|
|0|Observation+PDE|.30875|1.49402|1.23169|
|20|Unguided|.32999|1.39081|.96819|
|20|Observation|.31319|1.33330|.96068|
|20|Observation+PDE|.31290|1.33265|.95831|
|80|Unguided|.30497|1.36619|.50886|
|80|Observation|.27787|1.25204|.50738|
|80|Observation+PDE|.27780|1.25194|.50603|
|All|Conventional inverse|.16895|.76222|.09630|

Pressure improved with training but remained substantially worse than the
conventional inverse. Permeability error was non-monotonic. Log-permeability
error remained large. These are negative/limited scientific results, not a
validated generative advantage. Six of eight test inverse optimizations reached
the24-update cap; one met gradient tolerance and one objective tolerance.
All recorded forward solves passed their physical residual tolerance:299
forward and299 adjoint calls across eight test inverse cases, zero numerical
solver failures. Do not conflate optimizer nonconvergence with solver failures.

Study.json and per-case JSON retain separate a/log(a)/u errors, observation fit
by field, coarse flux residual, gauge, positivity, spread, model/solver calls,
runtime and stopping. The baseline's steering-grid residual is also its solved
forward discretization, not an independent continuum-accuracy test. The original
manufactured/refinement tests verify that forward method independently.
Representative figures and budget-comparison.png are saved under this run.
Total bounded study time was23.02s on the recorded host, not a scalable-cost
prediction. Training, calibration and reference costs are retained separately.
The earlier heat0/100/500 study is unchanged and was not rerun.

## Workbench and next scientific work

Generic job workers also run the conventional baseline without a reference.
Its observation/physics diagnostics are available immediately; reference-based
errors are computed only when truth is attached after inference. Removing truth
removes those errors. The UI exposes a conventional-inverse comparison and
separate physical-field error tables. CLI reconstruct includes darcy_inverse
alongside the three generative modes; evaluate uses the same saved output.

Next: more independent problems and training/sampling seeds, preassigned larger
optimizer budgets with verified stopping, matched observation designs and finer
or independent physical evaluation. Multiple forcings for one permeability
require an explicit new state/observation design; they were not introduced here.
