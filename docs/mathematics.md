# Implemented mathematics and conventions

All physical variables in this slice are nondimensional. Spatial coordinates
are x_j=o_j+i L_j/n_j, i=0,...,n_j-1; the periodic endpoint is excluded.
The spatial axes x,y,z are separate from physical time tau, generation time s,
optimizer step k, problem B and candidate particle P. Physical fields are
[B,C,T,*space]; candidate outputs are [B,P,C,T,*space]. Scalar u is a signed
diffusing field, not a chemical concentration. D>0 is known; the initial field
and all saved solution frames are unknown except at declared measurements.

## Reference physics and operators

Heat: partial_tau u = D Delta u on a periodic Cartesian box. For Fourier
wave vector k_j=2 pi m_j/L_j, each initial Fourier coefficient evolves as
u_hat(k,tau)=u_hat(k,0) exp(-D |k|² tau). NumPy FFT evolution is the CPU float64
reference, independent of the PyTorch finite-difference residual. No learned
prediction is used as reference. Initial profiles filter independent Gaussian
node values in Fourier space by exp(-|m|²/(2 b²)). The deterministic multiplier
sqrt(N/sum filter²) sets expected point variance to one, without per-record
normalization. All modes of the discrete grid can be present.

Central periodic differences use (u[i+1]-u[i-1])/(2 h) for first derivatives and
(u[i+1]-2u[i]+u[i-1])/h² for second derivatives. Divergence sums derivatives of
the corresponding named components. Spectral Laplacians multiply by -|k|².
Tests use nonunit boxes and shifted origins as well as true multidimensional
products, with observed second-order convergence.

Poisson: -Delta u=f. A periodic solution exists only for mean(f)=0. The solver
rejects incompatible means above the stated tolerance, sets the zero Fourier
coefficient of u to zero, and divides other coefficients by |k|². The gauge is
zero spatial mean. This is neither a general Neumann nor a geometry solver.

The scalar expression grammar supports named fields, constants, coefficients,
sum, product, powers 0–3, physical-time derivatives and Laplacians. JSON nodes
are validated; configuration strings are never executed. Gradient/divergence
are numerical APIs, not vector expression-tree nodes in this slice. Heat and
Poisson scalar expressions are tested. The neural encoder does not consume
expression trees or understand unseen equations.

## Data, transforms and observations

Split complete trajectories 60%/20%/remainder using a dedicated split RNG.
Per-record seeds and stable IDs, exact split indices and a content hash are
saved. The training subset alone supplies per-field mean mu and standard
deviation sigma (population convention), reduced over problems, times and
space. Model fields are q=(u-mu)/sigma. Decode u=sigma q+mu before observations
or physical differentiation. Save mu, sigma and fitted training IDs. No test
mask, target, per-test normalization or hidden initial condition is supplied
to the network. Input conditions are the heat family feature and log D.

On-grid point observations select explicit (field,time,spatial-index) tuples.
Physical coordinates are saved and checked against indices. Indexing remains
differentiable in field values; off-grid interpolation is not implemented.
Spatial sensors are chosen uniformly without replacement, then observed at
all saved times. Independent noise has standard deviation sigma_y. A Boolean
mask distinguishes missing values from observed zero.

The observation potential per candidate is
E_obs = (1/(2 M)) sum_valid ((H(u)-y)/rho)²,
where rho=max(sigma_y, configured noise_floor). For zero noise the positive
floor is an explicit soft-constraint tolerance; noisy runs also use the floor
to temper small-noise gradients. This mean-scaled, tempered potential is not
the full Gaussian log likelihood, and its scale is reported.

The differentiable PDE potential uses central time differences (second-order
nonuniform-grid coefficients through torch.gradient) and periodic spatial
finite differences. Only interior saved physical times enter the residual:
r[t]=d_tau u[t]-D Delta_h u[t], 1<=t<T-1. No hidden initial condition is
enforced. Trapezoidal physical-time weights restricted to interior frames
weight E_pde=1/2 mean_weighted(r²). Constant spatial cell volumes cancel in
these normalized means. Diagnostics additionally use a spectral spatial
Laplacian with the same finite time difference. Neither coarse-time residual
is an exact continuum constraint. Periodic topology is imposed by the grid;
there is no independent boundary residual. Seam-gradient magnitudes are
reported as variation, not constraint violation. Spatial integrals measure
mass drift across physical frames.

## Flow matching and endpoint estimate

The implemented path is X_s=(1-s)Z+sU, with independent Z~N(0,I), training
trajectory U in transformed coordinates, and s~Uniform[0,1). The loss is mean
squared error against U-Z. The target is a conditional straight-line velocity;
the optimal network estimates v_s(x,c)=E[U-Z | X_s=x,c]. Generation integrates
dx/ds=v_theta(x,s,c) from independent Gaussian source at s=0 to s=1. There is
no diffusion objective, score model, condition dropout, or classifier-free
guidance. See [Lipman et al., Flow Matching, v2](https://arxiv.org/abs/2210.02747v2)
for the general formulation; this implementation uses the path stated here.

From U=X_s+(1-s)(U-Z), conditional expectation gives
E[U|X_s=x,c]=x+(1-s)v_s(x,c). We use the learned approximation
U_hat_s(x)=x+(1-s)v_theta(x,s,c). It is an estimated conditional mean, not a
recovered unique training endpoint. At s=1 define U_hat=x and omit the model
call. At s=0 the same algebra is finite; there is no division by s or 1-s.

The FK continuation implements the following path-specific score identity, derived
by Gaussian conditioning for s<1: score_s(x)=-E[Z|X_s=x]/(1-s).
Since s v_s(x)-x=-E[Z|X_s=x], score_s=(s v_s-x)/(1-s). Its assumptions are an
independent unit Gaussian source and this linear path; it is singular at the
data endpoint and is not a general score/velocity conversion. The legacy ODE
remains deterministic; FK uses the algebraically finite drift correction and
vanishing diffusion derived in fk-mathematics.md.

## Numerical generation and heuristic steering

Euler: x*=x+h v(x,s). Heun: k1=v(x,s), k2=v(x+h k1,s+h),
x*=x+h(k1+k2)/2. Uniform nodes are s_j=j/N; quadratic nodes are (j/N)².
These are generation schedules, not physical-time meshes. Stepper tests
against x'=x verify order one/two and N/2N forward evaluations. Changing
the schedule or stepper requires only inference configuration changes.

After each ODE proposal at s_next, the optional split correction computes
g=grad_x [a E_obs(decode(U_hat_s_next(x))) + b E_pde(decode(U_hat_s_next(x)))].
Autograd differentiates through the endpoint model prediction and decode.
Clip g per candidate to configured Euclidean max_norm G, and apply
x_next=x* - eta h clipped(g). The energy is summed across independent
candidates before differentiation; each candidate's internal energy is a
mean, so adding candidates does not shrink their individual gradients.
At s_next=1 the estimator is identity and the final correction is defined.
Default a=.3, b=.001, eta=1, G=10; observation-only sets b=0, unguided skips
corrections. All values are stored in resolved configuration. There is one
gradient evaluation per guided step and one extra model evaluation per guided
step except the last. A model call here processes all independent candidates
as a batch; the count is forward invocations, not scalar candidate evaluations.

This is a distribution-changing reconstruction heuristic. It is not an exact
posterior sampler, an FK algorithm or a claimed reproduction of a published
guidance solver. Observation improvement is tested for a controlled small
step; improvement of held-out error is not guaranteed. Physical penalty and
gradient scales are explicit, fixed defaults rather than test-tuned values.

## Conventional inverse baseline and metrics

The baseline parameterizes the unknown initial field by a tensor product of
real Fourier modes {1,sqrt(2)cos(kx),sqrt(2)sin(kx)}, k=1,...,m per axis, m
below Nyquist. Each basis mode is evolved by the heat multiplier. Let A be
the observed evolved basis, y the valid observations, M their count, and
R_kk=(1+|k|²/(2 pi)²)². Solve
(A^T A/(M rho²) + lambda R)c = A^T y/(M rho²).
This minimizes mean standardized squared misfit plus a smooth initial-field
ridge prior. Defaults m=1, lambda=.001 are fixed before test evaluation.
The generator has a smooth full discrete spectrum, whereas the baseline is
truncated; sparse sensors need not identify every mode. Full-sensor analytic
basis tests establish solver correctness, not sparse-case superiority.

Relative L2 uses spatial cell volumes and trapezoidal physical-time weights:
sqrt(sum w error² / max(sum w truth²,1e-12)). Report unobserved entries
separately, each candidate and the ensemble mean; if no unobserved entries
exist their relative error is null. Also export physical observation RMSE,
tempered standardized misfit, per-field MAE, two PDE residuals, mass drift,
RMS ensemble spread, empirical 2.5%–97.5% interval width/coverage, wall time
and model/gradient invocation counts. Small candidate sets are not calibrated
uncertainty estimates. Process memory is a lifetime RSS high-water mark;
it is not isolated phase memory. Reference-data and offline training costs
are recorded separately from reconstruction.

## Historical FK contract and current implementation

The original contract below motivated the now implemented FK slice (see
fk-mathematics.md and fk-verification.md). A tilted law must state a base distribution, energy and
normalization. Required components are justified stochastic dynamics with
compatible drift/noise and endpoint conventions, endpoint estimator,
incremental potentials that telescope to the stated final tilt, normalized
log weights, ESS, stable resampling, genealogy and explicit RNG ownership.
Repeated absolute exp(-E) weighting changes the target. Adding noise to a
trained ODE does not preserve its marginals without a derived correction.
Proposal changes and rejuvenation need weighting/invariance analysis.

Before PDE steering, implement a tractable Gaussian/quadratic-energy toy
with analytic tilted moments; compare direct importance weighting, repeated
seeds and particle counts, ESS, ancestry and diversity. Only then adapt and
validate on PDEs. [Mark et al., FK-Flow, v1](https://arxiv.org/abs/2509.01543v1)
is a source to study, not evidence for concentration-PDE performance. Its
molecular geometry application is distinct from this project's planned
A+B reversible C system. No paper code was copied. The FK continuation consulted exact revision
f05b3b840a61291bc53611b1aedb21282fd64686; reference files/hashes and attribution
are recorded in fk-mathematics.md. Molecular results are not PDE evidence.

## Bounded budget study and validation selection

The continuation uses checkpoints at 0,100,500 updates on one 2D training
trajectory (16x16 spatial nodes, six physical times, width 16, two residual
blocks, 120 complete trajectories). Every split, observation set and source
seed is fixed. Two validation trajectories compare (observation,PDE) weights
(.1,.001),(.3,.001),(.3,.01),(1,.01) and Fourier ridge .001,.1. Selection minimizes
mean validation unobserved relative L2 at the final budget. Freeze it across
all test budgets. No test reference enters selection. This policy deliberately
separates training-budget comparison from repeated per-budget hyperparameter
selection. It is a two-case, one-seed pilot rather than a convergence study.

The selected weights were 1 and .01, ridge .1. Validation CFM objective fell
from 2.3326 to .8795 to .5743. Mean test reconstruction error did not decrease
monotonically: observation+PDE unobserved relative L2 was 1.1097, .7515, 1.1899.
The conventional inverse was .2364 at every budget. Lower velocity objective
and PDE residual did not ensure better reconstruction. Undertraining,
representation/translation modes and guidance limitations are hypotheses;
none was isolated by this single pilot. Report all budgets and preserve this
negative result; do not select a best budget using these test errors.

## Symbolic input conventions and larger heat profiles

The exact editing grammar, declared rank rules, supported derivative aliases,
source/AST/ProblemSpec boundary and recognition limits are in
[symbolic-editor.md](symbolic-editor.md). 𝒞=(𝓔,λ,Ω,ℬ,ℐ,y,M) describes the
physical request; numerical guidance strengths remain separate. τ (or its
supported physical-time alias t) is never generation coordinate s. Symbol names
are resolved from explicit declarations, with spatial coordinates reserved;
declared subscripted symbols take precedence over derivative shorthand. Static
drafts carry no physical-time initial condition. Only the recognized scalar
periodic heat equation is translated to the current executable residual.

New profile instances use u(0,x)=A F_b(x)+m, where F_b is the existing Gaussian
Fourier field with filter exp(-|integer mode|²/(2b²)) and deterministic expected
variance scaling. Independently sampled b∈[.6,2.4], A∈[.3,2], m∈[-1,1] and
D∈[.03,.12] define the stated distribution; these are nondimensional signed heat
fields. Exact CPU float64 Fourier evolution precedes float32 shard storage.
Every complete trajectory is assigned to exactly one split before sensors.
No per-test-instance normalization is fitted. Streaming training statistics
combine counts n_j, means μ_j and centered sums M2_j with the standard parallel
variance identity M2=M2_a+M2_b+(μ_b-μ_a)² n_a n_b/(n_a+n_b), then scale=sqrt(M2/n),
floored at 1e-8 as in the existing transform. The independent test compares
against direct train-only statistics; train/validation/test roles are fixed.

## Implemented static Darcy specialist

On a periodic Cartesian cell-centred grid x_i=o+(i+1/2)h, the physical unknowns
are positive scalar permeability a and pressure u; f is an explicitly supplied
known field. Solve -div(a grad u)=f with mean(f)=0 and mean(u)=0. The internal
[B,C=2,T=1,*space] layout contains no physical-time evolution or initial state.
Channels are physical (a,u); model channels are affine-normalized (log a,u).
Statistics fit only training instances; exp decodes a without clipping.

For each direction j, a_(i+1/2)=2 a_i a_(i+1)/(a_i+a_(i+1)),
q_(i+1/2)=-a_(i+1/2)(u_(i+1)-u_i)/h_j, and
r_i=sum_j (q_(i+1/2)-q_(i-1/2))/h_j-f_i. Periodic roll indexing shares each
face exactly, so the volume sum of divergence cancels. The NumPy reference
applies this SPD stencil matrix-free in the zero-mean subspace with projected
CG and checks the actual residual before success (default relative tolerance
1e-10, limit4000). Incompatible supplied f is rejected, not silently projected;
generator construction explicitly removes its independently drawn f mean.
A separate tensor implementation evaluates fluxes/residuals on decoded fields,
with gradients through a, u, exp, gauge subtraction and endpoint estimation.

Independent manufactured solutions use omega=2pi, u=sum_j sin(omega x_j),
a=1+0.1 sum_j cos(omega x_j) (or a=1), and analytically differentiated
f=a omega² u+0.1 omega² sum_j sin(omega x_j)cos(omega x_j) for the variable case.
The coefficient stays positive in true 3D. Refinement at 8/16/32 cells per axis
shows second-order solution error; this checks continuum consistency separately
from the generating discrete stencil. These tests do not validate learned
inverse accuracy on the coarse smoke grids.

Every physical decode subtracts the pressure spatial mean. This gauge
projection changes the generated distribution; it does not solve the PDE or
restore exact sampling marginals. The removed mean RMS and final decode timing
are reported; guidance decode work is charged in total sample seconds. The
same CFM endpoint and heuristic split gradient rules already documented above
apply; no new score conversion, posterior guarantee or SDE rule is introduced.

Darcy observations may select either physical channel or both. The low-level
ObservationSet accepts per-field noise/tolerance scales; the current
nondimensional UI presets explicitly share one scale. Static rows omit time.
Evaluation reports per-field ensemble-mean relative L2/MAE on all/unobserved
cells, flux residual RMSE, positivity and gauge. The combined L2 is dominated
by the larger-amplitude permeability, so it must not substitute for pressure
error. Heat spectral-time and mass-drift diagnostics are unavailable for static
Darcy, rather than being relabelled as independent Darcy checks. The Darcy
conventional inverse now uses the positive restricted Fourier parameterization
and verified implicit adjoint described below and in darcy-study.md.
Pressure-only permeability identifiability and ensemble calibration are not guaranteed.

## Online supervised reference examples and frozen coordinates

Online U is a newly sampled physical problem solved by the reference method,
then encoded; it is never a neural prediction reused as truth. ID allocation
hashes versioned physical distributions, solver specification, root seed,
namespace, logical index and bounded attempt. Clean cached values are independent
of transforms. New modes calibrate only a separate training namespace and freeze
its exact affine/nonlinear transform before optimizer updates. Calibration IDs
are not ordinary training presentations. Validation/test references and sensors
remain fixed, independent namespaces; mode changes do not change physical IDs.

The existing CFM derivation is unchanged in model coordinates. For chemistry
U is normalized sqrt(concentration). Decode(x)=q(x)^2. It is nonnegative with
exact zeros and no concentration clipping, but folds negative root branches.
The decoded endpoint estimate is square of an estimated root-coordinate mean,
not generally the conditional mean of physical concentration. The same caveat
applies to exp decoding in Darcy. Physical guidance differentiates through the
full decode and endpoint model; it remains a heuristic, not a posterior claim.

## Conventional Darcy inverse with implicit differentiation

A candidate a=exp(.2 tanh(z_o)+(.2+.4 sigmoid(z_A))tanh(B theta)) uses an additive
truncated Fourier B and explicit support approximately(.449,2.226). Pressure is
always solved from A(a)u=f with the periodic harmonic FV operator and zero mean.
The objective is mean tempered Gaussian misfit plus weighted Fourier ridge and
weak offset/amplitude ridge. Zero initialization and Armijo BFGS use no hidden
truth. Degree, ridge, stopping, forward/adjoint iterations and failures are saved.
The solver verifies true relative residual1e-10 rather than treating an arbitrary
unrolled iteration as converged. Twenty-four optimization updates are a cap;
max_iterations does not assert inverse optimizer convergence.

Let g_u=dJ/du and A(a)lambda=P_zero_mean g_u. Differentiating the constrained
forward equation yields dJ/da=direct-g_lambda where
 g_lambda= lambda^T (dA/da)u.
For face H_ij=2a_i a_j/(a_i+a_j), the objective sensitivity is
 -(lambda_j-lambda_i)(u_j-u_i)/h².
Its contribution to cell i multiplies2a_j²/(a_i+a_j)², with the rolled opposing
face included. Chain through exp, tanh and sigmoid. Adjoint-source projection
reflects the pressure gauge tangent subspace; user forcing is not projected.
Finite differences of the complete solved objective verify this in2D/true3D.

Zero forcing gives zero pressure for every admissible permeability, so pressure
alone carries no coefficient information. It is labelled diagnostically and
excluded from principal study cases. Single-forcing pressure observations do
not identify arbitrary permeability; the restricted prior and any coefficient
sensors are essential assumptions. Validation alone selects ridge/guidance;
all training budgets are reported. See darcy-study.md for actual negative results.

## Verified reversible reaction–diffusion

Species obey (a_tau,b_tau,c_tau)=(D_A Delta a-r,D_B Delta b-r,D_C Delta c+r),
r=k_f a b-k_r c. The chosen boundary is periodic, with positive known D and
nonnegative known rates. Local reaction preserves A=a+c and B=b+c. Its scalar
Riccati equation is c'=k_f c²-[k_f(A+B)+k_r]c+k_f A B. Stable roots and the exact
exponential update handle distinct roots; the repeated-root and k_f=0 limits
are implemented explicitly. There is no explicit unstable reaction step or
hidden concentration clipping.

Diffusion uses the positive periodic discrete semigroup. In one axis,
L=(2D/h²)(P-I), P=(shift_plus+shift_minus)/2; hence
exp(dt L)=exp(-lambda) sum_k(lambda^k/k!)P^k. The kernel is truncated at Poisson
tail<=1e-14, then normalized by its retained mass; substeps keep lambda<=16.
Separable axes commute. Strang half-diffusion / exact-reaction / half-diffusion
is second-order in physical time and FD space for smooth solutions. Reference
max_step bounds splitting error; no adaptive global tolerance is claimed.
Pointwise nonnegativity is maintained by positive kernels and exact reaction.
Spatial integrals of a+c and b+c conserve to measured roundoff, not necessarily
bit-exactly; the sum a+b+c generally does not conserve. Numerical tests use an
independent RK4 reaction reference, analytic diffusion and time/space refinement
in true2D/3D before training.

Physical residuals use decoded species, independent central physical-time
and FD/spectral-space derivatives at interior saved times, and the same stated
mass-action law. The learned known-parameter encoder receives log D and log1p
rates only. Square decoding guarantees positivity, not stoichiometric invariants;
ensemble invariant violations and negative root-coordinate fractions are reported.
No species projection is credited to the model. Sparse observations of one
species do not guarantee unique hidden-species recovery.

## Bounded feedback and physical refinement (25 October 2026)

The optional controller never changes the trained path coordinate: each attempt
starts at s=0 from a newly seeded unit Gaussian and follows its configured
Euler/Heun schedule to1. During sampling a check at interval k evaluates the
physical endpoint estimate with autograd disabled solely for policy diagnostics.
The separate SplitGradientGuidance evaluation has autograd enabled through the
model endpoint/decode, exactly once after the full ODE proposal; its negative
gradient correction is eta*h, including nonuniform schedules. No correction is
applied separately to Heun sub-stages. Input gradients do not accumulate model
parameter gradients. Generation-step refinement is not physical refinement.

For each observed field, report raw residual e=H(U)-y (null where mask=false),
RMSE, maximum absolute error and e/sigma where sigma>0. Noise-free standardized
residuals are undefined/null, while energy uses positive stated tolerance.
Guidance and policy's tempered misfit use rho_i=max(sigma_field,tolerance_field):
E_obs=.5 sum(mask_i*(e_i/rho_i)^2)/sum(mask_i) per candidate. This is independent
Gaussian, count-normalized, explicitly tempered scaling; correlated noise is
unsupported. Duplicate/conflicting rows and off-grid/time coordinates are
rejected. No spatial or physical-time interpolation is silently applied.

PDE report components use the worst candidate's own RMSE per equation, weighted
by normalized physical-time trapezoid weights and uniform spatial cell volume.
Transient derivative checks use saved interior times only. Darcy uses static
harmonic flux divergence minus actual known f. Positive a and pressure gauge
are separate. Heat checks drift of integral u; chemistry checks drift of
integral(a+c) and integral(b+c), never conservation of a+b+c. Reports keep
physical units, counts and separate observation/physical component thresholds.
There is no weighted total that can conceal a mandatory component failure.
Boundary periodicity is a representation property, not an independently passed
boundary residual; unknown initial data has no invented target residual.

An estimated endpoint cannot finish an attempt. Consecutive passing checks
(patience) disable subsequent configured guidance, and any failed check restores
it. At the actual final decode, all mandatory components must have finite,
configured, passing bounds before FINISH_CRITERIA_MET. Missing thresholds mean
unresolved, not passed. Bounds are user/problem-specific upper tolerances;
zero positivity-violation count is the only automatic exact admissibility test.
The process's physical report uses the steering discretization. Existing final
metrics additionally evaluate spectral space for transient problems; neither
certifies continuum accuracy or reconstructive correctness. Optional truth
errors are computed strictly after inference and never enter the policy.

Stagnation means no configured normalized maximum-component improvement above
stagnation_relative over patience recent checks. It never resets generation
time mid-path: available retries/refinement occur only at completed candidate
boundaries. Fresh attempts increment the seed by one modulo2^63, return the last
population and consume global attempt/model/guidance/sampling budgets. There is
no truth-based or best-score selection; retries and adaptive guidance still
change the output distribution and have no unbiased posterior guarantee.

One physical refinement level is supported for periodic nodal heat: N_j to2N_j,
same domain, boundary, D and physical times. Measured node indices multiply by2
and their physical coordinates/y/noise/masks remain unchanged. Diagnostic
restriction injects every other fine node; diagnostic periodic multilinear
prolongation inserts adjacent-node averages. These transfers do not generate
fine detail and are never supplied as fine Gaussian noise or measured evidence.
The fine model is independently loaded and compatibility checked; inference
restarts from fresh Gaussian seed+1 with only the same actual observations and
known conditions. This is a related fresh fine-grid reconstruction, not a
trained coarse-to-fine conditional bridge. Coarse truth is not interpolated
and reported as fine truth. No temporal, cell-centred, local AMR or other-family
refinement is advertised. The later FK slice adds weights, systematic resampling,
score-corrected stochastic dynamics and ancestry, but rejects FK plus refinement.
Its finite-step target and approximations are specified in fk-mathematics.md.

## Implemented FK formulas — 25 October 2026

For the supported path, score=(s*v-x)/(1-s), s<1. Choosing diffusion
sqrt(2*kappa)*(1-s) gives finite drift v+kappa*(1-s)*(s*v-x), with no terminal
clamp/division. Euler–Maruyama kernels define Q_h; the target is exactly the
terminal numerical law tilted by exp[-E], approximated by finitely many particles.
G0=exp(-phi0), Gi=exp(-phii+phi_previous), phi_N=E(actual decoded terminal state)
telescope to exp(-Eterminal). Fixed beta schedules and systematic ESS resampling
are distinct. No gradient proposal is added. Details, analytic Gaussian/mixture
posterior derivations and learned-flow consistency caveats are in fk-mathematics.md.
CPUfloat64 log weights and explicit CPU RNG streams accompany selected-device
physical model/energy evaluation. Final means/variances/quantiles use weights;
explicit terminal resampling resets equal weights with genealogy recorded.

## Orchestration continuation

The everyday commands do not change any mathematical target, observation
likelihood, transform fitting data, solver, guidance/FK proposal or residual.
Online normalization still uses only the calibration namespace. Optional best
inference export minimizes fixed validation CFM loss, not physical test accuracy.
Budget extension retains the existing constant Adam learning rate/optimizer
state; no new scheduler interpretation is introduced. Run-level validation/test
inspection is explicitly labelled and recorded. Scientific caveats and negative
heat/Darcy/FK results remain in their original experiment records.

## Reference experiments and comparison quadrature — 26 October 2026

For an explicit finite Fourier initial state
u0(x)=b+Σ A_k cos(2πΣ k_j(x_j-o_j)/L_j+φ_k), the periodic heat solution is
u(x,τ)=b+Σ A_k exp[-DΣ(2πk_j/L_j)²(τ-τ0)]cos(…). Integer modes are strictly
below Nyquist; this is analytic spectral evolution of a specified finite-mode
continuum input. Evolving FFT coefficients of arbitrary supplied grid values is
instead numerical evolution of the spectral interpolant. Neither solves arbitrary
symbolic PDEs. Darcy retains harmonic FV fluxes/compatible forcing/zero-mean gauge;
chemistry retains the independently verified positive diffusion/exact reaction
splitting, with tolerances and diagnostics saved with the numerical approximation.

Synthetic y=H(Uref)+ε uses the same exact on-grid differentiable observation
operator as inference, independent Gaussian ε and independent sampling/noise RNG
streams. The full initial states, hidden permeability and clean H(Uref) are
reference-only. Current checkpoints require scalar parameters and forcing known;
full target-field conditioning is rejected, not silently added.

For each physical field independently, MSE=Σw(Uhat-Uref)²/Σw, RMSE=√MSE,
MAE=Σw|Uhat-Uref|/Σw and relL2=√(Σw e²/Σw Uref²). Spatial cell volumes and
trapezoidal actual saved-time weights define w; static fields use spatial volume.
Empty masks yield null. Relative L2 is null when reference weighted RMS≤1e-12,
not a denominator-clipped enormous number. Full/observed/unobserved/per-time and
individual-particle summaries are separate. Weighted mean-of-errors is distinct
from error-of-weighted-mean. log-a uses weighted mean(log a); pressure uses its
existing zero-mean gauge without truth alignment. Spread remains empirical.

Native comparison checks equation, units, physical inputs and coordinate metadata.
Explicit fine-reference restriction uses coincident periodic nodes or conservative
cell-volume averages, including matching restricted known fields. Only exact saved
times (optionally the explicit intersection) are evaluated. No hidden interpolation
or learned resolution generalization is implied. Reports retain reference,
reconstruction and evaluation grids and state discretization-error limitations.

## Flow/capacity audit and linked local quantities

flow-matching-audit.md traces the executed Gaussian independent linear path,
all-element normalized model-coordinate MSE, conditional-mean velocity, real Adam
updates and base Euler/Heun integration, separately from gradient guidance, SDE/FK
and numerical references. New backbones do not change the generative formulation.
Passive snapshots store actual X_s, decode(X_s) and decode(X_s+(1-s)v), not a
fabricated physical trajectory. An analytic Gaussian vector field isolates ODE
integration from learning error; learned moments are reported with finite-budget
error. FK assumptions/terminal-law tilt remain those in fk-mathematics.md.

Linked heat point terms use the existing second-order nonuniform-time derivative
and native periodic FD Laplacian. Darcy uses the harmonic-face fluxes and their
conservative divergence minus f, retaining its declared pressure gauge. Chemistry
uses actual k_f ab-k_r c and species diffusion/time/reaction contributions. Terms
use the selected weighted mean or physical particle, never evaluation truth.
No PDE residual is invented for empirical ensemble spread. Reference/mean share
colour range; error has a separate scale. Coordinates/native resolution and any
strided display are explicit; reporting metrics still evaluates native declared
quadrature. See capacity-resolution.md for the actual1024² execution scope.

## Editable analytical heat references and finite consistency checks

For period L_j and origin o_j, an axis cosine has wave number 2π/L_j.
The product of cos(2π(x_j−o_j)/L_j) therefore decays as
exp(−4π² D t Σ_j L_j^{-2}); the single-axis mode uses only one term.
Superpositions evolve each mode with its own squared wave number. Unit-domain
2D product/single-axis and 3D product factors are 8π²D, 4π²D and 12π²D.
Reference expressions use actual physical t, never generation s.

Candidate checking evaluates safe bounded expressions in float64 and uses
independent central coordinate/time differences, periodic value and first-normal-
derivative translations, and agreement with the explicitly supplied initial field.
Tolerances and measured maxima are retained in the reference diagnostics. This
finite test can reject an inconsistent candidate; it is not a symbolic proof over
all coordinates/times. A passing edited expression remains labelled a candidate.
Relative errors retain the existing near-zero norm rule and are accompanied by
absolute errors and per-time reference norms.

## Optional heat latent-coordinate CFM — 27 October2026

The fixed-grid formulations above remain unchanged. `coordinate/` adds an
explicitly separate scalar unforced heat architecture. Spatial x=(x_j), physical
τ and flow s are separate inputs. One checkpoint fixes spatial dimension, box
lengths/origins, periodic boundary, scalar u schema, known D range and physical-time
interval; changing query counts is allowed inside this contract.

Let z_u=(u-m_u)/a_u, with m_u,a_u fitted only on training physical realizations.
A point-set reference encoder maps coordinate/value/known-D features to a fixed
K×W latent A=Eψ(z_u). It cross-attends learned latent queries to a deterministic
bounded subset of full reference points. The subset depends on the reference,
never on future inference queries. Stage1 trains Eψ and Dφ by per-realization mean
pointwise normalized squared reconstruction error. Each realization has equal
weight; multi-reference-resolution training is not implemented in this profile.

After stage1 completes, freeze Eψ,Dφ. Fit per-feature m_L,a_L across training
realizations/tokens, with scale floor0.01; use L1=(A-m_L)/a_L. Stage2 draws
L0∼N(0,I), s∼Uniform[0,1], L_s=(1-s)L0+sL1 and target L1−L0. Its loss is mean
squared latent velocity error per realization. This is the existing independent
Gaussian straight-path CFM objective in a fixed learned representation, not a VAE
ELBO or a likelihood-trained discrete normalizing flow. The decoder is not trained
jointly with a changing flow target. This does not prove that the AE or generated
prior is accurate, noncollapsed or invertible; held-out AE physical/spectral
reconstruction diagnostics are reported separately.

Each observation is (x_i,τ_i,field_id=u,y_i,σ_i). The scalar-field identity is an
explicit fixed u feature; arbitrary field identities are rejected. Value/noise
features use (y_i-m_u)/a_u and σ_i/a_u. Physical x_j becomes (x_j-o_j)/L_j for
periodic Fourier embeddings; physical τ is normalized within its trained interval.
Known constant D uses a log-range normalization, separately from the flow-time
embedding. Arbitrary coefficient fields/forcing/full initial conditions are not
encoded or silently summarized. Observations enter a masked cross-attention set;
a learned null/known-physics token remains valid for zero observations. Zero/nonzero
observation training is explicit before the configured count/noise/dropout mixture.
No observation likelihood is applied a second time as guidance in this branch.

There are no measurement-order positional embeddings. Simultaneously permuting
keys/values and masks leaves attention unchanged up to floating-point rounding.
Masked coordinates/features are zeroed before embedding and excluded from attention.
Query decoder rows have no query-to-query attention, so chunking/concatenation
preserves order and the same function within numerical tolerance. Coordinates use
periodic sine/cosine features, imposing periodic endpoint equality. They do not
force the heat equation, conservation or the correct fine-scale spectrum.

Inference integrates dL/ds=vθ(L,s;observations,D) with fixed-step Heun on[0,1],
then returns u=m_u+a_u Dφ(m_L+a_L L,x,τ,D). No target endpoint, full initial,
reference field or future query grid enters the inverse encoder/velocity. Query
points may be scattered or regular at other resolutions/times inside the declared
physical contract. Fine query counts evaluate the same learned representation;
they do not establish resolution generalization or fine physics. No SDE/FK,
score conversion, exact posterior or exact density is claimed for this branch.
Physical diagnostic autodiff computes ∂τu−DΣ_j∂²u/∂x_j² at a recorded bounded query
subset. Shared-grid comparisons use common finite differences for both models and
retain discretization error. Boundary equality is imposed; unknown initial truth
is evaluated only after generation.

Observation sigma is per scalar measurement. Relative synthetic noise is
σ_i=ρ RMS(reference field at τ_i); zero RMS gives sigma0. Clean samples remain
outside inference. Existing tempered guidance uses max(actual sigma, declared
field/noiseless tolerance). The Fourier ridge inverse now weights each design row
by1/(sqrt(N) max(σ_i,tolerance_i)), preserving the old scalar-noise objective when
all scales agree. Noiseless observations never cause division by zero.

Architectural inspiration: [Perceiver IO, Jaegle et al., arXiv:2107.14795v3](https://arxiv.org/abs/2107.14795v3), flexible latent/query attention; this compact
specialist is our implementation/adaptation, not a reproduction or paper-validated
PDE model. Flow objective attribution remains [Lipman et al., arXiv:2210.02747v2](https://arxiv.org/abs/2210.02747v2). No reference code was copied.

## Integrated representation sanity check

Let m_train be the scalar physical-field mean fitted on training realizations
only. For each held-out validation realization, average physical pointwise squared
error, then average those per-realization MSEs with equal weight. Let E_AE denote
this error for the supervised encoder/decoder and E_constant for prediction
u_hat=m_train. The default minimal gate requires E_AE/E_constant < 1. A configurable
ratio in (0,1] can require a larger improvement. For E_constant <= 1e-15 the ratio
is unavailable and the gate cannot pass automatically; nonfinite results fail.

This comparator does not prove an adequate representation, reconstructability
from sparse observations, fine-resolution generalization or a posterior guarantee.
Absolute RMSE, initial-condition error and spectral-power distortion remain
separate diagnostics. The gate uses validation, never test data. A poor result
requires explicit recorded continuation; execution-only pipeline checks record an
exemption to exercise both stages without claiming accuracy. No sampling formula
or training objective changes are introduced by this orchestration.

## Heat playground conventions (27 October 2026)

The default uses periodic endpoint-excluded physical nodes x_i=x₀+iLx/64,
y_j=y₀+jLy/64 and 31 physical times from 0 to T. Known constant D>0 evolves
FFT coefficients by û_mn(t)=û_mn(0) exp[−D((2πm/Lx)²+(2πn/Ly)²)t]. This is the
numerical finite Fourier interpolant of supplied grid data. The separate product
cosine exact solution is c+A exp[−D((2πkx/Lx)²+(2πky/Ly)²)t]
cos(2πkx(x−x₀)/Lx+φx)cos(2πky(y−y₀)/Ly+φy). Zero modes do not decay;
no erroneous offset removal is applied. Exact cosine and FFT agree independently.

Scattered clean scalar values evaluate that interpolant at the actual x,y,t;
observed y=u_Fourier(x,y,t)+σ ε uses independent standard Gaussian draws. Separate
location/noise streams are spawned from the recorded measurement seed. N means
total scalar observations, even for across-time or fixed-sensor patterns. Clean
values remain evaluation-only. Spatial brush distance uses minimum wrapped axis
distance; drawing changes initial data and adds no forcing. Output 16²/32²/64²
queries and saved physical times align with the reference for comparison.

See heat-playground.md for inverse isolation, metric definitions, playback versus
physical time, finite-grid interpretation and limitations. No accuracy or
posterior-calibration claim follows from execution or a small residual.

## Procedural coordinate heat training (27 October 2026)

The Colab path retains u_t = D Laplacian(u) on the periodic unit square, with
Fourier evolution coefficient uhat_k(t)=uhat_k(0) exp(-D |2 pi k|² t).
The deterministic six-family initial-field recipe uses resolved Fourier modes
and smooth periodic bumps exp(-(sin²(pi(x-cx))+sin²(pi(y-cy)))/(pi w)²).
Widths and Fourier wavelengths are validated against the reference grid.
Signed fields/backgrounds are intentional; this is not a concentration model.
See `coordinate/dataset.py` and `docs/colab-training.md` for exact distributions.

Underlying solutions are split before observations. Physical normalization is a
single mean/scale fitted on a recorded training-only calibration subset, not
per-field normalization. Scattered sensors evaluate the Fourier interpolant of
the native initial field, then add independent Gaussian noise with a per-example
absolute sigma. Output reference arrays remain evaluation/AE-target data and do
not enter conditional flow inference. A flexible decoder grid is not evidence
of training at multiple discretizations or resolution generalization.

For T saved physical times, AE query counts n_t are stratified with no
replacement at each time, and loss is (1/T) sum_t (1/n_t) sum_j error(t,j)²
in normalized units. Thus initial time retains 1/T of the objective even when
query counts are uneven. Flow remains linear conditional flow matching between
standard Gaussian latents and normalized frozen encoded training fields. The
loss is mean latent velocity MSE, independent of the physical output grid.
Microbatch gradients are weighted by actual microbatch_size/effective_batch_size;
one update is one optimizer step, including a smaller final accumulation group.

Adam, optional warmup/cosine decay and clipping are explicit per stage. Extending
a budget retains the saved scheduler horizon and holds its terminal LR beyond
that horizon. Validation gate AE MSE/constant-baseline MSE < threshold is only
a relative baseline test. Fixed validation probes and bounded physical diagnostics
never select a budget using the untouched test split. FP32 losses/parameters,
FP64 streaming moments, optional CUDA autocast/scaling and non-reentrant activation
checkpointing retain these intended objectives; actual CUDA numerical behavior
requires hardware verification. Training requests SDPA; inference retains the
attention path supporting physical second derivatives, without claiming any
particular GPU kernel or measured acceleration.

## Continuous heat physical time and sensor design — 28 October2026

New main heat2D coordinate profiles use explicit temporal_mode=continuous,
time_interval=[0,1]. The old times array remains a snapshot-only legacy contract;
animation_frames is display/export only. No D rescaling accompanies this extension.
If t'=t/T merely relabels an old trajectory, the same evolution needs D'=T D.
Physical derivatives of coordinate features retain 1/L and1/T chain factors via
autograd at physical input coordinates; residuals still use actual physical D.
Flow-matching s in[0,1], physical t in[0,T], optimizer updates and animation frames
remain independent concepts. No generative path/sampler mathematics changed.

For each role, sampling measure is p0 delta_0 + pE LogUniform[aT,bT] +
pU Uniform[0,T] + p1 delta_T, a=1e-4,b=.2. The early density is
1/(t log(b/a)) only inside the bounded strictly positive early interval; there is
no singular density arbitrarily approaching zero. Encoder weights are
(.35,.35,.25,.05), query(.25,.35,.35,.05), observations(.15,.35,.45,.05).
Point budgets >=8 allocate categories by largest remainder, then randomize
within components/order; smaller observation budgets use categorical draws.
The AE objective is mean normalized squared error under those rounded query
weights, deliberately not uniform space-time MSE. The fixed per-solution encoder
set uses a separate reproducible role seed; flow targets cannot drift because
encoder times refresh. Flow freezes that AE and its transforms.

Reference point values are Re sum_k c_k exp(i k_phys dot(x-origin))
exp(-D |k_phys|² t), k_phys,j=2 pi n_j/L_j, from the real initial grid FFT.
This declares a finite Fourier extension, including real-part Nyquist convention;
it is not the unknown continuum drawing. Constant and negative modes remain.
Point evaluation is chunked; discarded coefficient magnitudes <=1e-13 have a
uniform absolute bound equal to their magnitude sum. Original-grid frames use
all FFT coefficients. No temporal frame interpolation supplies training targets.
Reference and streaming training-only calibration statistics are explicitly CPU
float64; model optimization remains on the selected CPU/MPS/CUDA backend.

Validation separates initial, early, intermediate, late and endpoint errors,
absolute/guarded-relative and mean-subtracted errors. The uniform metric is
trapezoidal integration of spatial node-mean MSE at11 equally spaced times;
additional off-frame times populate stratum metrics. The gate's numerator and
constant-baseline denominator use that same uniform quadrature. It must strictly
beat the configured ratio, not just have low residual or attractive pictures.

Sensor heat focusing uses reference-assisted spatial contrast u-mean(u), including
both signs. The design is stratified, not IID: reserve ceil(alpha N) uniform
background points (alpha=.25 default), allocate remaining counts equally across
up to8 detected periodic regions, and jitter with broad wrapped kernels. First
regional anchors have bounded jitter; excess-region selection and count remainders
are seeded. Minimum-separation attempts are bounded and relaxation is recorded.
This guarantees allocation to retained regions when the focused budget permits,
not detection of every source. Positions carry reference information; no claim
of blind sensing, optimal design, random-policy equivalence or exact posterior
follows. Only actual noisy scalar values/positions/times/sigma and known physics
enter conditional inference. Exact policy and measurement provenance are in
continuous-heat.md and saved sampling records.

The constant Fourier mode is never pruned, including tiny nonzero constants.
Arbitrary-time latent views retain the recorded inference device; they do not
quietly move an MPS/CUDA result to CPU. Display-frame caches remain separate.

## Spatial heat display integrals —28 October 2026

For periodic endpoint-excluded u[i,j] on Nx×Ny nodes, mx[i]=(Ly/Ny)Σj u[i,j]
and my[j]=(Lx/Nx)Σi u[i,j]. Origins locate nodes; no endpoint is counted twice.
Magnitude mode replaces u with |u| explicitly. Raw signed integrals retain
background and cancellation; no probability normalization is applied. Reference
profiles use full available arrays before comparison downsampling. Shared cached
interval scales affect display only. Details: spatial-profiles.md.

## Field reconstruction losses —29 October2026

The coordinate heat2D representation now optimizes MSE, MAE, reference-contrast
WMSE, or L=(1−α)MSE_domain+α mean_k MSE_region_k (K0 falls back to MSE). New UI/
notebook runs explicitly select α.5; absent historical config retains MSE and its
point sampler. Per-solution, per-physical-time-group reductions are distinct from
latent flow time. Uniform domain supervision plus a fixed extra region-stratified
budget estimates the blend with detached reference-only four-neighbour periodic
regions. Gradients update encoder and decoder; latent CFM velocity loss is unchanged.
Global training normalization, exact thresholds/quadrature, importance factors,
contribution maps and uniform validation/gate semantics are derived in
field-reconstruction-losses.md. No temporal source identity or scientific benefit
is claimed.

## Structured joint latent flow —30 October2026

Let T=(E(S)-mu_g)/sigma_g for complete state S=(u,u0,D), with train-only
per-group/channel moments and sigma>=.01. Fixed slots do not change with disclosure.
For active-slot mask m, draw xi~N(0,I) and independent generative s~U[0,1]:
z_s=m*((1-s)*xi+s*T), target=m*(T-xi). The velocity output is also masked.
For each solution, average velocity squared error over tokens/channels within each
active logical group, then take a configured weighted mean over active groups;
finally average solutions. Known scalar groups are excluded, not supervised as
random targets. Partial solution fields remain active. No spatial region masks
are assigned to latent tokens.

Representation loss is (L_u+L_u0+MSE_log-range-D)/3 + lambda mean(z²), lambda1e-5.
Field losses use field-reconstruction-losses.md; u0 gets its own t=0 segmentation.
Initial loss inherits the solution loss unless explicitly configured separately.
D uses log(D/D_min)/log(D_max/D_min) and a sigmoid scalar head in that coordinate;
known D bypasses the head and enters typed context. Unknown D values are removed
before context encoding. Other physical quantities/types need declared schemas;
a generic field encoder does not establish an operational PDE specialist.

Coordinates use periodic Fourier spatial features and explicit continuous physical
tau, independent of s. Local selection uses stable nearest periodic anchors and
is piecewise differentiable away from selection boundaries. Math attention uses
FP32 score/softmax reductions for actual second derivatives. A generated latent
is reused across arbitrary supported physical-time queries; it is not evolved by
a mandatory numerical solver. Initial and solution heads are learned jointly,
not constrained to exact equality or conservation.

Optional decoded auxiliary training uses z_s+(1-s)v, not an exact ODE endpoint.
The decoder/normalization stay frozen. Observation likelihood, exact samples,
normalized heat residual, s cutoff and explicit beta weights are documented in
joint-latent-model.md. Bounded latent correction changes distributions; no exact
posterior guarantee is asserted. Euler NFE=steps; Heun NFE=2*steps.

## Operator-aware heat/wave v2

The active generated state contains initial physical quantities, not a trajectory.
Heat q=u obeys u_t=DΔu; wave q=(u,v) obeys u_t=v, v_t=c²Δu. A1 learns spatial
state/input representations. A2 learns E(z,t_a,t_b)=z+(t_b-t_a)F(z,t_a,t_b,graph,
physical context), with zero coefficient increment and absent-state masks. B fits
straight conditional flow matching to frozen initial-state/scalar targets. Physical
time t and Gaussian-generation flow time s are independent. Spatial region weights
only affect decoded physical supervision, never latent-token velocity losses.

Physical derivatives use the chain rule through E and the spatial decoder. Periodic
all-anchor smooth cosine kernels avoid hard routing boundaries. Strong smooth wave
residuals are u_t-v and v_t-c²Δu. Exact reference mean(v) is constant, mean(u) is
mean(u0)+t mean(v0), and E_wave=1/2∫[v²+c²|∇u|²]dx is conserved. Learned values are
measured with grid-dependent quadrature, not assumed. Heat L2 energy decay is not
applied to wave. Stable Fourier zero/Nyquist conventions, quantity-specific training
normalization, per-stage losses and metric reductions are derived/recorded in
operator-wave-model.md. The extension PDF is design guidance, not a proof of our
learned operator, posterior, numerical conservation or generalization.

### Resolved training-family contract

The training selector chooses enabled PDE families, independently of a displayed
reference. Positive family_weights normalize to sampling probabilities; state schemas
are heat:[u], wave:[u,v]. Initial-field shape-mixture labels are not PDE families.
The same family sampler feeds A1/A2/B. The selected reconstruction objective and alpha
apply to decoded normalized physical quantities in A1 and A2, before active-quantity
averaging. A2 additionally retains0.1 latent alignment and0.01 latent/decoded composition
weights by default; B's masked equal-group CFM velocity loss is unchanged. The routing
audit's controlled alpha0/.5 check changes real encoder/decoder optimizer updates.

## Reference acceleration without a new scientific objective

Generated operator-state examples already have |kx|,|ky| <= recipe.max_wave.
Their optimized spectral evaluator omits only out-of-band roundoff after verifying
sum(|discarded u_hat| + |discarded v_hat|) <= 1e-12 max(1, ||u0||inf, ||v0||inf).
Unrestricted reference imports keep all modes, with symmetric Nyquist splitting.
The coefficient evolution is unchanged: heat u_k(t)=u_k(0)exp(-D|k|²t);
wave u_k(t)=u_k(0)cos(c|k|t)+v_k(0)t sinc(c|k|t/pi), and
v_k(t)=-c|k|u_k(0)sin(c|k|t)+v_k(0)cos(c|k|t).
Thus k=0 displacement remains linear in time. Vectorization and exact-key bounded
memoization introduce no temporal quantization. Region queries/masks, alpha=.5
blend and per-solution/per-time/per-quantity reductions remain unchanged. B true
microbatch losses are summed and divided by the same effective batch, preserving
the original CFM objective up to numerical batching/reduction roundoff.

## Batched physical-state objective reduction

For realization i, let Q_i={u} (heat) or {u,v} (wave). The field objective is
F_i=(1/|Q_i|) sum_q L_iq, where L_iq remains the selected per-time stratified
whole-domain/region-balanced loss. The optimizer objective is sum_i J_i/B for the
fixed effective realization batch B; a shorter final microbatch contributes its
sum divided by B. Do not replace this by a pooled average of all quantities or
regions in the batch. Family mixture probabilities remain in realization sampling.
A2 latent alignment/composition use each realization's active-token denominator;
decoded composition also averages quantities within each realization first.
B retains its per-realization/group-masked CFM velocity objective. Batched graph
embeddings share the same differentiable graph network; reuse inside one forward
does not freeze the A2 graph parameters. Exact-time frozen encoder memoization
changes neither the physical-time proposal nor the supervision/conditioning RNG.

## Effective batch and successful-update accounting

For effective B and forward groups of sizes m_j summing to B, the optimizer loss
is sum_j sum_(i in group j) L_i/B. Accumulation is ceil(B/m), including a final
remainder; dividing every group by the group size would overweight that remainder.
Per-example/per-quantity/region reductions are unchanged. Scaler-overflow attempts
consume B examples and fresh sampling RNG but do not update parameters, scheduler
or successful progress. Fixed-example experiments count attempts/exposure; fixed
successful updates need not have equal exposure, especially across B or overflows.
Approximate example-preserving budgets round remaining updates up; the extra
examples and existing two-update B floor are explicit. No LR scaling is implied.

### FP16 attempts and successful updates

A field/latent objective is unchanged by loss scaling: GradScaler scales its
backward pass, unscales accumulated gradients once, checks/clips and steps once.
An invalid-element FP16 attempt is skipped and lowers the scale; it contributes
attempted examples but no optimizer update or learning-rate progress. With an
existing consecutive-overflow limit L, an anchor requiring N successful updates
needs at most N*L attempted updates without first hitting that failure limit.
A finite loss does not imply finite gradients. A finite-element global-norm
overflow must fail before stepping because element-based scaler detection cannot
catch it. See docs/anchor-fp16-recovery.md for failure diagnostics and evidence.

## Reconstruction diagnostic conventions —1 November2026

Physical time t, flow time s, wave velocity v=du/dt and latent flow velocity dZ/ds
are distinct. The production straight path is Z_s=(1-s)Z_noise+sZ_target, target
velocity Z_target-Z_noise; Z_s+(1-s)v is its clean endpoint estimate. The new
oracle initial test encodes complete X0=(u0,v0 for wave), not a trajectory. Oracle
evolution uses true coefficient and E(X0); these are evaluation-only upper-information
controls, not deployable inference.

All physical field metrics inverse-normalize with recorded training-only quantity
transforms. Spatial quadrature is the periodic node mean. Time quadrature uses
normalized trapezoid weights on explicitly saved (possibly nonuniform) times;
RMSE=sqrt(aggregate MSE), never mean(per-example RMSE). Relative L2 is unavailable
when physical reference RMS<=1e-10. Complex Fourier coefficients use FFT/N;
summed squared coefficient error equals node MSE by Parseval, including phase.
Low/high modes are split at radial integer domain-mode index2. This is not a
physical frequency cutoff invariant to a changed domain.

Exact sensor metrics query each actual x,y,t,quantity. Positive sigma gives
standardized residual (prediction-value)/sigma. Sigma0 is reported separately
using tolerance1e-5+1e-4*abs(value); no arbitrary uncertainty is assigned. Four-sample
empirical intervals are descriptive, not calibrated posterior intervals.

The optional latent correction minimizes, equally over samples,

`mean_sensors(((decoded_normalized - observed_normalized)/d_j)^2)
 + beta * mean_active_latent((z_standardized - z_original_standardized)^2)`.

Here d_j=sigma_j/training_quantity_scale for positive sigma; d_j=1 for zero sigma.
The zero-noise term is a deterministic penalty in training-field units, not a
Gaussian noise assumption. beta>0 anchors the original generated latent; it is a
local trust penalty, not evaluation of the learned prior density. Known/absent
groups remain fixed. Frozen T/D are differentiated to the latent at actual sensor
times; reference/held-out data never enters optimization. Final predeclared steps
are evaluated without hidden-reference stopping or sample selection. This changes
the generated distribution and has no exact-posterior guarantee.

For explicit new-run noiseless coverage, `joint.noiseless_probability=p0` samples
p0*delta_0+(1-p0)*Uniform[0,S] with a range including0. Reuse the existing uniform
quantile U: sigma=0 for U<p0, otherwise S*(U-p0)/(1-p0); p0=1 gives0. The event
probability is p0 and the conditional positive quantile is uniform. No extra RNG
draw shifts subsequent conditioning data. The absent/default p0=0 executes the
original sampling code exactly. This option changes B's declared observation-noise
distribution, not A1/A2 field objectives or latent velocity loss; resume changes
are incompatible and old checkpoints are not retroactively relabelled.

### Decoded endpoint likelihood and optional physical/spectral training (1 November)

The operator-v2 straight path remains z_s=(1-s)epsilon+s z_target, velocity target
z_target-epsilon, with the existing active-group-balanced MSE. The new optional B
term evaluates z_hat=z_s+(1-s)v_theta, inverts latent calibration once, evolves T
at each exact sensor physical time and decodes its physical quantity. Independent
positive noise uses .5 r²/(sigma²+delta_model²) plus optional Gaussian constant;
zero noise uses a separately declared finite tolerance penalty, never a silently
floored exact likelihood. Uniform bounded sensor sampling estimates per-case
means (or sums with N/K correction), then averages cases. No observations =>0.
A beta=0 fast path leaves base sampling/decoding untouched. The estimated endpoint
is not an integrated posterior draw and a sensor-fit gain is not calibration.

Optional A2 residuals are (u_t-D lap u)/(a_u/T), (u_t-v)/(a_u/T),
(v_t-c²lap u)/(a_v/T), through actual T/D with FP32 math attention and independent
uniform space/time collocation. Paired supervised dynamics remains essential.
Frozen representation parameters remain frozen while derivatives through them
are retained. Public particle residual coordinates are now independent leaves;
sharing one coordinate leaf incorrectly sums derivatives across particles.

Optional spectral_mse is domain MSE plus lambda times complex-DFT difference on
an extra declared structured periodic N² grid, forward normalization1/N², sum of
all modes with mean-one (1+kx²+ky²)^(p/2) weights. These are dimensionless mode
indices, not an isotropic physical Sobolev norm on non-square boxes. Real
separable DFT gives identical mathematics across devices without scattered FFTs.
p=0 obeys Parseval; phase-shifted equal-magnitude modes retain nonzero error.
Default lambda=0 unless spectral_mse is explicitly selected. No automatic
combination with region balancing or latent-token weights occurs.

Full equations, noise/quantity units, reductions, s activation, warmup, query
costs, defaults and separate ablations: observation-aligned-training.md.

### Standard-MSE baseline default (1 November continuation)

New-run field objectives default to domain MSE: sum of squared normalized errors
against detached reference values with the original spatial quadrature/sampling
weights, reduced per physical-time group/active quantity/realization as before.
Zero-weight invalid points are masked before subtraction, not counted as data.
RB remains (1-alpha)domain MSE +alpha mean(region MSE), alpha.5 experimental,
with unchanged signed periodic regions and corrected stratified estimator.
The standard path removes only regional analysis/query/cache work, not domain
queries or their distribution. The choice affects A1 and decoded A2 only; B FM,
likelihood, physics and displayed metrics remain distinct. Detailed compatibility
and exact selection commands: standard-mse-workflow.md.

### Continuous known measurement times and A2 diagnostics

A sensor is o_j=(x_j,t_j,q_j,y_j,sigma_j), y_j=u_q(x_j,t_j)+epsilon_j,
epsilon_j~N(0,sigma_j²). t_j is supplied information. New B whole-interval policy
uses t_j=a+(b-a)(j+U_j)/N, U_j~Uniform[0,1), followed by a random permutation.
N is total observations, not per time. For an optional recorded mixture, first
sample multinomial counts (whole, log-uniform early, exact t=0); whole weight>0.
The exact-zero component requires a=0; shifted intervals reject that component.
Reference targets use original t_j, not the bounded placement-analysis map times.
Wave v has its own quantity and global normalization. Masked sensors do not enter
set attention or endpoint likelihood. Frozen E/T/D weights do not detach generated
endpoint gradients. No latent-token region weighting or FM change was introduced.

A2 reference targets already evaluate continuous actual ta/tb, with explicit t=0
identity, early log-uniform and randomized whole-interval strata. Its default
25/35/35/5 initial/early/uniform/final mix remains independent from B. Report both
distributions, not merely acceptance of a time input. A2 audit relative L2 divides
by sqrt(max(mean(reference²),(training_quantity_scale*1e-6)²)). Compare learned
T(E(initial),t) against the SAME frozen initial encoding held constant. Reencoding
reference(t) is diagnostic, not a rigorous representation-error lower bound. The
fixed-time aggregate uses trapezoidal time weights, and individual physical metrics
remain available. Spectral power0 is phase-preserving grid MSE by Parseval, not
frequency-selective supervision. See measurement-times.md and a2-investigation.md.

### Trained-model audit reductions

Actual-run oracle aggregates average per-case physical MSE within each family and
quantity, then take a square root for RMSE. They do not average case RMSE or mix u/v
units. Within a trajectory use periodic spatial node quadrature and normalized
trapezoid physical-time weights on explicitly evaluated times. Reference-state
reencoding is a separate diagnostic, not an error lower bound. Large strong residuals
were cross-checked using independent decreasing-step differences; approximate grid
energy and four-point autodiff residuals are labelled as such.

B training component telemetry now applies the SAME per-example averaging over the
entire effective batch as the differentiated loss, including incomplete microbatches.
FM and beta-weighted endpoint contributions remain distinct; sensor counts sum,
raw per-case likelihood penalties average. This is detached logging only. Original
trained-run logs retain their historical last-microbatch meaning; no data or loss was
retroactively altered. See trained-model-analysis.md.

## Targeted B / wave kinematic experiments (1 November2026)

See spatial-conditioning.md for exact reductions. New sensor logits add a learned
bias of sin/cos periodic normalized displacement (local anchors only), physical
sensor time and quantity/group roles; global/scalar tokens get zero spatial bias
features. Attention remains global and set-permutation invariant. Chunked online
softmax uses running maxima, FP32 numerator and denominator; valid null prevents
empty-set normalization failure. Original pooled conditioning remains a baseline.
B's existing straight-path endpoint likelihood is reused, with exact per-sensor
times, explicit Gaussian/discrepancy/noiseless semantics and recorded beta/range.
A1 weighted active-field average defaults to equal u/v and parameter weight.1,
without changing TRAIN-derived normalization. Independent optional A2 kinematic
loss is weight×mean[((physical u_t-v)/(TRAIN u scale / interval duration))²] on
wave-only uniform continuous probes; derivatives traverse actual T and frozen D.
It does not assert latent dZ_u/dt=Z_v, replace supervised u/v or double count the
existing wave_u physics coefficient. Physical and flow times remain distinct.


## Physical mesh candidates —2 November2026

The new M1–M5 mathematical contracts and FM/VP-v/DDIM derivations are implemented
in mesh-family.md (Stochastic equations and Objectives sections). Physical state
is TRAIN-normalized nodal u0/v0 plus affine log D/c; heat-v/known parameters are
masked. M1 directly infers initial state; M2 evolves it by state+tau*learned_delta,
static coefficient exact. State-only smooth periodic decoding supports derivative
losses; representation bandwidth and soft residual probes limit accuracy claims.
Per-example equal active quantities and explicit initial/trajectory weights avoid
pooling solutions/regions/times. Region MSE remains a detached signed periodic
reference-defined stratified estimator; standard MSE is the new deterministic
default. Physical sensor time, query time, flow/diffusion time and training update
are distinct. FGN independent two-draw fair CRPS includes the subtraction term;
zero-noise supervision is not an ensemble mean. M4 straight physical-state FM
and M5 discrete cosine VP-v are genuinely distinct. No sampling conversion or
exact posterior guarantee is asserted. Complete wave presets, c² convention,
matched velocities, periodic pulses and zero-mode formulas are in mesh-family.md.


## Mesh numerical precision boundaries —2 November2026

Autocast can reduce a matrix product even when its operands have been explicitly
cast to float32. Mesh sensor QK products, streaming weighted-value accumulations
and decoder kernel products now execute under autocast-disabled FP32. Linear/
backbone layers keep the explicitly resolved AMP policy. Softmax/normalizers and
physical-state carrier remain FP32; known scalar assignment is exact in that dtype.
Explicit scalar/anchor squared-error helpers cast before subtraction/squaring.
Whole-field shared differentiable MSE already used FP32. No change to quadrature,
per-example/quantity/family weights, region balancing, time distributions, objective
terms or optimization schedule. No nan replacement or arbitrary clamping. This
repairs a demonstrated finite-operand FP16 overflow path; it does not prove the
remote long-run failure originated there or guarantee stable future optimization.
