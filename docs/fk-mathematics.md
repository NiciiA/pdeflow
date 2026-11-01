# FK mathematics and supported target

## Reference and scope

Consulted Mark, Galustian, Kovar and Heid, *Feynman-Kac-Flow*, arXiv
[2509.01543v1](https://arxiv.org/html/2509.01543v1), 1 September2025, especially
Theorem1, Proposition2 and the difference-potential construction. Reference code:
[heid-lab/fkflow](https://github.com/heid-lab/fkflow), revision
f05b3b840a61291bc53611b1aedb21282fd64686, MIT. Consulted files and hashes are
recorded under outputs/fk-20261025/references/. The paper is CC BY4.0. No source
code is copied into the package; the implementation below is independent.
The molecular application is methodological context, not PDE validation.
Network access required reviewed host execution and the installed trusted CA
bundle; no TLS verification was disabled. There were no local reference copies.

## Base, source and physical target

Write X_s=(1-s)Z+sV, independent Z~N(0,I), with physical U=decode(V) and known
condition C. s runs0→1 and is not physical simulation time. The implemented CFM
learns v_s(x,C)=E[V-Z|X_s=x,C]. Its endpoint estimate is
m_s(x)=x+(1-s)v_s(x), the conditional mean in model coordinates, not a sampled
latent target or decoded conditional physical mean under nonlinear transforms.
At1 use m_1(x)=x with no model call. Heat uses affine u, Darcy affine(log a,u)
with positive decode/pressure gauge projection, chemistry affine sqrt species
with square decode. These maps and known-coefficient conditioning come from the
checkpoint. Measurements are not supplied to the trained network: they enter
only inference energy, so no observation likelihood is multiplied twice.

The intended physical tilt is pi(dU|C)=p_theta(dU|C) exp[-E(U;C)]/Z(C), with
Z=integral exp[-E(U;C)] p_theta(dU|C). Z is unknown. Known physical lambda and
inference penalty coefficients are distinct. This is a learned-prior energy
model, not an exact physical posterior or exact PDE-manifold conditioning.

For independent Gaussian observations with known positive standard deviation,
E_obs=.5 sum_measured((H(U)_i-y_i)/sigma_i)^2, omitting only constants independent
of U. No division by measurement count is implicit. For sigma=0, replace it by
the explicit positive numerical tolerance; this is a soft constraint, not a
Dirac likelihood. An explicitly selected tempered_mean convention instead uses
rho=max(sigma,tolerance) and divides by valid measurement count. It therefore
changes the target temperature and is labelled as such. An additional configured
observation weight also tempers the likelihood unless it equals1.

PDE penalties sum separate equation components .5 mean_quadrature((R_c/scale_c)^2)
with explicit scales/weight. Physical-time quadrature uses interior saved times
for transient residuals; Darcy is static and uses harmonic flux divergence.
Only periodic supported boundaries and unknown initial fields execute; there
is no invented boundary/initial target. Positivity/gauge are decode properties
and separately verified. No hidden reference, unknown initial state or fixed
conservation constant is supplied as evidence. Unimplemented boundary/initial/
invariant energy terms are rejected rather than silently added as zeros.

## Score and stochastic extension

Gaussian conditioning gives score_s(x)=-E[Z|X_s=x]/(1-s). Since
E[Z|x]=x-s v_s(x), score_s=(s v_s-x)/(1-s), for s<1. This is specific to the
independent unit-Gaussian source and the linear path, not a general velocity to
score conversion. For Gaussian V~N(mu,Sigma), q_s=(1-s)^2 I+s^2 Sigma,
v_s=mu+(s Sigma-(1-s)I)q_s^-1(x-s mu); substitution yields
score_s=-q_s^-1(x-s mu), verifying signs and time convention.

Choose state-independent sigma(s)=sqrt(2*kappa)*(1-s), kappa>=0. To preserve
p_s with an exact path-consistent velocity, the Ito drift must be
b_s=v_s+.5 sigma(s)^2 score_s, since its extra Fokker-Planck drift cancels the
diffusion term. Algebraically b_s=v_s+kappa*(1-s)*(s v_s-x). This expression
has no terminal division/clamp; at1 both added drift and noise vanish. The
continuous process is initialized from N(0,I) and reaches1; no truncated interval
or singular endpoint denoising is hidden. Assumptions include smooth positive
path densities, suitable growth/local Lipschitz conditions and independent
Gaussian source. State-dependent diffusion is unsupported.

Euler-Maruyama uses x_next=x+h*b_s(x)+sqrt(h)*sigma(s)*xi at left nodes,
xi~N(0,I) independently per descendant/problem/step. This is a genuine Ito SDE
stepper, not ODE Heun with noise. Fixed uniform/quadratic nodes are supported.
No gradient guidance drift is combined with FK proposals. For learned v_theta,
the plugged score need not equal the score of the actual learned ODE density;
continuous marginal equivalence is then an approximation, not a theorem.

## Exact finite-step path semantics

Let Q_h(dx_0:N)=N(0,I)(dx_0) product K_i(dx_i|x_i-1), where K_i is exactly the
specified Euler-Maruyama transition with the model/score-corrected drift above.
This defines a base terminal Q_h,N even for an imperfect neural velocity.
The algorithmic target is Q_h,N(dx) exp[-E(decode(x))]/Z_h. Its relationship to
the desired learned-flow tilt includes numerical SDE/ODE error, learned
flow/score consistency error and physical decode/model approximation.

Set phi_i(x)=beta_i E(decode(m_s_i(x))) with a fixed nondecreasing schedule,
beta_i=beta_0+(1-beta_0)*s_i, and phi_N=E(decode(x_N)) on the actual terminal
state. Include G_0=exp(-phi_0) and G_i=exp(-phi_i(x_i)+phi_i-1(x_i-1)). Their
product is exactly exp(-E(decode(x_N))). beta_0=0 makes G_0=1, but its factor
is not otherwise omitted. Endpoint-estimation error changes intermediate
variance/selection, not this telescoping identity. Finite intermediate energies
are required to retain support; NaN/infinite energies fail explicitly.

Importance sampling is the same base kernels with beta=0 before the terminal
and no intermediate resampling. An annealed no-resampling run telescopes to the
same final weights on identical paths. Repeated absolute exp(-E) factors are
not used. Initial/final potentials, carried previous phi and joint ancestry
must all be consistent when parents are duplicated.

Log weights start at-log(P), accumulate log G, and normalize by logsumexp per
problem. ESS=1/sum(w^2). Eligible resampling steps (fixed interval) and the ESS
fraction trigger are independent of beta. Systematic resampling uses one
uniform offset per problem and gathers every associated component/previous
potential/root ancestor; descendants are cloned and weights reset to1/P.
Future Gaussian increments are independently assigned by particle slot.
Terminal output is weighted unless explicit final systematic resampling is
requested; that procedure and the pre-resampling ESS are recorded. Finite-P
self-normalized summaries are biased/variable approximations, not exact draws.

All proposals are unchanged base K_i, so there is no unaccounted guidance
proposal ratio. Outer controller checks may log/stop for invalid numerics,
cancellation or budgets, but cannot alter beta/energy, reset weights or finish
mid-path as a terminal target. FK retries and FK-plus-refinement are rejected
in this slice. Legacy gradient controller/refinement continue separately.

## Particle axes, RNG and devices

State components have [problem B,particle P,...]; model fields then have packed
channel/physical-time and spatial axes. Resampling is independent within B.
Normalized weights/previous potentials use explicit CPU float64 for stable
MPS-compatible reductions. Model/physical energy runs on the selected device
and dtype; transfers and time are reported. Source/noise/resampling generators
are separate CPU streams per problem with stable SHA256 seeds. Noise is drawn
for the full population before chunked evaluation, preserving particle identity
and RNG assignment. This is an explicit CPU random-number policy, not a hidden
model fallback. Cross-backend/chunk-size bitwise equality is not promised for
neural kernels; same-backend repeatability and chunk statistics are verified.

Population snapshots retain states, weights, ancestry, RNG states, coordinate
and budgets for inspection/export. Inference resume is not implemented; training
resume is unrelated. Incomplete paths are labelled incomplete and are never
claimed terminal-target samples. FK cannot repair an inaccurate learned prior.

## Analytic reference targets

For V~N(mu,Sigma) and y=V+epsilon, epsilon~N(0,R), the stated sum likelihood
gives Sigma_star=(Sigma^-1+R^-1)^-1 and
mu_star=Sigma_star(Sigma^-1 mu+R^-1 y). These follow by completing the square
in the Gaussian density times exp[-.5(V-y)^T R^-1(V-y)]. The toy uses diagonal
Sigma=diag(1.44,.64), mu=(.7,-.4), y=(1.4,.3), R=diag(.25,.49).
Thus mu_star=(1.296449704,-.003539823),
Sigma_star=diag(.213017751,.277522124). The base-flow check compares unweighted
SDE moments to (mu,Sigma), whereas tilted methods compare to this posterior.

For the equal mixture of N((-2,0),.36I) and N((2,0),.36I), use y=(.5,.2),
R=diag(.9,.8). Each component has covariance C=(Sigma^-1+R^-1)^-1 and
mean m_k=C(Sigma^-1 mu_k+R^-1 y). Its posterior mixing mass is proportional
to prior_mass_k N(y;mu_k,Sigma+R). For general unequal covariances this includes
the Gaussian determinant factor; it cancels here because both are .36I.
The overall mean is sum a_k m_k, covariance C+sum a_k(m_k-m)(m_k-m)^T.
Positive-half-space occupancy is sum a_k Phi(m_k1 / sqrt(C_11)), not simply the
right component's mixture mass. This gives an independent analytic reference
without learned-model error, reference MCMC or an energy-based best-sample pick.
The implemented analytic mixture velocity uses conditional component
responsibilities at each s; it is not a neural model trained on toy samples.

## Failure policy and cadence

Normalization subtracts the largest log weight before logsumexp, including
equal energies near1e300. Finite concentrated weights are valid; NaN, positive
infinity, a row with no finite weight, nonfinite energy, or a nonfinite accumulated
log-normalizer raises InvalidPopulation. No invalid row is changed to uniform.
Successful traces report zero nonfinite counts; failures raise and the API marks
the job failed, rather than exporting a successful diagnostic with hidden zeros.
Every numerical step's ESS/energy/ancestry is retained. diagnostic_every controls
live notification cadence only; resampling events and the terminal step always
notify. This does not change kernels, potentials, RNG or stopping rules.
