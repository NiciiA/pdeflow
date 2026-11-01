# Executable periodic A+B reversible C specialist

The third specialist reconstructs unknown concentration trajectories a,b,c in
2D and true3D. It solves a_tau=D_A Delta a-r, b_tau=D_B Delta b-r,
c_tau=D_C Delta c+r, r=k_f a b-k_r c. Only periodic Cartesian nodal boxes are
implemented. Initial information is unknown unless supplied as explicit point
observations. D_A/B/C are known positive constants; k_f,k_r are known nonnegative
constants. No kinetic-parameter inference, no-flux wall, advection or generic
chemical mechanism is implied.

## Reference and numerical assumptions

The CPU reference uses Strang splitting: half diffusion, exact local reaction,
half diffusion. Diffusion is the periodic second-order FD Laplacian semigroup,
not a renamed heat trajectory. For each axis, L=(2D/h²)(P-I), where
P v=(roll(v,+1)+roll(v,-1))/2. Thus exp(dt L)=exp(-lambda) sum_k lambda^k/k! P^k,
lambda=2D dt/h². A nonnegative periodic convolution kernel is built from this
Poisson series. Substeps keep lambda<=16; the omitted tail is <=1e-14 and the
kernel is renormalized by its remaining mass. This approximation is recorded,
positive, and conservative to floating arithmetic; no exact machine-conservation
claim is made. Spatial order is two. Maximum physical step controls splitting
error; there is no adaptive global-error guarantee. The diagnostics' converged
flag means completed bounded physical steps and converged diffusion series.

For reaction, local A=a+c and B=b+c remain fixed. Solve
c'=k_f(A-c)(B-c)-k_r c analytically as a Riccati equation, including k_f=0 and
the repeated-root irreversible limit. Then a=A-c and b=B-c. No unstable explicit
reaction update or concentration clipping is used. Negative arithmetic results
are errors, not silently repaired. Physical step count is bounded by the
configured solver_max_iterations; failures enter the common bounded attempt
policy. The complete solver/step configuration is retained in each sample.

For closed periodic boxes, spatial integrals of a+c and b+c are conserved.
The unweighted sum a+b+c generally changes. Physical mass conservation requires
appropriate species weights and units; concentrations here are nondimensional.
No independent per-species invariant projection is applied during generation.
The generic exported mass_max_drift metric is retained for compatibility and
means change in individual species integrals for chemistry. It is not a chemical
conservation error. Use invariant_a_plus_c_max_drift and
invariant_b_plus_c_max_drift for the actual invariants. The UI explicitly labels total species
sum change and does not present that sum as conserved.

## Distribution, coordinates and learned representation

Initial species are independent fields .05+.5 sigmoid(F), where F is the
existing stationary smooth Gaussian Fourier field with integer bandwidth1.2
and fixed expected-variance scaling. The positive range is (.05,.55); this is
not a per-instance normalization. Known diffusivities are independently uniform
[.01,.04], k_f uniform[.3,1], k_r uniform[.1,.5]. Zero-rate limits are numerically
verified, but their trained generalization is not established.

Physical layout is [problem,C=3,T,*space] in (a,b,c) order. Fixed physical times
and spatial axes are explicit; generation coordinate s is separate. The common
provider supports offline, fresh online and bounded hybrid replay. A training-only
calibration stream fits affine statistics of sqrt(concentration), then freezes
them. Decode squares the affine root coordinate. Zero encodes/decodes exactly;
there is no physical concentration floor. The affine standard-deviation floor
is1e-8 for constant channels. Squaring folds negative root branches, and its
zero derivative at a zero root can weaken near-zero guidance. Diagnostics report
the negative-root fraction. This positivity parameterization changes model
geometry and does not enforce reaction invariants or yield an exact posterior.

Conditioning contains only log(D_A/B/C), log1p(k_f/k_r). Initial/reference species
are never known model conditions. Observations select physical a,b,c at arbitrary
saved grid/time indices, including one species only. Gaussian measurement noise
can yield negative observed values; those are measurements, not negative clean
reference concentrations. Residual guidance differentiates through the square
decode and endpoint estimate in physical concentration units. It uses independent
central-time/FD-space residuals; spectral-space diagnostics are also available.

## Symbolic recognition and workbench

The existing editable A+B reversible C preset now routes to chemistry. The
allowlisted four-row system and explicit r definition are recognized from the
current parsed operators, independent of preset label and row order. Species
rows can also use lhs-(rhs)=0. The rate definition must remain r=k_f a b-k_r c;
adding a term or changing its kinetic law invalidates execution. Fixed names
(a,b,c,D_A,D_B,D_C,k_f,k_r,r) are required; consistent renaming and arbitrary
algebraic equivalence are not implemented. Roles, scalar ranks, units, known
parameters, boundary, initial information and exact checkpoint layout are checked.

Use the existing physical-field selector for species, physical-time slider,
observation overlays and orthogonal true3D slices. Reference-free results expose
residual/positivity/invariant diagnostics, not invented reconstruction errors.
Attach truth only after inference for per-species errors. One sparsely observed
species does not guarantee unique recovery of hidden species. Empirical spread
is not calibrated uncertainty. The UI exports offline/online/hybrid training
configuration and CLI commands; it does not launch training.

## Actual execution and limits

The 2D8² online smoke performed8 optimizer updates; true3D8³ performed4, both
with four physical frames, width8/one block/batch2, save/resume/export and all
three guidance modes. Outputs are outputs/online-chemistry-20261021/chemistry-2d
and chemistry-3d. Clean generated references showed maximum a+c/b+c integral
drift1.35e-14 and2.03e-14 respectively, with zero clipping. These tiny models
are undertrained; their concentration errors and invariant violations must be
reported, not confused with the reference solver's conservation. Saved MPS
preflights verify two updates/resume/guidance/CPU reload at16² and true8³.
See verification.md for final tests/browser outcomes. CUDA/actual Colab remain
untested. Larger user-launched profiles are in online-training.md.

The next scientific task is repeated training/sampling seeds, more independent
held-out chemistry problems, controlled observation designs and invariant-aware
reconstruction comparisons at matched cost. Do not add an unverified invariant
projection and credit its benefit to the neural generator.
