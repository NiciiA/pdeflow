# Variable-coefficient Darcy in 2D and 3D

Status: implemented periodic specialist, numerically verified and smoke-trained
in 2D/3D. See multispecialist.md for commands and verification.md for evidence.
The conventional restricted inverse benchmark and bounded online learning
study are now implemented (darcy-study.md). Robust scientific accuracy and
identifiability have not been established.

## Physical task and identifiability

Use a periodic Cartesian box with lengths recorded in DomainSpec. Solve
-div(k(x) grad p(x))=f(x), with mean(f)=0 and gauge integral(p)=0.
This periodic first slice makes the gauge and compatibility explicit; it is
not a generic no-flux boundary solver. A later homogeneous Dirichlet comparison
requires a separate verified boundary stencil and no pressure gauge.

Permeability k is scalar and positive. Represent log k as a smooth Gaussian
field, with a declared finite range for log k or positive k_min; do not silently
clip generated physical coefficients. Record physical/nondimensional units.
Forcing f is known and mean-zero. Generated problems use nonzero f; zero f is
accepted as a compatible degenerate case and gives no permeability information. Recover k and p jointly from sparse
pressure observations and explicitly specified permeability sensors where
available. Pressure alone generally does not identify k uniquely: include the
prior and any known mean/scale of permeability in the task definition and
report ambiguity. Never fill hidden permeability from the reference.

## Conservative reference and differentiable residual

Use cell-centred finite volumes. Store pressure and permeability at cells;
face transmissibilities use harmonic means of adjacent positive k values.
Flux in direction j is q_face=-k_face (p_right-p_left)/h_j. Its discrete
divergence is the opposing-face difference divided by h_j. The residual is
div(q)-f, consistent with -div(k grad p)=f. Periodic faces are shared so
internal fluxes cancel exactly in the volume sum.

The CPU reference applies the symmetric positive semidefinite stencil
matrix-free and solves in the mean-zero subspace using projected conjugate gradients,
with residual tolerance and iteration count reported. An independent tensor
flux evaluator supplies differentiable guidance. Avoid reusing the assembled
matrix to define all validation answers. Verify constant-k Fourier modes,
positive variable-k manufactured solutions with analytically derived f,
second-order grid convergence in 2D/3D, face-flux cancellation, source mean,
pressure gauge and solver-tolerance independence. Manufactured 8/16/32-grid checks establish second-order convergence before
training. Tiny 8-grid smoke data approximate the PDE coarsely; they do not
establish continuum accuracy at a scientific target tolerance. Reject incompatible
forcing rather than implicitly removing its mean.

## State, transforms and observations

Static unknown state is [B,C=2,T=1,*space] with named channels log_k and p in
model coordinates; physical outputs contain k and p with explicit units.
Use training-only affine transforms of log k and zero-mean p separately.
Decode k=exp(log_k); constrain positivity through this representation and
report any generated out-of-training-range values. Subtracting pressure mean
is a distribution-changing gauge projection: record its timing, cost and
magnitude. Do not apply the gauge to permeability. Future tensor-valued k
requires a positive-definite parameterization and is outside this slice.

Observation operators select named physical pressure or permeability values;
the low-level observation contract supports separate per-field noise/tolerance
scales. The current nondimensional UI/CLI presets declare one shared scale
for both channels; physical-unit applications must configure appropriate scales. Support existing on-grid
point selection first. Forcing and known coefficient-scale constraints belong
to Conditioning; measurements remain inference-time steering until an
observation-conditioned training comparison is explicitly implemented.

## Integration changes justified by Darcy

The spatial CNN and Euler/Heun schedule/steering interfaces can be reused.
The heat-specific fixed scalar layout must become dataset-declared channel
metadata, including static T=1. Residual evaluators must declare their physical
time requirements: Darcy cannot invoke the heat evaluator requiring >=3
frames. Config/data/model factories must declare the new family and supported
operators. Add a spatial forcing encoder rather than mislabelling forcing as
a scalar diffusivity. TransformBundle needs a named per-field positive decode
for log permeability. Checkpoints retain both channel transforms, forcing
schema, boundary/gauge/flux conventions and exact grid. The browser now routes executable Darcy requests to compatible two-field
checkpoints and exposes physical a/u slices with no physical-time controls.

Joint CFM training generates the complete paired unknown state. Split complete
(k,f,p) problems before masks/statistics; source noise is independent for the
paired state. Guidance combines pressure/permeability observation potentials,
conservative flux residual, known source and gauge only. It must not use the
true k or p. No claim of identifiability or posterior calibration follows from
small residuals.

## Conventional baseline and acceptance

Optimize a finite Fourier parameterization of log k with a smoothness prior,
solving pressure numerically at each candidate. Select regularization on
validation problems, report forward/adjoint/linear-solve counts and wall time.
Compare joint log-k/k and pressure errors on unobserved values, flux residual,
source compatibility, gauge, positivity, ensemble spread and observation fit.
Match forcing/observations/splits and disclose prior truncation differences.

Executed integration acceptance: manufactured/convergence/conservation tests in genuine 2D/3D;
reference-free regression; train/save/resume/portable export;
bounded held-out reconstruction; API compatibility and browser slice tests.
The earlier user permitted the inverse benchmark to remain unfinished; the
subsequent task implemented the restricted baseline in darcy/inverse.py.
Implementation/smoke training are separate from scientific evaluation; no
oracle forward solve is used as a conventional inverse baseline. A subsequent complete chemistry
slice uses A+B reversible C with positivity and a+c, b+c invariants; Darcy
must not displace the project's chemistry/steering research purpose.

## Concrete generator and representation

The generator draws a stationary smooth Gaussian Fourier field g from white
noise, using integer Fourier-mode filter exp(-|m|²/(2 bandwidth²)) and fixed
expected-variance scaling sqrt(N/sum(filter²)). No per-instance normalization
uses target statistics. Its random zero mode is retained, so realized spatial
means and RMS values vary. Correlation lengths scale with the recorded box
lengths and bandwidth; bandwidth is in integer modes. It uses
a=exp(offset+amplitude*tanh(g)).
The supplied Darcy profiles have amplitude uniform [0.2,0.6], offset uniform
[-0.2,0.2], bandwidth=1, hence exp(-0.8)<a<exp(0.8) (approximately 0.449–2.226).
This bounded transformed smooth Gaussian construction is not a claim of an
exact Gaussian log-permeability law or a fitted material correlation model.
The forcing is a separate independent smooth field with its generated mean
removed explicitly to construct compatibility; supplied user forcing is never
silently projected. Forcing remains float64 on disk to retain compatibility;
model conditions may be float32. Per-instance CG diagnostics are retained in
shard sidecars. Every complete (a,f,u) instance is split before statistics.

The two affine statistics use training log(a) and pressure separately. exp
ensures positive decoded permeability without clipping; numerical overflow is
an error, not silently clamped. Pressure mean is subtracted on every decode,
including guidance. It changes sample distributions and enforces the gauge
only. Diagnostics record removed-mean RMS and final decode/projection time;
guidance decode costs are included in total sampling time. No projection
enforces the Darcy PDE itself.
