# PDEFlow: project specification

Status: proposed research and implementation specification, not a statement of completed capabilities.

## 1. Goal

Build an extensible research framework for generative PDE reconstruction in one, two, and three spatial dimensions. Given a physical problem and incomplete measurements, generate plausible reconstructions of unknown fields and, where specified, unknown parameters. Compare diffusion and flow matching, observation and physics guidance, and Feynman–Kac (FK) steering.

Use specialist models behind common interfaces. Different families, dimensions, representations, and regimes may use different checkpoints. A shared conditional model is an experiment, not a prerequisite. Do not advertise universal PDE solving.

The contribution is a reproducible reconstruction and steering framework, validated across physical behaviours, with chemical reaction–diffusion as a substantive application. A report must distinguish implemented functionality, numerically verified functionality, trained capability, and scientifically evaluated performance.

## 2. Problem definition

Represent each task by C = (E, lambda_known, Omega, B_known, I_known, y, H, noise, unknowns, discretization).

- E: equation or coupled system, represented by a family identifier plus structured operators where supported.
- lambda_known: known constants, forcing, and coefficient fields.
- Omega: spatial dimension, domain extent, geometry, and coordinate conventions.
- B_known / I_known: known boundary / initial information, including masks when incomplete.
- y: measurement values; H: operator mapping physical fields to predicted measurements.
- noise: observation likelihood assumptions, distinct from source noise and physical stochasticity.
- unknowns: exactly which fields, parameters, and missing conditions are inferred.
- discretization: coordinates, cell/node conventions, physical times, boundary labels, and validity masks.

A quantity cannot be both fully known and a reconstruction target. Unknown physical parameters must be carried by the generated state or another explicit inference mechanism, not silently filled from reference answers.

The output is an ensemble of candidate reconstructions with provenance and diagnostics. It is not automatically an exact or calibrated posterior. When claiming a posterior target, state the prior, likelihood, physics model, weighting scheme, and approximations.

## 3. Supported scientific tasks

1. Reconstruct full fields from sparse observations of the solution.
2. Jointly reconstruct input/coefficient fields and solutions when trained and identifiable enough for the stated evaluation.
3. Infer missing initial conditions and trajectories.
4. Forward prediction with fully supplied inputs as a comparison.
5. Compare separate specialists and shared conditional models on specified operator families.

Partial observation is central. A numerical forward solver with fully specified inputs is not equivalent to the reconstruction task.

## 4. Architecture scope and benchmark targets

The design must accommodate regular-grid fields in d=1,2,3, scalar/vector/multispecies states, static and dynamic problems, and explicit capability selection. Three spatial dimensions must not be confused with two spatial dimensions plus physical time.

| Family | Physical role | Intended dimensional coverage | Distinct requirements |
|---|---|---|---|
| Elliptic: Poisson, heterogeneous Darcy pressure | Equilibrium and coefficient inference | 1D, 2D, 3D | Positive coefficient fields, boundary handling, pressure gauge where relevant |
| Advection/diffusion and coupled reaction systems | Chemical movement, mixing, conversion | 1D, 2D, 3D | Conservative transport, species channels, reaction invariants, positivity |
| Waves | Propagation | 1D, 2D, 3D | Initial displacement AND velocity, appropriate energy checks |
| Viscous Burgers | Nonlinear transport benchmark | 1D initial benchmark; other formulations explicit | Conservative nonlinear term, viscosity and steep gradients |
| Incompressible flow | Fluid evolution and pressure constraints | 2D, 3D | Dedicated velocity/vorticity representations, incompressibility, geometry |
| Fokker–Planck | Probability evolution for stochastic dynamics | Selected 1D–3D state spaces | Probability normalization, nonnegativity, specified drift and noise |

This table is a target matrix. A row is not implemented merely because a class or configuration exists. Start with box domains and periodic / homogeneous Dirichlet boundaries, with precise per-family declarations. Irregular meshes, local adaptive meshes, open boundaries, and new equation classes remain explicit additions.

Schrödinger and other quantum modules are outside the current required implementation. Do not add them just because the general interface can describe complex fields.

## 5. Initial substantial implementation

Establish a real dimension-independent numerical foundation, then one working generative reconstruction pipeline:

- Verified periodic heat/diffusion data and manufactured Poisson checks in 1D–3D, including a true 3D case.
- A real conditional flow model for complete diffusion trajectories, including 2D and small true 3D smoke configurations.
- Use a spatial backbone with explicit temporal handling (a documented initial option is stacking fixed physical-time frames as channels for 1D/2D/3D spatial CNNs). This is fixed-time-window support, not arbitrary temporal resolution generalization.
- Sparse observation generation, observation guidance with gradients through an endpoint estimate, physical residual diagnostics, and checkpoint reload.
- Euler and Heun ODE sampling with interchangeable generation schedules.
- CPU smoke execution, an accelerator-oriented profile, and a thin Colab launcher.

Heat is an integration/verification anchor, not the final scientific scope. Retain elliptic, chemistry, waves, and fluids in the capability roadmap. Do not let the entire project collapse to a heat-only example. Subsequent complete slices add variable-coefficient Darcy and A+B <-> C chemistry in 2D/3D, diffusion baselines, and FK steering.

## 6. Physics and chemistry

Chemical concentrations satisfy, for example,

dc_i/dtau = -div(q c_i) + div(D_i grad(c_i)) + R_i(c; eta).

For A+B <-> C, r = k_f a b - k_r c and R=(-r,-r,+r), under mass-action assumptions. For no-flux/periodic boundaries, integrals of a+c and b+c are conserved; the unweighted sum a+b+c is generally NOT conserved. Physical mass conservation requires the correct molar-mass weights. Boundary fluxes must be included for open systems.

Fokker–Planck densities describe stochastic physical trajectories; the generative model distribution over entire fields is a different level of probability. Likewise the sampler's artificial SDE is not the physical stochastic process.

## 7. Training and inference methods

Training components: dataset, transforms, condition encoder, backbone, probability/noise path, target construction, objective, optimizer, validation policy, checkpointing.

Inference components: model adapter, sampler dynamics, numerical stepper, generation schedule, optional adaptive integration controller, endpoint estimator, observation/physics evaluation, steering, particle management, stopping and output policy.

Keep these independently configurable where mathematically compatible. A discrete diffusion process is not interchangeable with a continuous ODE by merely renaming its stepper. Score/velocity conversions must be derived for the chosen path and endpoint conventions.

Baseline flow path: X_s=(1-s)Z+sU, Z~N(0,I), s~Uniform(0,1), target U-Z. The network approximates the conditional expectation of this target given its inputs, not a unique hidden training endpoint at inference.

Learned conditioning, classifier-free guidance (CFG), gradient guidance, and FK steering are distinct. Use known physical parameters as initial network conditions, and measurements through inference steering. Observation-conditioned training/CFG is a separate comparison. Avoid counting the same measurement likelihood twice without defining the resulting target.

## 8. Refinement

Separate generation integration adaptivity from physical space/time refinement. Finer sampler steps do not recover unresolved physical frequencies. First support a hierarchy of global regular grids and measured convergence. Local refinement requires mesh/model support, valid prolongation/restriction, boundary treatment, and missing-detail generation/correction.

Interpolation alone is not demonstrated resolution recovery. Record whether refinement uses a learned model, a numerical solver, or both. A hybrid correction must not be credited entirely to the generator.

## 9. Runtime requirements

- One installable Python package, Python environment and dependency requirements explicitly documented and reproducibly pinned for tested workflows.
- PyTorch as the initial neural/tensor runtime; NumPy/SciPy may be used by CPU reference solvers. Select dependency versions after checking compatibility in the implementation environment.
- CPU must execute small end-to-end verification cases. CUDA is the main Colab training backend; MPS is optional acceleration on supported Macs. Intel Macs use CPU unless another supported backend is present.
- No unconditional .cuda(), CUDA-only required extensions, hidden device fallback, or implicit downloading of large datasets/weights.
- Device, dtype, precision, paths, model size, particle count and resolution are configuration settings.
- Colab notebook installs and calls the same package. Runtime-local cache is disposable; checkpoints/results are copied to a configured persistent location.
- MacBook loads exported checkpoints without the training dataset or CUDA and can run inference, evaluation, and small training. Large 3D workflows get smaller explicit local profiles rather than a false promise of identical resource requirements.
- Resume training must restore appropriate optimizer/scheduler/scaler/RNG states. Cross-backend reproducibility is tolerance-based, not bitwise guaranteed.

## 10. Research deliverables and claims

Deliver package, configs, verified numerical generators or licensed dataset adapters, Colab/local workflows, checkpoint metadata, reproducible experiments, figures/tables, and a report.

Report training and sampling costs separately; include reference-data generation cost. Measure unobserved reconstruction error, observation fit, independently evaluated physical violations, conservation, sample diversity/coverage, runtime/memory, and FK particle diagnostics. Compare to an appropriate conventional inverse baseline.

Do not infer scientific success from a smoke test, low training loss, low residual alone, or attractive pictures. Tests with exact or manufactured solutions establish numerical correctness; held-out reconstructions establish empirical model performance.

## 11. References and interpretation

- DiffusionPDE: Generative PDE-Solving Under Partial Observation (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/eb3878c1dcbfff9ee95d5d033e5f5942-Paper-Conference.pdf
- Feynman-Kac-Flow: Inference Steering of Conditional Flow Matching to an Energy-Tilted Posterior, arXiv:2509.01543: https://arxiv.org/abs/2509.01543
- Associated code: https://github.com/heid-lab/fkflow
- MIT flow/diffusion course: https://diffusion.csail.mit.edu/2026/index.html
- User-provided AMLD2026 poster and PDE classification notes: optional local reference inputs, not project specifications.

Record exact paper and code revisions when reproducing methods. The molecular transition-state application generates atomic geometries, not chemical concentration PDE solutions; PDE residual steering is our adaptation. Do not copy code without preserving its license and attribution.

## Current implemented slices (2026-10-22)

Heat, variable-coefficient Darcy and A+B reversible C chemistry now execute
through common offline/online/hybrid training and equation-driven routing.
This fulfills the first three physical slices, not the broader universal/PDE
roadmap. Online sampling still requires clean numerical reference solutions
and fixed evaluation. Numerical tests/smoke training are distinct from scientific
accuracy. The restricted Darcy inverse and bounded negative learning study are
documented in darcy-study.md; chemistry and provider assumptions are in
chemistry.md and online-training.md. Repeated-seed scientific evaluation is next.

Current process continuation (25 October 2026): the reference-free bounded
feedback controller now connects diagnostics to configured gradient steering,
legal boundary retries, final verification and explicit stop reasons. A real
nested heat spatial refinement route reselects a fine checkpoint and preserves
actual measurements. This historical process slice preceded the now implemented
fixed-step FK resampling recorded in fk-verification.md. Generic/temporal/local
refinement remains unimplemented. process-audit.md maps the full diagram, including absent branches;
process-verification.md records execution and scientific limits. The broader
chemistry/steering research purpose and 1D–3D architecture remain unchanged.


27 October2026 optional representation continuation: a scalar periodic heat
set-encoder/frozen-AE latent-CFM/coordinate-decoder branch now operates beside the
fixed-grid specialists. It does not replace the broader chemistry/steering scope
or introduce universal equation/domain/dimension transfer. Exact capability and
scientific limits: coordinate-specialist.md and coordinate-verification.md.

The integrated training continuation adds a shared UI/CLI orchestration layer:
choose problem/model/budget/device, train, inspect validation and explicitly use
the resulting artifact. It reuses both existing trainers, preserves their
scientific boundaries and does not claim improved model accuracy. The ordinary
workflow no longer requires JSON or manual AE/flow path coordination. See
training-workflow.md and training-workflow-verification.md.

### Continuous-time coordinate heat slice (28 October2026)

The optional heat2D coordinate branch now declares continuous physical support
[0,1], direct Fourier-reference values at sampled coordinates, fixed per-solution
reference encodings and fresh weighted-time supervision/observations. This does
not change the original fixed-grid stacked-time contracts or broad1D–3D purpose.
Display frames and generation integration are separate. Reference-assisted
measurement design is disclosed, with no reference field entering inference.
Execution flexibility and scientifically demonstrated accuracy remain separate.
