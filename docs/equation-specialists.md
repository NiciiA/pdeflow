# Presets and the next executable specialists

A parsed template is not a solver, dataset or trained model. The symbolic AST
is separate from the tensor residual evaluator and from checkpoint conditioning.
Narrowly recognized periodic heat, static Darcy and chemical reaction–diffusion requests now cross
that execution boundary; multispecialist.md documents the exact patterns. Additional heat instances and broader initial-condition
coverage do not add an equation family. The following is an implementation map,
not a claim that its planned execution already exists.

| Preset | Current status | Reference solver and physical residual | Parameter / initial sampler and boundaries | Verification required | State and training configuration |
|---|---|---|---|---|---|
| Heat / diffusion | Editable, parsed, executable with compatible checkpoint | Existing exact FFT evolution; independent tensor FD time/Laplacian residual | D>0 uniform; random Fourier bandwidth/amplitude/offset. Periodic Cartesian boxes | Existing analytic decay, mean conservation, 1D–3D convergence and guidance gradients; larger-profile quality still unmeasured | Scalar signed u over fixed saved times; D conditioning. Existing smoke and new mps_* heat configs |
| Advection–diffusion | Editable/parsed; execution planned | Start exact periodic Fourier evolution for constant velocity, then conservative finite-volume variable transport; residual u_τ+v·∇u-DΔu | Positive D and bounded constant vector v first; scalar Fourier initial state; periodic boundaries. Variable v requires explicit compressible versus incompressible transport choice | Analytic phase/decay, mass/flux, transport convergence and vector conditioning gradients | u[C=1,T], separate known velocity vector/map; new condition features, generator, residual and dedicated config/checkpoint |
| Poisson | Editable/parsed; standalone periodic numerics already verified; no symbolic execution/neural adapter | Existing spectral -Δu=f with compatible mean-zero source and zero-mean gauge; independent residual discretization | Smooth compatible source; no initial state; periodic box | Existing manufactured/gauge/source checks; new sparse inverse identifiability/observation validation | Static scalar u[T=1], known f field; static model/condition adapter and new training config required |
| Darcy | Executable and smoke-trained periodic 2D/3D; restricted inverse baseline implemented | Conservative finite-volume -div(a grad u)=f; face flux residual, independent solver/convergence | Positive permeability via log fields; compatible forcing; periodic gauge or explicit Dirichlet/flux problem. No initial state | Flux conservation, gauge/compatibility, coefficient positivity, 2D/3D manufactured convergence and inverse baseline | Joint unknown log(a), pressure u; known source/boundaries as separate conditions; separate transforms/channels and dedicated 2D/3D configs |
| Viscous Burgers | Editable/parsed; execution planned | Conservative 1D finite-volume/IMEX reference for u_τ+∂x(u²/2)=νu_xx; independent residual | ν>0, bounded smooth Fourier or localized periodic initial velocity; periodic domain first | Conservation, viscous travelling/analytic cases, shock-layer resolution/convergence, guidance gradients | Scalar velocity u[T,x]; explicit viscosity condition and dedicated data/training config; current heat checkpoint is incompatible |
| Wave | Editable/parsed; execution planned | Spectral space plus verified second-order/symplectic time integration for u_ττ=c_w²Δu | c_w>0; sample both initial displacement and velocity independently; periodic boundaries first | Dispersion, energy behavior, propagation/convergence and complete initial-data treatment | Displacement and initial velocity roles; state may include velocity for a first-order representation. New time adapter/residual/condition/training config |
| A+B ↔ C | Executable periodic2D/3D; actual CPU/MPS smoke | Verified positivity-aware reaction integrator and conservative diffusion splitting; residuals a_τ-D_AΔa+r, b_τ-D_BΔb+r, c_τ-D_CΔc-r | Nonnegative species initial fields; positive D_A/B/C and k_f/k_r; closed periodic/no-flux boundaries implemented explicitly | Positivity, reaction stoichiometry, equilibrium, convergence and spatial integrals of a+c and b+c. The sum a+b+c is generally not conserved | Three species channels; distinct positivity/scale transforms; derived r is not an independent predicted species; dedicated multispecies generator/config/checkpoints |
| Fokker–Planck | Editable/parsed; execution planned | Conservative finite-volume density evolution ρ_τ=-div(bρ)+DΔρ; flux residual | D>0; declared physical drift/potential; normalized nonnegative initial mixtures; periodic or explicitly no-flux domain | Density mass/positivity, equilibrium and flux, convergence, separation of physical stochastic process from generative noise | Density channel with normalized/positive representation, known drift field; new family residual/conditioning/training config |
| Incompressible Navier–Stokes | Editable/parsed vectors and constraint; execution planned | Verified periodic spectral projection or conservative velocity-pressure reference; momentum and div(v)=0 residuals | ν>0; divergence-free initial velocity, compatible body force; pressure gauge, periodic geometry first | Divergence, pressure compatibility, forcing/energy balances, dealiased convergence in genuine 2D then 3D | Named velocity components plus pressure/gauge treatment; vector channels/conditions, proper pressure transform and dedicated fluid configs |

All planned families need an actual generator, physical-unit residual adapter,
capability declaration, observation mapping and trained checkpoint metadata
before execution is enabled. Static, vector or coupled parsing alone does not
establish any of these. Boundary masks do not implement their fluxes. Avoid
adding heat data under another family label. No planned training configuration
is supplied as a runnable fake implementation.

Next scientific task: repeated training/sampling seeds and independent held-out
problems for Darcy and chemistry, controlled observation designs, baseline
stopping and finer-grid physical diagnostics. The broader diffusion/FK/waves/
fluids roadmap remains explicit future work. No extra family was added as a
placeholder. See online-training.md, chemistry.md and darcy-study.md.
