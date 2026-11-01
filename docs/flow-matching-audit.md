# Audit of actual conditional flow matching

This audit follows executed package code, not an assumed label. The formulation
is `independent_gaussian_linear_cfm`; both legacy CNN and new periodic U-Net use
it. No new generative path or general score conversion was introduced.

## Training path

| Quantity/action | Actual implementation |
|---|---|
| Clean physical sample | data/physical.py specification/solve_spec and specialists/*.py sample_spec/solve_spec; providers.py PhysicalProvider.training_batch. Numerical reference solving occurs during training data production. |
| Frozen model coordinates | training.train creates/loads provider.transform once, fits only the training/calibration namespace, calls transform.encode then TimeChannelLayout.pack. TransformBundle is per-field mean/std over batch, physical time and space; Darcy additionally log(a), chemistry square-root concentration. Saved fitted IDs exclude validation/test. |
| Source Z | generative.LinearCFM.training_state draws torch.randn of exactly the packed target shape using the independent CPU source Generator, then explicitly transfers to the selected model device/dtype. |
| Generation coordinate s | Same function, independent CPU time Generator, torch.rand per problem/batch item, broadcast across all packed channels/space. The finite generator samples [0,1); the mathematical ideal is Uniform(0,1). |
| Path/target | X_s=(1-s)Z+sU_model and velocity target U_model-Z. U_model is the encoded clean field, not an unscaled physical concentration. |
| Conditioning C | core.Conditioning.from_problem plus conditioning.py known-scalar/forcing encoders. Heat: known D; Darcy: known forcing f; chemistry: known rates/diffusivities. No hidden initial/permeability arrays or clean evaluation samples; observations guide inference separately. |
| Network output | models.make_model → ResidualCNN or MultiscaleUNet, same packed shape as X_s. Generation s is embedded separately from fixed physical-time channels. |
| Objective/reduction | generative.LinearCFM.loss → functional.mse_loss with default mean, over every batch/channel/time-packed/spatial element. No observation-mask weighting. All current training fields are complete; no invented missing-value mask. Field scales come from the frozen transforms. |
| Update | training.train: finite loss check, zero_grad, loss.backward, clip_grad_norm_(10,error_if_nonfinite=True), Adam.step. Adam state, RNG streams, provider progress, transform and full resolved architecture/config saved atomically. |
| Held-outs | Fixed validation randomness and disjoint physical problems, independent source/time streams. Test is not used to choose training budgets/capacity. |

For each independently coupled pair, dX_s/ds=U_model-Z, with X_0=Z and X_1=U_model.
Minimizing squared error gives v*(x,s,C)=E[U_model-Z|X_s=x,s,C]. This is a
conditional expectation over pairs, not a claim that one realized target is
available at inference. Fixed physical time τ indexes trajectory channels;
generation s is a different coordinate. This path is defined in model coordinates;
its decoded physical path need not be linear (log-a, square-root concentration and
Darcy pressure gauge projection change it).

## Inference and distinct methods

`training.load_inference` checks formulation, specialist capabilities, exact grid/
domain/boundary/convention/times, fields, transforms and architecture metadata;
loads weights with CPU remapping then selects runtime; freezes parameters while
retaining gradients through inputs. `sampling._sample_once` draws independent
standard normal model states using its source seed. It constructs C from only
the inverse problem and integrates 0→1 using generation_schedule and make_stepper:

- Euler: x_next=x+h*v(x,s,C), one batched model evaluation.
- Heun: k1=v(x,s,C), k2=v(x+h*k1,s+h,C), x_next=x+h(k1+k2)/2,
  two evaluations. Uniform/quadratic schedules use their actual h.
- Decode uses the saved transform. Darcy exp/log-a and zero-mean pressure and
  chemistry squared affine square-root coordinates remain explicit parameterizations.
- Endpoint estimate is x+(1-s)v_theta(x,s,C); at s=1 it is x. It is generally an
  estimate for a learned conditional-mean velocity, not the paired training target.
- Observation/PDE gradient guidance applies the existing heuristic split correction
  after a full ODE proposal, through the endpoint model and physical decoder.
  Strength is multiplied by actual h. It changes the generated distribution;
  there is no exact-posterior claim.
- Fixed-step stochastic flow and FK remain separate: fk.simulate uses the existing
  score-corrected vanishing-noise SDE and telescoping potentials/resampling, with
  the numerical terminal-law tilt and assumptions in fk-mathematics.md. It is not
  the base deterministic flow, gradient guidance, or a numerical PDE reference solve.

No reference target is read by the inference worker. Automatic checkpoint selection
uses physical compatibility and registered IDs, never truth/error. An experiment's
reference context is stored separately, then attached after successful inference.
Display controls, reference attachment and metric choice cannot change sampler
particles, weights, controller actions or stopping.

## Actual snapshots and accounting

Optional InferenceConfig.snapshots selects bounded frames/particles/fields/times.
`sampling/snapshots.py` records actual model states, decoded current fields and
actual endpoint estimates. Display striding is explicit; native samples remain
unchanged. Default UI cap is9 frames,4 particles, selected saved fields/times;
collector total numerical arrays are bounded to16 MiB. No interpolation creates
fake states. ODE endpoint snapshots add passive model evaluations, excluded from
scientific controller budgets and reported separately as snapshot_model_evaluations.
Their time is inside measured sampling time; they do not consume RNG or mutate
states. Incomplete runs retain actual last position; an unavailable endpoint is
labelled unavailable.

FK reuses already evaluated velocities and records/composes actual parent maps
between captured frames. The display may lack a parent outside its four recorded
slots; it shows a gap and disables following that unavailable parent. It never
connects unrelated resampled particles with a smooth fabricated path. Sparse-parent
composition and passive snapshot invariance are independently tested.

## Mathematical and numerical verification

`tests/unit/test_flow_capacity_snapshots.py` covers 1D–3D tensor paths/endpoints,
finite-difference derivative U-Z, all-element loss reduction, real gradients and
parameter changes, native shapes, architecture metadata, exact analytic Gaussian
vector-field integration and Euler/Heun error order, snapshot/returned-state identity,
unchanged generated samples/weights/stopping and sparse FK parent composition.
Existing reference-isolation/controller/FK tests remain in the full suite.
`capacity.probe` executes the production objective and actual native forward/
backward/update, endpoint guidance and production Heun source-to-candidate sampler.
`training.train` is also exercised by real managed capacity runs and resume.

The analytic Gaussian vector field, independent of neural learning, is
v(x,s)=μ+[(sσ²-(1-s))/((1-s)²+s²σ²)](x-sμ).
Its solution is x_s=sμ+sqrt((1-s)²+s²σ²)z, so x_1=μ+σz. Tests isolate integration
order from network approximation. A separate MLP learns using the **production**
LinearCFM.training_state/loss and production Heun. With2,000 updates/batch256,
8,192 generated particles and64 Heun steps, target μ=1.25,σ=0.65, measured
mean1.30336/std0.66601 (absolute errors0.05336/0.01601). Target mean Monte Carlo
standard error is0.00718, so the mean discrepancy is not explained away as sampling
noise. This is finite-budget learned-flow evidence, not exact moments or PDE
accuracy. Seeds/budget/history/weights/samples: outputs/flow-workspace-20261026/
gaussian-toy/{toy.json,toy.pt}. A smaller deterministic learning test has a broad
predeclared0.2 moment-error bound; it is not used to tune the reported experiment.

Two matched production runs use identical physical/calibration IDs,32² layout,
20 actual updates and the same held-out validation case. CNN1,956 parameters:
fixed-validation CFM1.51396; U-Net132,436:1.53007. Observation+PDE validation
unobserved relative L2 is1.47368 versus1.55722; Fourier inverse is0.52520.
The larger model is not better in this bounded comparison. All results remain;
no test-selected winner. Detailed counts, memory/cost and artifacts are in
capacity-resolution.md and flow-workspace-verification.md.
