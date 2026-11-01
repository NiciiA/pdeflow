> Historical implementation record. The active heat/wave physical-state v2 workflow is documented in [operator-wave-model.md](operator-wave-model.md) and the current README. Previous whole-trajectory checkpoints are preserved and incompatible.

# Physics-Conditioned Joint Latent Flow — implementation contract

This is PDEFlow's implementation of the supplied design, not a reproduction of
DiffusionPDE or FlowPDE. The PDF is design guidance; the user's implementation
request governs scope. The first runnable schema is periodic scalar heat2D,
continuous physical time [0,1], with either known or unknown constant diffusivity.
The broader 1D–3D chemistry/steering research scope remains in project.md.

## Contract established before implementation

B = solutions, N = reference samples, O = observations, Q = requested points,
L = total latent slots, W = channels. Physical coordinates are (x,y,t), in domain
units; generative time s is an independent scalar in [0,1]. Space is periodic,
upper endpoint excluded in training. Missing time is represented by a flag,
not t=0. Static fields are sampled once, not independently at each time.

Complete representation targets contain solution u(x,y,t), initial u0(x,y), and
positive scalar D. Encoder coordinates [B,N,3], normalized values [B,N] and
log-range-transformed D [B] are training-only. A deterministic train-recipe seed
fixes reference coordinates per realization. Initial aggregation uses explicit
t=0 samples. These complete targets never enter inference conditioning.

Fixed slots: solution, initial, diffusivity. Each field has global and regular
periodic local anchors; the scalar has global slots only. Slots have quantity,
anchor and global/local embeddings. A shared self-attention processor couples
all groups. Tokens [B,L,W] do not specify an output grid. Forcing is absent;
periodic boundary/domain/equation descriptors are conditioning only. Unimplemented
physical schemas are rejected by the specialist, not silently encoded as heat.

Statuses are known, partial, unknown, absent. Padding has a separate boolean.
Known zero has a valid known value. Solution is partial with measurements,
otherwise unknown; initial is unknown. D disclosure is randomized per solution
using a recorded probability. Known D slots are zeroed/excluded in generation
and replaced by typed known-context tokens in decoding; the scalar head is
bypassed. Supplied D is authoritative. Unknown D is absent from context, even
though it was available to generate training references. Its scalar head is
bounded by the declared positive training prior. No full initial field is
required at inference. Noisy observations condition the generator; no hard clamp.

Observation tuples: coordinates [B,O,3], features [B,O,3] (normalized u,
normalized absolute sigma, valid type flag), valid mask [B,O]. Quantity ID and
physical-time-present/type flags are embedded explicitly. Scalar u is the current
runnable observation schema. Learned aggregation queries and a learned null token
produce bounded set conditioning. No sequence position encoding. Zero noise is
encoded separately from log(sigma+epsilon). Conditioning is cached per ODE solve.
Typed context modules accept scalar, vector, sampled field/initial/boundary inputs;
only constant scalar D, periodic boundary, box domain and heat operator are
advertised by this specialist.

Decoder queries [Q,3] or [B,Q,3] plus explicit quantity name produce [B,Q]
normalized signed fields. A separate scalar head produces [B] D. Global tokens
and deterministic nearest periodic anchors are attended separately. Selection is
independent of chunking/order and output resolution. Smooth math attention permits
second coordinate derivatives. One generated latent defines the entire trajectory.

Stage A jointly reconstructs u, u0 and D, plus mild configurable latent L2.
Field objectives remain quantity-specific: the configured reference-only
region-balanced objective applies to u; initial-field supervision uses the same
per-time reference segmentation with t fixed at zero. Scalar loss is squared
error in the recorded log-range coordinate, never spatial region weighting.
New UI/Colab runs select RB alpha=.5; old empty loss configs still mean MSE.
Physical normalization is one train-only global transform, not per-sample scaling.

After A, freeze encoder/decoder and train-only per-group/channel latent mean/std
(floor .01). Targets are fixed for each complete realization, independent of
conditioning mask. Cache identity includes dataset, frozen encoder weights,
sampling policy, normalization and schema/anchors. Stage B samples Gaussian noise
and straight CFM paths. Mask known slots at source/target/velocity. Average squared
error over channels/tokens within each active group, then weighted active groups,
then solutions. More field slots do not give a quantity greater total weight.
The flow velocity loss has no spatial masks. Flow time uses FiLM in every block.

Optional decoded auxiliary training uses z_s+(1-s)v, a one-step estimate, not an
exact ODE endpoint. It is explicitly enabled; decoder remains frozen. Optional
bounded latent correction changes the generated distribution and uses observations
and declared physics only. Neither is an exact posterior procedure.

Resolution flexibility and tiny execution checks do not establish scientific
accuracy or parameter identifiability. Large presets are capacities, not hardware
fit promises. Existing weights are legacy and remain available for storage deletion;
no semantic state-dict conversion is attempted.

## Capacities and exact parameter counts

| Preset | Encoder | Decoder | Observations | Physics context | Velocity | All stages | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | 10,320 | 14,475 | 8,520 | 7,080 | 10,008 | 50,403 | 40,083 |
| xsmall | 1,542,144 | 1,533,699 | 866,048 | 669,440 | 5,663,744 | 10,275,075 | 8,732,931 |
| small | 4,651,776 | 3,431,043 | 1,941,120 | 1,495,680 | 16,862,976 | 28,382,595 | 23,730,819 |
| medium | 10,401,792 | 6,608,387 | 3,444,224 | 2,649,600 | 44,665,856 | 67,769,859 | 57,368,067 |
| large | 23,368,704 | 14,827,779 | 7,734,528 | 5,940,480 | 133,498,368 | 185,369,859 | 162,001,155 |
| xlarge | 58,310,656 | 28,425,219 | 13,736,960 | 10,542,080 | 354,805,760 | 465,820,675 | 407,510,019 |

Counts are exact module numel on the meta device; All stages includes frozen
parameters, not simultaneously trainable parameters. Decoder includes its typed
context path. Velocity excludes the separately listed observation/context encoders.
No large model was allocated with optimizer state. No GPU-fit, accuracy or scaling
claim follows. Custom settings have no width/depth/token capacity cap; positive
sizes, group allocation and head divisibility are enforced.

| Preset | W / velocity depth / heads | Total slots | Solution global + local | Initial global + local | Scalar | Encoder / decoder depth | Observation slots |
|---|---|---:|---|---|---:|---|---:|
| Smoke | 24 / 1 / 4 | 8 | 2 + 1 | 2 + 1 | 2 | 1 / 1 | 4 |
| XSmall | 256 / 6 / 8 | 64 | 5 + 25 | 5 + 25 | 4 | 2 / 2 | 16 |
| Small | 384 / 8 / 8 | 128 | 13 + 49 | 13 + 49 | 4 | 3 / 2 | 24 |
| Medium | 512 / 12 / 8 | 256 | 3 + 121 | 3 + 121 | 8 | 4 / 3 | 32 |
| Large | 768 / 16 / 12 | 384 | 19 + 169 | 19 + 169 | 8 | 4 / 3 | 48 |
| XLarge | 1024 / 24 / 16 | 512 | 27 + 225 | 27 + 225 | 8 | 6 / 4 | 64 |

Observation/context depths default1 independently of velocity depth. Local fields
use square anchor lattices in normalized periodic domain coordinates. Four nearby
anchors across the field groups are selected with stable distance ordering;
ties use slot order. This piecewise neighbour selection is differentiable within
a fixed selection, not across neighbour-switch boundaries. Global tokens provide
a separate path. Encoder local aggregation uses a periodic distance kernel;
shared processing couples all quantities. The width/depth ladder changes capacity,
not the six-family physical data recipe. Configurations: configs/models/joint-*.json.

## Loss, stages and reproducibility details

A optimizes (L_u + L_initial + MSE_log-range-D)/3 + lambda mean(z²),
lambda=1e-5. Initial-field loss inherits field_loss unless joint.initial_field_loss
is explicitly set. It gets its own t=0 segmentation and query budget; no spatial
weights apply to scalar D. Each field decodes query_points whole-domain queries
plus a fixed region_queries total when RB is enabled, not per region. This can
roughly double the field supervision/decode cost relative to solution-only A.
Physical query times are drawn in groups from the recorded initial/early/uniform/
final mixture, then averaged equally per group and solution. The training objective
is intentionally time-weighted; validation uses separate uniform-time quadrature.

D is represented by log(D/D_min)/log(D_max/D_min). The unknown scalar head uses
sigmoid in this coordinate, defining an explicit bounded positive prior. It is
not an unconstrained universal coefficient model. Decoder signed fields have linear
heads. The initial decoder ignores query t; its consistency with u(t=0) is measured,
not enforced by an exact heat rollout. No mandatory reference solver at inference.

Known D is disclosed with probability .5 by default. For nondegenerate disclosure
policies, the first two updates explicitly cover both statuses, recorded by
warmup_disclosure_coverage and per-update known/unknown counts. All later draws
follow the configured probability. A policy which never trains a status cannot
advertise inference in that status. Encoder targets do not depend on this mask.
Resume rejects loss, architecture, disclosure, recipe, normalization or schema
changes. Flow-only auxiliary/cache switches may differ from the frozen-A dependency;
changing them for an existing flow resume is incompatible. Start a deliberate new
flow stage with the existing representation to enable auxiliary training.

Optional auxiliary settings live in joint.auxiliary (field, observation, pde,
minimum_s). All beta defaults are zero. Example: field=.01, observation=.0001,
pde=.00001, minimum_s=.5. These are implementation starting points, not calibrated
physical weights. The frozen decoder receives z_s+(1-s)v after inverse latent
normalization. PDE derivatives are explicitly enabled even during no-grad validation.
The heat residual is (u_t-D laplacian(u))/(train_std / physical_time_span).
Positive-noise observation discrepancy divides by normalized sigma with floor1e-4;
exact observations use normalized squared error separately. PDE probes are eight
fixed bounded coordinates, so passing their residual does not certify the field.

Optional inference correction is available in CoordinateBundle.sample and the
coordinate API with correction={steps:1,learning_rate:.01,observation:.001,pde:0}.
At most20 gradient steps update active normalized latent groups only. Reconstruction
exports preserve uncorrected_samples, correction cost, and final consistency.
This alters sample distributions and gives no exact posterior guarantee. Base
Euler/Heun sampling needs no correction; NFE is steps or twice steps respectively.

Frozen-target cache entries are FP32, per realization. Their directory digest
includes dataset identity, frozen checkpoint hash, encoder sampling, train-only
latent normalization and schema/anchors. Large caches can consume tens of GB;
summary reports the maximum payload. Disable joint.target_cache explicitly or
provision disk. Dataset and segmentation caches remain bounded and contain no
reference autograd graphs. Cache targets are reused across disclosure masks.

BF16 and scaled FP16 are CUDA-only choices in the shared precision guard; CPU/MPS
use FP32 explicitly. Efficient attention is used for ordinary encoder/velocity
training. Coordinate decoding uses FP32 math attention for higher derivatives.
Activation checkpointing, microbatches and accumulation remain separate controls.
Use training_options.profile_components=true for synchronized component timings;
CUDA uses events and host/reference work uses wall time. Timings are inclusive,
not additive. First update is warmup; two updates are not a throughput benchmark.
Validation/checkpoint and persistent sync are recorded separately. Notebook text
refresh and plot intervals are configurable (10/120 seconds); display overhead is
recorded separately. No GPU utilization is inferred from allocated memory.

## Replacement inventory

- Replaced coordinate/model.py's solution-only encoder, decoder and velocity.
  No old set_latent_coordinate factory remains in the active path.
- Kept one coordinate/engine.py trainer and the existing stage orchestration,
  reference pipeline, field loss, observation placement, persistence and storage.
- Normal training/UI model choices, profiles and active Colab notebook use the
  joint architecture. Fixed-grid options and the Legacy page link were removed
  from the normal application. Old generic training/reconstruction CLI commands
  and old generative API resolve route reject with migration guidance.
- Registry/import rejects old architecture weights; storage identifies them as
  Legacy architecture — incompatible with current model and retains explicit
  deletion controls. No state conversion, automatic deletion or loading occurs.
- Historical fixed-grid research modules/tests and colab-legacy.ipynb are retained
  as source/evidence for old studies, not selectable normal generators. The legacy
  notebook is excluded from new source archives. This does not remove numerical
  heat/Darcy/chemistry/reference functionality or rewrite old scientific evidence.

## Next bounded scientific experiment

First compare 500/2000 representation updates at XSmall, fixed six-family splits,
three seeds and the same field objective/compute accounting. Use only validation
for selection; keep test untouched. Inspect uniform physical MSE, t=0/early error,
small/large regions, background, spectra, initial-head agreement and scalar-D error.
Do not spend a large flow budget until representation quality is adequate. Then
compare known/unknown-D flow on fixed sparse measurement sets, including ambiguity
and parameter-error/field-error tradeoffs. Region balancing is experimental: it
may trade whole-field/background accuracy for feature preservation; alpha=.5 is
an unvalidated starting point. No existing model changes when display metrics change.

Static initial-field queries are available through CoordinateBundle.decode_initial
and exported as initial_samples/initial_queries in reconstruction result.npz.
They use the same generated latent and dedicated static head, without a new draw
or solver. Extra unique-spatial-query decoding time is reported separately.
Continuous field normalization is fitted to training calibration initial grids;
one global mean/std is retained across solutions/times, so amplitude is preserved.
The current whole-field validation metric is computed after inverse normalization.
