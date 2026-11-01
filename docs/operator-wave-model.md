# Operator-aware generative latent neural operator v2

The extension PDF is design guidance, not an executable specification or proof
of accuracy. The user request governs this implementation. It supersedes the
whole-trajectory joint latent v1 in the active dynamic workflow. Old artifacts
remain untouched and incompatible. Broader 1D–3D chemistry research remains a goal;
this executable shared specialist covers unforced periodic rectangular heat/wave2D.

## Contract before implementation

Physical state q(x,t)=[u] for heat and [u,v] for wave, v=u_t. B=realizations,
N=spatial reference samples, Q=queries, L=TOTAL slots, W=width. Physical t in[0,1]
is independent of flow s. Coordinates are physical(x,y,t), upper endpoint excluded.
Input states [B,N,2] are padded with a separate quantity mask: heat v is absent,
not a measured zero. Scalar D and wave speed c have distinct type IDs/ranges.

Generated physical slots: initial_u, initial_v, coefficient. Initial_v slots are
inactive for heat. No solution/trajectory group exists. Representation encoders
process each state quantity separately with only within-group attention; scalar
encoding sees only its scalar. Group labels alone do not prove disentanglement.
The flow couples groups through shared attention. Known scalar slots are excluded
from noise/targets/loss and supplied through typed authoritative context. Partial
initial fields stay generated. Forcing is absent, periodic boundaries known.

A1 trains a spatial state encoder/decoder and coefficient encoder/head on states
at arbitrary sampled physical times. The decoder receives space only: no t and no
coefficient shortcut. A2 freezes those encoders/decoder, trains a shared graph-
conditioned direct-time transition on full state q=(u,v), with decoded field,
frozen latent alignment and composition losses. E(z,t,t)=z by the explicit dt
factor. Coefficient slots are copied identically. B freezes A1/A2 and train-only
normalization, then trains CFM to generate encoded INITIAL states and unknown
scalar. Thus no evolving latent target, trajectory shortcut or solver inference.

Decoder evolution receives actual start/end t. Every coordinate t derivative
passes THROUGH E, not just a decoder time feature. Periodic local attention uses
a smooth cosine-distance kernel over ALL local anchors, without hard neighbour
switches. Normal field queries group equal physical times; derivative probes use
a pointwise differentiable path. Direct queries do not require sequential rollout.

One shared observation encoder, edge-aware graph encoder, flow, evolution backbone
and spatial decoder serve both families. Type-specific projections/heads handle
u/v and D/c. Family masks handle absent v. A canonical typed directed graph records
field/parameter/coordinate nodes, derivative orders, Laplacian, multiplication,
power(c,2), and residual equality. Ordered argument edges are retained. The graph
enters both flow and evolution. Validated graphs map to registered physics;
unregistered/static/weak/other boundary branches reject execution explicitly.

Observations are scalar rows with field=u or (wave only)v, actual(x,y,t), value,
absolute sigma and optional sampling metadata. No hidden reference velocity,
initial field or scalar enters inverse requests. Value/noise normalization is
quantity AND family specific, fitted on training data only. Per-example loss
averages active quantities; balanced family batches prevent the one-channel heat
state dominating or two-channel wave receiving double weight. RB alpha=.5 remains
an experimental decoded field objective, with separate signed reference masks per
quantity/time and uniform-domain validation retained.

Reference wave is finite periodic Fourier evolution with independent u0/v0,
explicit zero mode, stable sinc, physical domain frequencies and symmetric Nyquist
interpolation. Neural inference never calls this solver. Wave c range is chosen
from the saved reference previews and phase coverage, not copied from D.

## Executed design and reductions

The design source is the user-supplied *Design PDE Latent Flow Model Extension.pdf*
(read all nine pages; diagram checked visually). This is our bounded adaptation,
not a reproduction or a scientific performance claim. Scientific PDFs are untouched.

A1 objective is the mean over active normalized quantities of decoded field loss,
plus0.1 typed-scalar log-range reconstruction MSE and1e-5 latent squared norm.
A2 uses decoded field loss +0.1 frozen-target latent transition MSE +0.01 times
(latent composition MSE + decoded normalized composition MSE). Latent and decoded
composition include both u and v for wave. Weights/depths are in `evolution`.
A1 scalar/regularizer defaults are recorded in source/config; exact t=0 agreement
comes from identity evolution. Composition is a soft penalty, not a semigroup proof.
B uses masked straight-path conditional flow matching with separate group means,
then an active-group weighted mean per example. Known coefficient and absent wave
velocity groups contribute neither noise nor loss. No spatial mask is applied to
latent tokens. Optional field/observation/PDE auxiliaries act through frozen physical
modules and are off by default; bounded post-generation correction is separate.

Training field losses remain MSE, contrast-weighted MSE, blended region-balanced
MSE and MAE. Versioned missing historical loss means MSE; new-run presets now use Standard MSE.
Explicit region balancing retains experimental α=.5. See standard-mse-workflow.md.
Reference-only detached signed regions are segmented separately for u and v at the
supervised time. Periodic four-connectivity, contrast/area floors and bounded analysis
grid remain shared `field_losses.py`. Uniform-domain queries estimate the domain
term; explicitly budgeted region-stratified queries estimate each retained region
mean. Every retained region receives queries or execution rejects the insufficient
budget. Domain + region budgets apply PER active quantity; wave therefore costs two
field decodes/reference sets per time. Equal region contribution is not equal source
recovery probability; merging is per-time segmentation, not source tracking.

Physical normalizations are fixed mean/scale per family AND quantity, fitted on
training realizations only at calibration times0,.137,.4,.8. State and field gradients
optimize normalized values; physical diagnostics invert those transforms. Known D/c
are log-mapped over their distinct bounds. Scalar heads return bounded positive
physical values. Different units of v cannot dominate u just through magnitude.
A1/A2 training errors, uniform whole-field physical validation and scalar error are
separate. The gate uses comparable normalized whole-field MSE and constant baseline,
with equal families, examples and active quantities; it never compares RB loss with
an ordinary-MSE baseline. Grid metric RMSE is sqrt(aggregated MSE), not mean RMSE.

The existing initial/early/uniform/final time mixtures remain configurable in
`time_sampling`. Uniform components use32 jittered phase strata; early components
are log-uniform over configured early fractions. A1 samples complete states at one
coherent time; A2 samples target time and initial/intermediate transition pairs and
midpoint triples. Conditioning times use the independent observation policy. This
is a training-weighted objective, not uniform space–time MSE. Validation uses spatial
node means and explicit trapezoidal time quadrature; initial/early/per-time errors,
region metrics, spectra, phase/amplitude and parameter errors remain separate.
Display frames do not constrain training or arbitrary-time querying.

Heat D range is .03–.12. Wave c range is .2–.6 on the unit box. With maximum component
mode4 the highest temporal frequency is .6 sqrt(4²+4²)=3.394 cycles over[0,1]. The32
strata give about9.43 strata per highest period. Saved training previews and analytic
checks support this resolved *reference* choice; two updates do not cover all phases.
Initial fields are explicitly bandlimited; approximately constant independent u/v
examples occur every seventeenth realization after the coverage prefix. First six
cases of each enabled family have distinct train/validation/test realization IDs.
Independent initial v is the default outside labelled zero/travelling/velocity-only
cases. Observation random/even/feature-focused positions and continuous times are
separate from supervised query placement.

## Fourier and derivative conventions

Grid nodes exclude duplicated upper endpoints. For physical wave vector
k=(2π n_x/L_x,2π n_y/L_y), omega=c|k|. The reference uses
u_hat(t)=u_hat(0)cos(omega t)+v_hat(0)t sinc(omega t/π),
v_hat(t)=-omega u_hat(0)sin(omega t)+v_hat(0)cos(omega t).
At k=0 this gives u_hat(0)+t v_hat(0), v_hat(0). Each even-grid Nyquist axis is split
between±n/2 with half coefficients; corner modes split four ways. This defines a
real continuous finite trigonometric interpolant, including arbitrary spatial queries.
It does not recover an unknown continuous drawing exactly. Reference energy uses
Parseval on those explicitly split modes; its zero mode is handled without division.

Autograd differentiates E(z,0,t) AND the periodic spatial decoder. Smooth cosine
anchor distances replace hard nearest-anchor routing. Math attention is used for
frozen/evaluation physical modules and spatial second derivatives; actual CPU/MPS
higher-gradient checks pass. CUDA AMP policy is implemented but unexecuted here.
Heat residual is u_t-DΔu. Wave diagnostics use u_t-v and v_t-c²Δu, each with its own
physical quantity/time scale when normalized for auxiliary loss. Generated unknown
scalars feed their own sample's dynamics and residual; true reference scalar is
attached only after generation. Energy drift is grid quadrature with a spectral
derivative of the sampled predicted field, labelled resolution-dependent. Reference
energy and neural diagnostic energy are not interchangeable exactness claims.
Mean v is conserved by the exact wave reference; mean u changes linearly for nonzero
mean v. The learned model reports errors in these laws rather than assuming them.

## Capacity and backend accounting

Exact parameter counts (one inventory across all stages; frozen modules are the
complement of the trainable column, not additional independent family networks):

| Capacity | A1 trainable | A2 trainable | B trainable | Inference | Total inventory |
|---|---:|---:|---:|---:|---:|
| Smoke |18,868|17,904|23,376|50,620|60,148|
| XSmall |2,537,476|3,637,248|6,997,248|11,697,668|13,171,972|
| Small |6,882,052|10,229,376|19,849,344|32,459,908|36,960,772|
| Medium |14,884,868|25,516,544|49,962,496|80,228,868|90,363,908|
| Large |33,435,652|73,883,904|145,387,776|229,935,364|252,707,332|
| XLarge |78,290,948|190,061,568|375,917,568|587,017,220|644,270,084|

A1 encoder/decoder/coefficient heads train; graph/evolution wait. A2 trains the shared
edge-aware graph encoder and evolution backbone. B trains one observation encoder,
typed context encoder and coupled flow with group-specific projections. The frozen
A1/A2 modules do not move. Inference excludes the training-only state encoder, but
includes graph/evolution/decoder/scalar heads, observation/context encoders and flow.
Detailed encoder/decoder/graph/evolution/conditioning counts are in saved JSON.

Large still means65,536 training realizations, width768/depth16/384 TOTAL slots and
100k updates for EACH of the three stages. XLarge is larger and unprofiled. No GPU-fit
promise follows from meta counts. The current engine accumulates one realization
at a time, respecting configured microbatch as an upper bound; microbatch>1 does not
yet give vectorized throughput. Flow activation checkpointing is implemented. Workers
and pin-memory settings do not currently accelerate procedural generation. Step wall
time is synchronized; cumulative component clocks are host-inclusive, not individual
CUDA kernel timing. Do not add nested clocks or extrapolate a GPU speed claim from CPU.

## Limits and next bounded experiment

No useful accuracy, calibrated posterior, exact conservation, universal equation
understanding, or benefit of joint training is demonstrated. Static solution operators,
weak/shock handlers, forcing, nonperiodic boundaries and other neural dimensions or
geometries reject execution. Larger local-anchor inference cost scales with all anchors;
there is no hard-neighbour shortcut. The training preview uses bounded training-only
scale estimates; the actual trainer records its full calibration. The old fixed-grid,
solution-only and joint-v1 artifacts remain incompatible and untouched.

Recommended next slice: controlled A1 learning first (fixed train/validation IDs,
seed, architecture, α=.5 and recorded query budget), then A2 with the frozen chosen
representation; compare whole-field, u/v initial/early, region, spectral/phase and
invariant errors. Do not allocate substantial B training until A1 beats its comparable
constant baseline and A2 demonstrates dynamics on validation. Use validation, never
test performance, to choose budgets. Region balance can sacrifice domain/background
accuracy; α=.5 is experimental, not a validated optimum. A scalar metric change in
the UI never retrains an existing checkpoint.
