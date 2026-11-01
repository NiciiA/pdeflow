# Capacity and native-resolution policy

Architecture and flow formulation are separate. Existing `residual_cnn` exports
remain exact-layout specialists and load with their original widths/blocks/weights.
No weights are silently expanded. New `multiscale_unet` uses periodic convolutions,
average-pool feature levels, skip concatenation, nearest replication **inside the
feature hierarchy**, and full native-grid input/output. It does not enlarge an8²
solution into a1024² field or independently tile a globally coupled PDE.

## Configurable capacity

ModelConfig records name,width,blocks,levels,conditioning_width,coordinate_features,
activation_checkpointing,temporal_handling. Supported temporal handling is explicitly
fixed_channels: named physical frames pack into channels, while generation s has
its own embedding. The new U-Net mixes those fixed-time channels; it is not a
variable-time temporal transformer. Depth is controlled by blocks/levels; metadata
counts deepest-path spatial3×3 layers separately from1×1 projections/condition MLPs.
Condition width is independent of spatial width. Configurations are editable JSON.

| Choice | Native layout used here | Structure | Actual parameters |
|---|---|---|---:|
| Small matched |heat32²,T4|Legacy CNN,width8,1 block|1,956|
| Medium matched |heat32²,T4|U-Net,width16,3 levels,1 block/level,condition32,coordinates|132,436|
| Large local |heat128²,T4|U-Net,width24,4 levels,2 blocks/level,condition48,checkpointed training activations|1,257,436|
| Native1024 execution profile |heat1024²,T3|U-Net,width8,4 levels,1 block/level,condition16,checkpointed training activations|73,939|

Parameter counts depend on fields and saved physical times; checkpoint metadata and
Models & Training show the actual count/config. Existing CNNs support1D/2D/3D;
all remain tested. New named capacity/high-resolution profiles are2D. The multiscale
code uses dimension-specific operators; no new3D scaling or1024³ claim is made.

The U-Net increases spatial context through its pooling hierarchy, with a finite
receptive field. It is not a guarantee of global Darcy coupling or scientific
quality. Coordinates enter as periodic sin/cos axes plus log spacing, domain lengths
and actual saved-time constants. Known scalar/forcing encoders remain separate.
Exact checkpoint grid/domain/origins/boundary/convention/times are still required.
Different-grid execution needs a separately trained compatible export or the
existing explicit validated refinement route. No automatic downsampling, resolution
range expansion or resolution-generalization claim is introduced.

Saved architecture metadata is validated against the constructed model/parameter
count before state loading; state_dict shape checks remain strict. Old exports
without the added field derive a display summary from their original safe config.
That is a metadata compatibility path, not a different architecture or retraining.

## Matched bounded comparison

Small/medium: same32²/T4 distribution, physical IDs, calibration IDs, root seeds,
20 optimizer updates,batch1; independent processes; fixed validation problems.

| Capacity | Validation CFM | Validation obs+PDE unobserved relL2 | Process RSS high-water mark | Total seconds including validation inverse evaluation |
|---|---:|---:|---:|---:|
| Small |1.5139575|1.4736837|277,970,944 bytes|20.2745|
| Medium |1.5300691|1.5572219|308,707,328 bytes|20.2855|

Fourier baseline on that same validation case has relL2=0.5251954. All modes,
loss histories, weight-change checks, solver costs, exact seeds/IDs and results
are saved in outputs/flow-workspace-20261026/capacity-comparison/. RSS is the
isolated process lifetime high-water mark including preparation/evaluation; it is
not isolated activation memory. One short comparison cannot establish a capacity
ranking. Larger was not better here; no test performance selected a model or budget.

## Native resolution by component

| Component |32²–512²|1024² heat|What the evidence establishes |
|---|---|---|---|
| Reference solver |Configured periodic heat|Actual finite-mode reference / native training-reference solve|Real native arrays, tolerances/classification retained; not a general symbolic solver |
| Observation sampling |Shared exact-grid operator|Actual sparse sampling from1024²/T3 reference|No snapping/interpolation; clean samples stay evaluation-only |
| Online training generation |Existing physical provider|Actual native physical training sample in probe|Full native solve, no image resizing |
| Backbone |Explicit per-grid configurations|Actual73,939-parameter U-Net forward|Native input/output; no validated cross-resolution checkpoint |
| Neural training |Named local pilots; bounded matched training|One actual CFM forward/backward/update|Resource/gradient execution only; no trained1024² accuracy claim |
| Base flow reconstruction operations |Existing exact-layout inference|Actual two-step production Heun source→candidate, one particle,T3|Native shape[1,1,1,3,1024,1024]; one-update resource probe, not scientific reconstruction validation |
| Guidance |Existing observation/PDE gradients|Actual finite endpoint gradient through model and physical decoder|Bounded gradient probe; long guided schedules unmeasured |
| FK populations |Existing bounded small-grid FK|Not exercised/validated|No default maximum-resolution population-fit claim |
| Visualization |Bounded linked planes/slices|Explicit strided display; native shape retained|Reference/slice endpoints≤128²; linked inspection≤64², no full-ensemble JSON |
| Evaluation |Physical quadrature/coordinate checks|Native metrics implementation; native reference/observation checks|No high-resolution model-quality study; different grids need explicit transfer |

Supplied configs/resolution/heat-{32,64,128,256,512,1024}.json preserve requested
native grid,T3,one batch item and practical width8. MPS probe actually used1024²;
intermediate-size configs are supplied, not individually benchmarked at full
training budgets. Only heat is the declared1024² execution slice. Darcy/chemistry
keep their existing reference budgets/checkpoints; their maximum-resolution
capabilities are not inferred from heat.3D support is retained at existing layouts.

Heat reference budget is at most1,048,576 spatial nodes and8,388,608 field-time
nodes; other families retain the2,000,000 field-time-node budget. The early bound
precedes coordinate-expression allocation. API sample-array limit remains128MiB,
not a full accelerator-memory estimate. Controller/FK grid/particle budgets remain
separate explicit limits. Default snapshots≤9 frames,4 captured particles and16MiB
of strided numerical history; native returned samples remain unchanged.

## Measured resource probes

Actual MacBook arm64 MPS,float32,heat1024²,T3,batch1,one update and2 Heun steps:
reference0.1999s; CFM forward/backward/update1.0088s; endpoint guidance0.9305s;
source-to-candidate1.2045s. RSS470,695,936 bytes; MPS driver allocation2,189,869,056
bytes and current allocation52,457,216 bytes after the probe. These allocations
are point-in-time, not peak accelerator memory; cold/contended timings vary.
The native candidate and real snapshots are retained in
probe-1024-delivery-mps/{probe.json,native-candidate.npz,generation.npz,generation.json}.

Large128² MPS also passed: actual1,257,436 parameters; one update, physical guidance,
two Heun steps; probe-large-128-mps/. No full pilot or long high-resolution training
was launched. CUDA and actual Colab execution remain untested. A successful forward
pass/update does not establish model quality, identifiability, convergence or a
calibrated posterior. If another machine cannot execute its declared grid, the
probe reports failure while preserving the config; it never silently shrinks it.

## First-class native 512² continuation

heat-2d-512 supplies an explicit MPS/float32, batch1, 10000-update profile with
73,939-parameter direct-field periodic U-Net, three times [0,.05,.1], frozen
training-only calibration and online physical generation. configs/reference/heat-512.json
provides its exact matching reference layout. The bounded MPS update/guidance/
Heun probe passed; no full pilot or validated 512² accuracy is claimed.

Exact structural dependency propagation for an interior output index256 on a
512-node axis gives94 possible source-axis nodes (indices209–302) in one velocity
evaluation for this four-level/one-block network. It is finite, not global random
state communication. Coordinates/known D and repeated ODE propagation do not by
themselves establish learned global recovery. Evidence: catalogue-512-20261026/
receptive-field.json. Default small-grid feedback limits are preserved; the UI
requires explicit native-grid budget review or feedback disabling, and the API
rejects insufficient budgets before submitting a job.

Full catalogue coverage, reviewed layouts, resource limits and commands:
[training-catalogue.md](training-catalogue.md). Prior 1024² operations remain
recorded evidence; CUDA and actual Colab are still untested.
