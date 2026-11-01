# Field reconstruction losses and heat error metrics

The **representation autoencoder training loss** now supports `mse`,
`weighted_mse`, `region_balanced_mse`, and `mae`. These are optimizer objectives:
errors from the coordinate decoder backpropagate through the representation
encoder and decoder. Latent flow still uses its unchanged mean squared CFM
velocity objective. Spatial masks are never assigned to latent tokens.

New heat/wave runs default to **Standard MSE**. Choose **Region-balanced MSE —
experimental** explicitly; alpha remains .5 as a starting value. Local presets,
UI, CLI and Colab agree. The selection affects decoded A1 and A2 field losses,
not B FM, observation likelihood, physics or display metrics. Standard mode skips
region detection/reference frames/cache and adds zero region queries; its original
domain sampling/count is unchanged. Advanced region controls appear only for RB.
See [current workflow and compatibility](standard-mse-workflow.md).

Recorded historical runs preserve their loss, alpha and estimator. Empty/absent
legacy settings have versioned ordinary-MSE semantics; genuinely ambiguous resume
metadata fails rather than inheriting a new preset. No existing model is retrained
or migrated automatically. The existing 1D/3D and fixed-grid branches remain unchanged.

Region balancing is experimental. It intends to preserve distinct small and
large features, can sacrifice whole-field/background accuracy, and depends on
region definitions. α=0.5 is a starting point, not a validated optimum. Equal
region importance does not imply equal source-recovery probability.

## Training UI and Colab

Start the existing app from the repository root:

```bash
source .venv/bin/activate
python scripts/dev.py
```

Open http://127.0.0.1:5173 → **Train model**. Heat2D / resolution-flexible shows
**Field reconstruction loss**. Only the chosen objective's ordinary controls
appear: α for region balancing, λ for contrast weighting. **Advanced region
analysis** contains thresholds, physical-area fraction, smoothing, analysis
resolution, time groups and the fixed extra query budget. **Preview regions on
training examples** explicitly generates only training-split examples and uses
the exact global training-calibration scale. Its bounded live calibration allows
up to 512 solutions; larger calibration budgets need saved reports. Test examples
are never exposed for tuning. The preview is at initial physical time.

The notebook's settings cell exposes `FIELD_LOSS` and `FEATURE_BALANCE` and merges
`ADVANCED['field_loss']` using shared repository code. All Large/Extra-large data,
capacity and update budgets remain intact. `training_preset('large')` is an
explicit new-run variant; `preset('large')` retains historical MSE. No scientific
algorithm is implemented in notebook cells. On resume the notebook adopts the
recorded run's loss before profiling/running, rather than the new-run selector.
Direct `Session` callers must supply recorded settings; incompatible changes
are rejected before mutating the session record.

## Definition, units and regions

Let σ be the frozen training-only global scale and μ its global mean. Training
optimizes errors between `(prediction−μ)/σ` and `(reference−μ)/σ`, not per-sample
normalized amplitudes. Physical metrics are evaluated after inverse normalization.
Thus normalized squared error is physical squared error/σ²; MAE scales as 1/σ.
Physical origins/lengths and periodic cell areas are retained.

For each solution and supervised physical time, with e=prediction−reference:

```text
MSE_domain = ∫ e² / |Ω|; MAE_domain = ∫ |e| / |Ω|
WMSE = ∫ w e² / ∫ w
L_RB = (1−α) MSE_domain + α/K Σ_k [∫_(R_k) e² / |R_k|]
K=0: L_RB = MSE_domain; 0≤α≤1
```

No extra equal-weight background region is appended. Background is the complement
of retained contrast regions and already contributes to the domain term. Regions
are contrasts relative to the current spatial mean, so a sufficiently contrasting
broad cold plateau can itself be a region; labels do not establish source identity
or a known physical background. Positive and negative regions never merge.

`src/pdeflow/field_losses.py` is the shared training/evaluation/preview implementation.
Defaults are:

| Setting | Default / meaning |
|---|---|
| α |0.5 |
| λ (`contrast_strength`) |2 |
| Relative threshold |0.25 of peak absolute contrast |
| Absolute threshold |0.01 σ in physical field units |
| Numerical relative floor |1e-6 × max(max absolute reference, σ) |
| Minimum region area |0.003 of physical domain area |
| Smoothing |0, optional Gaussian σ_s = fraction × shortest domain length |
| Analysis resolution |32×32 periodic endpoint-excluded nodes |
| Continuous time groups |4 per solution per update |
| Additional region queries |256 total per solution, across all groups/regions |
| Estimator |`grouped_stratified_v1` for explicit new loss configs |

Detection uses an optional periodic Fourier Gaussian smoothing, then contrast
`a_signed = detection − spatial_mean(detection)`. The effective threshold is
`max(relative_threshold × max|a_signed|, absolute_threshold × σ,
numerical_relative_floor × max(max|detection|, σ))`. If peak contrast does not
exceed the absolute/numerical floor, there are no regions. Separate positive and
negative masks undergo **four-neighbour periodic connected components**, including
edge and corner wrapping. Components smaller than the minimum area fraction are
removed as part of the recorded segmentation definition. Pixel count alone is
not the criterion; area fraction × LxLy defines physical area. Optional smoothing
is a physical length, not a fixed number of pixels. No Gaussian fitting or temporal
source tracking is used.

Contrast weights use the **unsmoothed reference analysis field**:
`a=|u−mean(u)|`, `A=max(a)`, `ε=max(absolute_threshold σ,
numerical_relative_floor max(max|u|,σ))`. When A>ε,
`w=1+λ a/(A+ε)`; otherwise w=1. The bounded weight is between 1 and1+λ.
This suppresses amplification of nearly uniform late-time numerical noise.
All masks, thresholds, labels and weights are detached and reference-derived.
No predictions affect region selection. No effective-weight cap is applied;
`cap_activations=0` is recorded. Existing optimizer gradient clipping is separate.

## Continuous-time estimator and cost

Continuous times are not reverted to snapshots. A few times per solution are
sampled from the existing query-role initial/early/uniform/final mixture, with
continuous early/uniform draws and explicit endpoint atoms. With fewer than eight
groups, mixture components are sampled stochastically. Group means get equal
weight 1/G, then solution means equal weight 1/B. Different point or region counts
cannot make one time or solution dominate. Snapshot-mode explicit losses instead
use the saved physical times equally. Historical MSE keeps its existing sampler.

At each group, reference analysis evaluates a bounded N×N field on demand;
a32-entry LRU reuses identical reference/time/settings/normalization combinations.
No full space–time population array or reference autograd graph is retained.
Scientific reference grid, analysis grid, supervised queries and output grid
are distinct. Region cells are periodic uniform-area Voronoi cells around the
analysis nodes; this is a finite analysis-grid approximation of continuous regions.

- `query_points` is the **total whole-domain budget per solution**. Split it
 across groups; each group's uniform spatial queries estimate domain accuracy.
- Region-balanced MSE adds exactly `region_queries` decode slots per solution,
 allocated across all retained time-regions, at least one per region. Draw cells
 uniformly within each region, then jitter uniformly within its periodic cells.
 Each point in region k/group g has coefficient `α/(G K_g n_gk)`.
 Domain points have `(1−α)/(G n_g)` when K_g>0, otherwise1/(G n_g).
 The region term is therefore equal-region, not equal-area/pixel.
- If the explicit fixed region budget is smaller than the number of retained
 time-regions, fail descriptively. No region is silently dropped for zero queries.
 Groups with no regions use ordinary MSE. Unused fixed decode slots are padded
 with zero loss weight; logs distinguish active and decoded queries.
- Weighted MSE uses uniform domain queries with `w(cell)/mean_analysis(w)` density
 factors. Cell weights are constant on analysis Voronoi cells. Its denominator
 is deterministic analysis quadrature, not a noisy ratio from a focused sample.
 This estimates that declared piecewise-weighted integral; analysis discretization
 remains an approximation.
- MSE and MAE use the uniform domain queries only. Domain/time and extra-region RNG
 streams are separated, so controlled comparisons keep the same underlying data,
 times and domain queries across objectives. No density correction is omitted.

Default Development queries rise from256 to256+256=512 per solution for RB;
Large rises from512 to768. Pipeline profiles can have a larger relative increase.
These counts, analysis settings, group times/region counts, active/decode counts,
data and step seconds are logged. Equal update counts do **not** mean equal compute.
Observation placement is separately recorded inference conditioning, never the
supervision proposal. Early-time weighting and spatial balancing act separately:
the training objective is **not uniform space–time MSE**.

Regions are segmented at each supervised time. Diffusing features can merge into
one region or disappear below significance. Their original source identities do
not retain separate loss contributions. Initial/early metrics remain necessary.

## Validation and comparison display

Every objective reports physical MSE, RMSE, MAE, relative L2, contrast-weighted MSE,
region-balanced MSE, per-region area/MSE, mean/worst region error, background MSE
where defined, and initial/per-time/time-stratum metrics. Spectral-power errors
remain available. Uniform validation is separate from the recorded time-weighted,
normalized training objective. The representation gate still compares whole-field
uniform MSE with its **matching whole-field constant baseline**, never RB loss with
an MSE baseline. Passing that ratio is only a minimal check.

Per-time metrics use periodic node-mean spatial quadrature. Trajectory summaries
use normalized trapezoidal physical-time weights on the listed times; solution
summaries weight each solution equally. Aggregate RMSE is sqrt(aggregate MSE),
not the mean of per-solution RMSEs. Aggregate relative L2 uses the corresponding
aggregate reference squared norm and is unavailable for reference RMS≤1e-10.
The existing gate retains its 11 equally spaced time quadrature; auxiliary time
strata and the additional time-grid metric set are explicitly identified.
Explicit grouped-loss validation grids are at least as fine as region analysis.
A requested coarser external comparison that misses a retained region reports RB
metrics unavailable with a reason, not a falsely complete region average.

In the playground, **Error metric** selects one scalar at the current exact
physical time from the existing predictions. **Error map** is independent:
signed, absolute, squared, weighted squared contribution, or RB contribution.
**Show detected regions** adds numbered boundaries. Details show actual policy
and errors. Changing these controls never changes a trained model, training loss,
observations, or inference samples. Paused comparison waits for the exact-time
saved-latent decode; playback asks the user to pause for scalar evaluation. No
whole-field error is invented in measurements-only mode. The spatial marginal
profiles continue to integrate the physical heat fields.

For equal-area evaluation nodes and retained masks with fractions f_k, the RB
contribution map is `[(1−α)+α/(K f_k)] e²` inside region k and `(1−α)e²` outside;
with K=0 it is e². Its **spatial mean** equals the RB scalar. WMSE's map is
`w e²/mean(w)`. RMSE/relative L2 have no independently labelled pointwise map.
Absolute/squared maps retain the interval error bound. Contribution colour limits
only grow as new times are evaluated and reset for a new result; no extra inference
or future-time solver calls are made to estimate those scales.

## Configuration, resume and future controlled comparison

All loss/threshold/analysis/query settings are in resolved config, run identity,
checkpoint config and explicit field-loss provenance, logs and portable exports.
Changing type, α, thresholds or estimator is an incompatible continuation; start
a new run. Source compatibility rules remain in force: resuming an old scientific
source needs that source or the existing explicit recorded source-change override.
Inference from existing exports remains supported without training data.

New Colab/local programmatic config:

```python
from pdeflow.colab import training_preset
config = training_preset("medium", "mps", "fp32", {
    "field_loss": {"type": "region_balanced_mse", "alpha": 0.5}
})
# Write config.to_dict() to your own JSON for the existing coordinate-train CLI.
# For historical point-sampled MSE, use preset(...) or field_loss={} in a fresh config.
```

Low-level local execution/resume, using your saved configuration and fresh paths:

```bash
pdeflow coordinate-train --config my-config.json --stage autoencoder --output outputs/my-rb/ae
pdeflow coordinate-train --config my-config.json --stage autoencoder --output outputs/my-rb/ae --resume
pdeflow coordinate-train --config my-config.json --stage flow --autoencoder outputs/my-rb/ae/autoencoder.pt --output outputs/my-rb/flow
```

Plan the reproducible four-objective comparison, **without training**:

```bash
python scripts/compare_field_losses.py --preset medium --updates 200 --device mps --output outputs/loss-comparison-plan
```

To execute later, use the same command with `--run` and a **new empty output root**.
CPU/MPS/CUDA are explicit device choices. The four variants hold architecture,
physical recipe/splits, seed policy, time/observation distributions, optimizer and
update budget fixed. Their measured supervision cost differs and is recorded.
Only validation is evaluated; test stays untouched. No winning loss or training
budget has been selected. Next scientific step: inspect small/large-region,
background, early-time and spectral validation tradeoffs across repeated seeds
before funding substantial flow training. Better representation is not guaranteed.

## Evidence and limitations — 29 October2026

Evidence root: `outputs/field-losses-20261029/`. See field-loss-verification.md for
exact commands, overlapping test counts, actual CPU/MPS/notebook/browser execution,
retained checkpoints/screenshots, initial failures and preservation record.
The new models are two-update pipeline checks, not accuracy evidence. CUDA and
actual Colab/Drive execution were not run. This objective is implemented for the
scalar periodic 2D coordinate heat representation, not other PDE families, 3D
segmentation, persistent source tracking or a decoded auxiliary flow loss.
