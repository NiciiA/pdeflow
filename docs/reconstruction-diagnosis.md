# Reconstruction diagnosis — 1 November 2026

## Findings and attribution limits

The current saved playground jobs use the imported Medium physical-state v2 model
`ce489d7b97c402317349c712545fb599bbc36e30abccd107f92abdad0e2a206e`, at
`outputs/local/model-imports/2de8489ab6f34bc7985a583451f62b6a/coordinate.pt`.
This is **not** a two-update smoke model. Its B history records2,500 successful /
2,500 attempted updates,0 skipped updates,80,000 examples and finite gradients.
The preceding A2 dependency records2,500 updates and SHA256
`c19cbf74df7136d376b8b88ab6236390b8fd09746da10e3fa4f3531a30910fc8`.
That file, the matching A1 encoder and gate outcome are not present locally.
A1's2,500-update value is a configured budget, not independently verified execution.
No matching Colab stage bundle was found. Do not substitute another local encoder.

Three saved jobs match the described types of poor reconstruction and replay
**exactly** on MPS (maximum field difference0):

| Case | Saved job | Measurements | Physical times |
|---|---|---:|---|
| Periodic product cosine | `068a8fe3998840b78d2aeca105d9c418` |150 u, noiseless | all at0 |
| Compact hotspot | `4c99bd67f9f042b7bc5d47a10fa725b2` |50 u, noiseless | all at0 |
| Travelling wave | `1728129bf47948a69f15a249aba1a5e0` |50 u, noiseless | all at.15 |

The screenshots themselves and browser-selected displayed time were not attached
to this request. These are the actual saved jobs, not a claim to have recovered an
unrecorded screenshot frame. The displayed playground statistic is the ensemble
mean of four samples; the reports also retain every sample.

The earliest **demonstrated** loss in these reconstructions is already present in
the generated initial state, before physical evolution. For the periodic case,
initial u RMSE=.475579 and exact conditioning-sensor RMSE=.653284. Learned T is
identity at t=0, so A2 evolution is not required to explain that error. Disclosing
true D gives initial RMSE=.478729 and sensor RMSE=.656269: incorrect D alone does
not explain it. **This does not establish whether A1 encoding or B is the first
bad trained stage.** A1/A2 oracle diagnostics are implemented and tested, but
reported unavailable for this imported export until the matching encoder arrives.

The periodic observations are informative within the declared max-wave4 Fourier
initial-field subspace:150 equations,81 real coefficients, rank81, minimum singular
value.12794. Observation-only least squares gives sensor MSE1.53e-30 and initial
field MSE2.67e-29. It uses no hidden field/parameter for fitting. This is a restricted
linear information control, not a general uniqueness proof or neural prediction.
D remains unidentifiable from t=0 measurements alone. The hotspot has rank50/81;
its unregularized minimum-coefficient-norm fit has initial MSE133.16 despite tiny
sensor error. Sensor interpolation alone is not accuracy or a reliable inverse.

## Verified model, budget and input inventory

Width512,256 total tokens,12 flow layers,8 heads,32² reference recipe,512 encoder
points,256 domain queries and256 region queries per supervised time group. The stored generic time_groups=4 setting is not used as four simultaneous states
in this operator path: A1/A2 plans force one coherent physical-state time per
example/quantity, continuously sampled across updates. No-region region slots
are zero-weight padding; no retained region is silently dropped. Equal
heat/wave sampling; heat state u, wave state (u,v), with v=du/dt. Scalar D/c has its
own group; c is mapped to c² only by wave physics. Initial-state E, spatial D,
learned graph-conditioned T and observation-conditioned flow are separate.

| Module | Parameters, excluding buffers |
|---|---:|
| State encoder (omitted from portable export) |10,135,040|
| Spatial decoder |3,693,058|
| Coefficient module |1,056,770|
| PDE graph encoder |2,637,824|
| Learned evolution |22,878,720|
| Conditional flow, including observation/context encoders |49,962,496|
| Full training model |90,363,908|
| Portable inference parameters |80,228,868|

The imported export has strict complete state dictionaries for every inference
module. No `strict=False`, missing-key substitution or random encoder fallback is
used. It runs float32/eval on CPU/MPS; the checkpoint records CUDA/FP16 training,
not a new local CUDA verification. Recorded scaler values are65536 and131072,
with no skipped/nonfinite updates. B loss decreases3.3255→.33794; this is latent
velocity loss, not decoded field accuracy. Latest successful boundary is exported;
no best-physical-validation selection is recorded.

Current config is effective32, actual microbatch16,2 accumulation passes,2,500
updates and80,000 example exposures per declared stage. Medium's unmodified preset
has10,000 updates/effective8: the same arithmetic example exposure does not imply
identical optimization. The export's initial config already says32/2,500 and has
no continuation history, so the earlier conversion itself cannot be reconstructed.
Warmup1,000 consumes40% of this update budget; executed B LR starts3e-7 and ends
3.00003e-5. This is a possible optimization constraint, not an established cause.
A2's default schedule has no warmup. Actual A1/A2 schedules/gate results require
the missing matching records. An unrelated local10k A1 passed its previous minimal
constant-baseline check; it is **not** evidence about this imported model.

Training normalizations are global, train-fitted and quantity-specific (heat-u
scale.31696025, wave-u.42403877, wave-v3.24093127), not per-case rescaling. Latent
mean/scale and fitted identities are inventoried; the scale floor is.01. All
reported errors below are after inverse normalization in physical units.

Distribution differences are explicit: the periodic case has150 sensors versus
the trained1..64 range; its spatial modes are within the declared band. The hotspot
has about.004788 of initial-u power beyond that band and is narrower than the
training pulse recipe. All-one-time and u-only sensor sets differ from randomly
mixed training sets. These flags do not establish causality: held-out cosine, signed-feature and nonzero-initial-velocity wave cases
using the exact training observation sampler also reconstruct poorly. Validation identities are separate from all training/normalization IDs.

## Physical field and exact-sensor results

Ensemble mean of the same four samples; domain metrics use spatial node means and
normalized trapezoidal physical-time quadrature. Saved UI cases cover[0,.15];
validation cases cover[0,1] on the listed nonuniform times. Do not rank their
aggregate RMSEs as matched tasks. Early errors can be hidden by late heat decay.

| Case / quantity | Initial RMSE | Domain/time RMSE | Conditioning sensor RMSE | Held-out sensor RMSE |
|---|---:|---:|---:|---:|
| Periodic u |.475579|.293518|.653284|.290591|
| Hotspot u |.150112|.119490|.230791|.111537|
| Saved wave u |.652477|.627673|.616498|.643126|
| Saved wave v |2.064126|2.047609|not measured|2.024603|
| Training-policy validation cosine u |.516530|.077878|.311437|.250258|
| Training-policy validation signed u |.246000|.121167|.318750|.190508|
| Training-policy validation wave u |.486396|.490295|.387597|.557589|
| Training-policy validation wave v |3.625566|3.610259|3.581746|3.590195|

The training-policy cases use the exact shared training observation sampler:51
measurements each, physical sigma about.007107 for heat and.011417 for wave,
with the trained count/placement/time/quantity policies and known coefficient
status. Standardized ensemble conditioning residual RMS is43.82/44.85/193.36
respectively; these are descriptive fit errors, not a posterior calibration test.
The earlier three validation-field cases with designed clean sensors are retained
as additional controls, not presented as fully matched conditioning distributions.

Each report also contains per-sample MSE/RMSE/MAE/relative L2, physical means and
variances, training-mean constant baseline, recorded-alpha region-balanced metrics,
region-size/error summaries, background error, peaks/troughs and their locations,
complex Fourier low/high-mode error (phase retained), early/per-time errors and
signed mean differences. Wave combined scores explicitly divide each quantity's
MSE by its training scale squared before averaging. Physical invariants are
resolution-dependent diagnostics, not accuracy proofs.

Held-out sensors are64 independently placed, saved evaluation-only queries at
recorded physical times, never fed to generation. They are not the conditioning
set with entries relabelled. Predictions are queried at exact x,y,t,quantity,
not nearest plotted pixels. Positive-sigma residuals divide by supplied physical
sigma. Sigma=0 has separate errors and tolerance1e-5+1e-4*abs(value), no invented
uncertainty. Empirical interval summaries for four samples are not calibration.
Signed marginal profiles retain their physical integral/RMS definitions.

## Conditioning and integration controls

Checkpoint, noise tensor, parameters, query grid and solver are fixed across each
ablation. Heat periodic field RMS changes on a common12² first/last-time probe:

| Change | Field RMS change from correct observations |
|---|---:|
| Permute rows |1.61e-7|
| Shuffle values among fixed positions |.000692|
| Shift values by.2 training standard deviations |.050212|
| Nested half-count |.016545|
| Masked NaN padding |context difference0, finite|

Thus observations affect the output, but this heat example shows weak sensitivity
to value/location associations. This is not proof of useful conditioning. The
held-out wave-field control with mixed clean sensors supports mixed/u-only/v-only controls; final shuffling
preserves u/v quantities and nested count reduction retains both. See
`validation-wave-quantity-preserving-ablations.json`. The initial mixed-wave
shuffle/count records are retained as superseded diagnostic controls: they mixed
u/v units or changed the quantity mixture. No scientific conclusion uses those
confounded records. No-observation inference is correctly unsupported for this
checkpoint (unconditional_probability0 and minimum count1); it is not forced.

Heun4/8/16-step controls on the same small probe give u MSE.11745/.13089/.14047;
8 vs16 field RMS difference=.06140. Integration is not converged at8, but increasing
steps does not repair field accuracy here. No step count is selected using these
reference errors. Sampling direction s=0→1, active groups, parameter masks and
straight path match training. For Z_s=(1-s)noise+s*target and v=target-noise,
Z_s+(1-s)*v=target; the existing clean-endpoint auxiliary uses that expression.
Physical t, flow s, physical wave v and latent flow velocity are distinct.

## Confirmed corrections and what they do not establish

1. **Masked padding:** explicit quantity IDs previously skipped NaN sanitization.
   Padding now remains inert for all call forms. Real saved jobs had valid rows,
   so this is not their demonstrated cause. Strict valid-row replay remains exact.
2. **Missing export provenance:** new stage checkpoints carry actual successful /
   attempted/skipped/example budgets and schedules through A1→A2→B. Colab export
   attaches the available representation-gate record/override and identifies the
   matching A2 file needed for diagnostics. Missing old provenance stays unknown;
   existing artifacts are not rewritten or labelled as gate-passed.
3. **Noiseless training coverage:** the encoder contains sigma.eq(0), but old
   Uniform[0,.05] sampling had no explicit atom at zero. In this actual checkpoint,
   that input column is exactly its initialized value; other columns changed by
   up to.06797. The initialization audit reproduces the engine's seed/construction
   order. Zeroing only this feature changes initial RMSE.475579→.474874; conditioning
   with sigma1e-6 gives the same limiting result. It is a coverage gap, not the main
   measured explanation for the large error. No inference-time sigma is coerced.

   Opt-in `joint.noiseless_probability=p0` implements
   p0*delta_0+(1-p0)*Uniform[0,sigma_max]. It requires a range including zero.
   It reuses the old uniform quantile, so coordinates/counts/quantities and later
   RNG draws stay paired. At p0=0 or absent, values/RNG are exactly backward
   compatible. Defaults/presets remain0: existing studies are not silently changed.
   New B runs may explicitly test.1; this is not a validated optimum. Changing it
   on resume is rejected; physical dataset/representation compatibility is kept
   separate. Resolved config, checkpoints and counts persist the setting.
4. **UI meaning:** the collapsed Reconstruction diagnostics view reports ensemble
   sensor MSE/RMSE/MAE by quantity, checkpoint identity and flow updates. It reads
   saved numerical predictions and does not rerun inference or modify arrays.

A1/A2 region-balanced field reconstruction, alpha=.5, is unchanged. Both supported
quantities use the recorded reference-only periodic estimator. A2 also uses latent
alignment and latent/decoded composition terms. B's group-balanced latent velocity
loss remains separate; optional decoded auxiliary weights in this model are0.
No new observation-to-decoder shortcut or wholesale architecture was added.

## Bounded correction experiment: actual Medium case

The informative but poorly fitted periodic case justifies a reachability probe.
`coordinate-diagnostic-correct` optimizes only active standardized initial/unknown
parameter latents through the frozen T/D at exact measured coordinates/times.
Known coefficient/absent groups stay fixed. The unknown scalar stays within its
trained positive domain through the existing decoder. Its objective is mean sensor
error plus.01 times mean squared departure from the original standardized latent;
20 predeclared Adam steps at LR.02, no reference-based stopping/sample selection.

For positive sigma it uses exact noise-standardized residuals. For sigma=0 it
uses deterministic squared error in recorded training-field units, **not** an
assumed uncertainty. The quadratic term is a local trust penalty, not evaluation
of the learned generative prior density. Tiny positive sigma fails explicitly if
arithmetic becomes nonfinite; it is not silently floored or replaced.

| Periodic ensemble metric | Original | After20 correction steps |
|---|---:|---:|
| Exact conditioning sensor RMSE |.653284|.092363|
| Held-out sensor RMSE |.290591|.224809|
| Initial u RMSE |.475579|.176351|
| Domain/time u RMSE |.293518|.216893|
| Maximum absolute heat mean drift |.080978|.055144|
| Maximum sampled L2-energy increase |.001207|.000938|

Frozen tensor hashes are identical; inferred D values are unchanged (t=0 cannot
inform D). Runtime28.4s includes generation, before/after evaluation and frozen
hashing after model loading, not just20 optimizer steps. This is a single-case
improvement and a probe of decoder reachability, **not** a retrained model,
calibrated posterior, exact correction, accurate heat solver or proof E works.
Mean conservation/dissipation still fail. Optimized latents can leave the encoder's
training manifold despite a small trust penalty. Production sampling is unchanged;
correction is an explicit separate diagnostic command.

## Deterministic diagnostic baseline

`coordinate-diagnostic-baseline` reuses the actual frozen observation/typed physics
encoders, geometry, E, T and D. A new one-attention-layer head predicts standardized
initial latents (known/absent groups masked) without generative ODE integration.
Only that head trains against E(X0); this is latent regression, not flow velocity
training or an assertion about decoded conditional means. It uses the production
training observation sampler and independent train/validation identities, with
fixed saved observation sets for a tiny-subset overfit control.

The actual Medium baseline is **unexecuted: matching E is missing**. No target was
invented. A20-update CPU execution check on the separately labelled two-update
pipeline fixture reduced train latent MSE1.27814→.89489, while decoded u MSE worsened
1.97543→2.03422; validation latent1.28098→.97529, decoded u2.17671→2.23479. Frozen
in-memory tensor hashes and checkpoint files stayed identical. This demonstrates
why lower latent loss is insufficient, not that real observations lack information.
No new decoded FM auxiliary or decoder cross-attention remedy is enabled: their
causal justification awaits the matching A1/A2 oracle result.

## Exact runnable commands

From the repository root after activating the environment. Every output must be
fresh; none of these commands overwrites an old run.

```bash
source .venv/bin/activate
PDEFLOW_MODEL=outputs/local/model-imports/2de8489ab6f34bc7985a583451f62b6a/coordinate.pt
pdeflow coordinate-diagnostic-case --checkpoint "$PDEFLOW_MODEL" \
  --job outputs/local/workbench/jobs/068a8fe3998840b78d2aeca105d9c418 \
  --output outputs/my-periodic-case
pdeflow coordinate-diagnose --checkpoint "$PDEFLOW_MODEL" \
  --case outputs/my-periodic-case --output outputs/my-periodic-diagnosis \
  --device mps --ablations --integration-steps 4,8,16
pdeflow coordinate-diagnostic-correct --checkpoint "$PDEFLOW_MODEL" \
  --case outputs/my-periodic-case --output outputs/my-periodic-correction \
  --device mps --steps 20 --learning-rate .02 --prior .01
```

For an independent training-like validation case, replace `--job ...` with
`--validation-index 0 --training-observations` (heat cosine),6 (heat hotspot),8 (signed features), or3
(travelling wave with nonzero v0); choose explicit `--steps 8 --particles 4
--resolution 32`. The flag uses the actual shared training observation sampler;
without it, diagnostic validation fields use a separately labelled clean sensor
design. Saved user measurements are never regenerated. For a custom saved reference: `--reference REF_DIRECTORY
--observations ROWS_JSON --problem INVERSE_PROBLEM_JSON`. A reference alone never
silently regenerates the measurements. Case directories contain reference initial
u/v, generator metadata, exact rows/noise/held-out rows, explicit flow noise,
queries/problem/settings, original saved arrays where available, and hashes.

Once the **matching** A2 file is restored (it embeds frozen E), run:

```bash
pdeflow coordinate-diagnose --checkpoint "$PDEFLOW_MODEL" \
  --case outputs/my-periodic-case --evolution /path/to/matching/evolution.pt \
  --stages a1,a2,b --output outputs/my-oracle-diagnosis --device mps
pdeflow coordinate-diagnostic-baseline --checkpoint "$PDEFLOW_MODEL" \
  --evolution /path/to/matching/evolution.pt --output outputs/my-direct-diagnostic \
  --steps 20 --train-cases 4 --validation-cases 4 --device mps
```

Optionally supply `--representation /path/to/matching/autoencoder.pt` to verify
A1's dependency hash and frozen E/D/coefficient tensors as well. A wrong hash or
missing/unexpected key is an error, never repaired with defaults. Unavailable
oracle stages are explicit in diagnosis.json and CLI output.

For Colab, reload the delivered source ZIP through the notebook's source-archive
cell after restarting the session; keep RUN_TRAINING=False. The notebook science
implementation stays in the shared package. To obtain the missing dependency from
the **original matching training session**, not another run:

```python
from google.colab import files
from pdeflow.runtime import file_digest
path = session.root / 'evolution/evolution.pt'
assert file_digest(path) == 'c19cbf74df7136d376b8b88ab6236390b8fd09746da10e3fa4f3531a30910fc8'
files.download(str(path))
```

If the original session is gone, restore its saved snapshot; the portable model
alone cannot reconstruct the omitted encoder. On Colab the same CLI supports
`--device cuda`, but these new commands have not been executed on CUDA here.
To request the explicit noiseless mixture for a future **new** controlled B run,
merge `"joint":{"noiseless_probability":0.1}` into existing ADVANCED overrides
instead of replacing the other settings. Preserve source/dependency checks; no
old checkpoint/source hashes or scientific gates are edited to resume it.

## Evidence, verification and next experiment

`outputs/reconstruction-diagnosis-20261101/` contains inventories, all nine immutable
cases, raw fields/metrics, source snapshot, tests/failures, shared-scale figures,
noise-column/feature controls, integration probes, correction, pipeline fixtures,
source ZIP/manifest and preservation hashes. Case/evidence ZIP excludes weights;
source ZIP excludes outputs, datasets/checkpoints, caches, environments and personal
files. Figures are regenerated from saved arrays without rerunning inference;
initial-state figures and evolved figures are separate, with correct periodic node
centres and common physical colour limits. `summary.json` is the compact table.

CPU tests check exact sensor x/y/t/quantity, physical normalization, near-zero/noisy
metrics, replay, strict dependencies, missing tensors, permutation/NaN padding,
straight-path endpoint, region estimators/gradients, frozen baseline/correction
weights and heat/wave reference isolation. Actual MPS replays, mixed-quantity
ablations and final-source noiseless fixtures passed. Frontend tests/build and lint
ran; no new CUDA/Colab execution or substantial model training is claimed. Failed
development checks are retained, including a test assertion that confused
coefficient.encoder with state encoder, a wrong test device type, a t=0 invariance
assertion initially using multiple-time sensors, an MPS validation tensor bug in
the new correction caller, and correctly rejected stale-source fixture reuse.

The smallest next step is the matching E/T oracle, **before further main-model
training**. If initial E/D fails, use a bounded fixed-training-subset A1 overfit
control before training B; no capacity increase is justified yet. If E/T passes,
the ready20-update/four-training/four-held-out direct head is the bounded next
training experiment. Only then consider a controlled B conditioning/observation
auxiliary ablation. No test-based budget or architecture winner is selected.
