# Observation-aligned training and complete recovery

This revision implements objective and persistence changes, with no capacity
increase or architecture replacement. Read `observation-aligned-change-map.md` for
the PDF assessment. The PDF is design guidance, not proof of this adaptation.
Its cited Tanyu review is not present in the repository; no code or theorem from
that review was reused. The modular 1D–3D chemistry/steering goal remains; the
active operator specialist is smooth, autonomous, periodic scalar2D heat/wave.

## What remains structural, and what is experimental

A1 represents complete physical states: heat u, wave displacement u and velocity
v, plus typed positive D or c. A2 learns a complete-state evolution operator T;
B generates an initial state and optional unknown scalar using observation-
conditioned flow. T(z,ta,tb)=z+(tb-ta)/Tmax R(...) on active dynamic slots, with
static scalar preserved and exact identity at ta=tb. Known scalars override the
scalar decoder exactly. Signed u/v remain unconstrained. Periodic spatial
features give periodic value/derivative behavior; they do not guarantee the PDE.

The observation set encoder retains x,y,t, quantity, value, sigma and validity,
uses an explicit null token and masks padding before feature construction.
Permutation/padding and meaningful observation sensitivity regressions are
retained. Weak conditioning/poor representation quality remains a scientific
problem. No richer attention or direct observation-conditioned decoder was
justified by the available diagnosis. Stage C remains deferred: joint updates
would change latent targets/calibration and require an alignment strategy.
No new PDE, shocks, weak/entropy solver, forcing or geometry was added.

## Exact objectives and defaults

Historical configurations retain their recorded field objective (absent means
ordinary MSE). New local/Colab training presets now select Standard MSE. Explicit region-balanced
MSE retains alpha=.5 as an experimental starting value. See standard-mse-workflow.md. Its region/query estimator is unchanged: per realization, quantity and
supervised physical-time group, uniform domain queries plus an explicit fixed
region stratum budget; equal regions within each time, equal active quantities
within each realization, equal realizations in the optimizer batch. The existing
initial/early/uniform/final physical-time mixture is not uniform space–time MSE.
Validation field metrics use separate uniform spatial/physical-time quadrature.
See `field-reconstruction-losses.md` for detached signed periodic detection,
minimum area/contrast, no-region fallback and normalized-versus-physical metrics.

A1 = selected normalized decoded field loss + .1 scalar reconstruction MSE
+ existing latent regularization (default1e-5). Both encoder and decoder receive
field-loss gradients. A2 = the SAME selected decoded field loss through T +
latent_weight (default .1) paired-state alignment + composition_weight (default
.01) latent/decoded autonomous composition consistency + optional physics.
Frozen A1 is unchanged in A2. Composition uses the complete reusable state and
is experimental, not evidence of physical correctness. Paired trajectory
supervision is always retained. Initial identity alone cannot pass the gate.

B baseline is the existing masked/group-normalized straight-path velocity loss:

```
z_s = (1-s) epsilon + s z_target        epsilon ~ N(0,I)
v_target = z_target - epsilon
L_FM = equal-weight active-group means (with recorded group weights)
endpoint_standardized = z_s + (1-s) v_theta(z_s,s,conditioning)
endpoint_physical_latent = endpoint_standardized * latent_scale + latent_mean
L_B = L_FM + beta_eff L_observation(endpoint_physical_latent)
```

Physical time t is distinct from flow time s; wave physical velocity v is distinct
from generative flow velocity. Calibration is inverted exactly once. Known
scalar slots and absent heat-v groups remain excluded from FM. The endpoint
estimate is not an integrated posterior sample.

`joint.observation_auxiliary` defaults to `{}` => beta=0. This performs no sensor
decode, random selection or reference calculation, preserving the baseline path.
Legacy `joint.auxiliary` is retained for compatibility; its old floored penalty is
NOT the new Gaussian likelihood. Do not enable legacy auxiliaries alongside the
new objective when doing a one-factor ablation. Legacy auxiliaries retain their
old serial implementation; the new endpoint objective retains actual mixed-
family B neural microbatches and weighted accumulation, including remainders.

New endpoint settings (all persisted/resume-checked):

| Setting | Library default | Explicit notebook experimental choice |
|---|---:|---:|
| beta | 0 | .01 when `FLOW_OBJECTIVE="observation-aligned"` |
| minimum_s / maximum_s | .5 /1 | .5 /1 |
| sensor_budget / query_chunk |8 /8 |8 /8 |
| selection / reduction | uniform / mean | uniform / mean |
| noiseless_policy | tolerance | tolerance |
| tolerance_absolute / relative | .001 /.05 | .001 /.05 |
| discrepancy_absolute / relative |0 /0 |0 /.05 |
| gaussian_constant | false | false |
| warmup_updates |0 |100 |

For quantity q, let `a_q` be its global training-fitted normalization scale.
Packed values/sigmas are normalized; prediction and reference use the same scale.
The physical residual is `(prediction_normalized-y_normalized)*a_q`; the mean
cancels. Supplied physical sigma is `sigma_normalized*a_q`. For positive sigma,

```
variance = sigma_physical² + (discrepancy_absolute + discrepancy_relative*a_q)²
penalty = .5 residual_physical² / variance
         + optional .5 log(2*pi*variance)
```

No implicit floor is applied to small positive noise. Underflow/nonfinite values
raise errors; choose an explicit discrepancy policy if needed. Discrepancy is
an additional modeling assumption, NOT supplied measurement noise. Default0
preserves independent supplied Gaussian noise. Correlated covariance is rejected
as an unknown configuration/noise specification. Zero-noise sensors instead use
`.5 residual²/(tolerance_absolute+tolerance_relative*a_q)²`, or reject explicitly.
This is a finite approximate consistency penalty, not exact constraint enforcement.
External observation rows reject duplicate scalar x/y/t/quantity identities
(including conflicts). Internal selected sensors are also checked. Metadata
validation/selection is explicitly CPU on every backend (MPS lacks unique_dim);
the differentiable prediction/loss/reductions remain on the selected device.
Masked NaNs/padded quantities are inert. No observations => zero auxiliary.

Uniform selection without replacement retains at most sensor_budget per case.
The mean estimates the case's full sensor mean; `reduction="sum"` multiplies by
its total valid sensor count (N/K correction). Cases are still equally weighted.
No selection of high-error sensors, no hidden reference inputs. Independent
example seeds keep selection invariant to microbatch partition and preserve the
base conditioning/query RNG stream. `selection="all"` rejects if the declared
budget is too small. Activation outside the declared s interval contributes0,
without renormalizing by the eligible fraction. beta_eff is beta times
min(1,(successful_update+1)/warmup_updates), or beta without warmup.
The frozen T/D path is differentiated to the endpoint; gradients reach B's flow
and observation encoder. Frozen AE/T/graph tensors do not update during B.
Decoded queries use exact x/y/t/quantity, including unknown endpoint-derived
parameters and authoritative known scalars, in bounded cross-case/time chunks.
There is no new full reference field in B. Conditioning observations remain fresh.

## Optional A2 strong residuals

`evolution.physics={"weight":0}` is the default. An explicit positive weight adds
per-case uniform domain/time collocation, independent of paired-supervision RNG.
Default query_points=4 is bounded1..128. This is an added cost, not replacement
supervision. Strong derivatives run FP32 with the existing explicit math attention
path (fused SDPA higher derivatives are unsupported). No architectural weights
are added. Both temporal derivatives through T and spatial derivatives through D
are retained. For domain period Lx,Ly and horizon T:

```
heat:    (u_t - D lap(u)) / (a_u/T)
wave_u:  (u_t - v)        / (a_u/T)
wave_v:  (v_t - c²lap(u)) / (a_v/T)
```

Square/mean each residual over uniform independent collocation points. Heat uses
heat_weight (default1); wave averages wave_u_weight and wave_v_weight terms
(default1 each). Cases are averaged equally. Each active residual is logged
separately as physics_heat/physics_wave_u/physics_wave_v. Normalization means/scales
are inverted before differentiating. A2 supervised coefficients are known; the
shared decoder helper also supports unknown endpoint-derived coefficients.
The public particle residual report now uses independent differentiable coordinate
copies per particle, fixing erroneous summation across a shared coordinate leaf.
Heat mean/dissipation and wave kinematics, mean-v, linear mean-u and energy remain
diagnostics, not enforced conservation. Mean initial v need not be zero.

## Optional phase-sensitive spectral field loss

Select `field_loss.type="spectral_mse"` with explicit spectral_weight>0
(e.g. .1), spectral_resolution=16 (8..64), spectral_frequency_power=0 (0..4).
Other field objectives require spectral_weight=0. The base is domain MSE; this is
not an additional RB/spectral combination. A1 and A2 add one bounded structured
periodic endpoint-excluded N² grid per supervised quantity/time. Normalized
prediction and reference are compared on exactly that grid, with extra query cost
reported. No FFT is applied to scattered queries. No gradient/Sobolev loss is added.

The real separable DFT computes complex coefficients with forward normalization
1/N² on every backend. Sum `w_k |F(pred-ref)_k|²`, where
`w=(1+kx²+ky²)^(power/2)` normalized to mean1 over discrete mode indices. These
are dimensionless mode-index weights, not a physical isotropic Sobolev norm on a
non-square box. With power0 Parseval gives structured-grid MSE, retaining phase;
a phase-shifted mode has nonzero loss even with identical magnitude spectra.
The reference is detached. This same real arithmetic avoids unsupported complex
MPS FFTs. The UI exposes the extra grid cost only for this explicit option.

## Recovery contract and artifact inventory

Train on local fast disk. Before execution, probe configured storage with a
non-destructive write/read/delete, record resolved config, archive exact installable
source, and publish the source/config snapshot. `Session.new` creates a unique run
ID separate from display_name. `Session.from_store` restores recorded science;
it never substitutes a current preset, silently recreates missing pretrained
modules, or overwrites a recorded source. A source mismatch fails with reload
instructions. Local conflicts fail before restore writes any new files.

At the first successful update and configured cadence (default100), save local
resume.pt atomically, then synchronously publish a safe immutable snapshot.
An optional minimum save time explicitly throttles ordinary events; first/stage/
final barriers bypass it. Existing checkpoint events also remain save opportunities.
No queue, concurrent checkpoint mutation or per-update Drive data I/O is introduced.
The default stop-on-save-failure policy uses at most3 attempts (2 retries), logs
traceback/status visibly, preserves the last verified snapshot and stops training.
Explicit continue is available and warns of data-loss risk. Saving status includes
stage/local checkpoint in session.json and latest verified snapshot/age/bytes/error
in sync-status.json. Notebook display recomputes age; a save's age0 is not timeless.

Persistent run layout:

```
<DRIVE_RUNS_ROOT>/<unique-run-id>/
  .pdeflow-store.json                 owned store identity
  objects/<sha256>                    shared immutable file contents
  snapshots/<id>.json + <id>.commit    complete manifest + body-hash marker
  latest.json                         advisory pointer, published LAST
```

Upload to a temporary object, check size/hash and unchanged local source, publish
and read back object; then write/verify manifest and completion marker. Verify
EVERY referenced file for THIS snapshot before updating latest. No Drive atomic
rename claim. Recovery scans complete manifests if latest is stale/missing;
partial/corrupt uploads are ignored. Manifests record sequence, parent, stage,
successful update, config/source identity, files/hashes/bytes and completion time.
The last N complete snapshots plus tagged best/final are retained. Content objects
are removed only when no retained/remaining manifest references them. Cleanup is
scoped to this explicitly owned run, never other user runs. There is currently
no automatic best-validation checkpoint selection; a tag does not invent one.
Frozen dependencies deduplicate by SHA. Caches and datasets are not duplicated;
procedural generation, split IDs, normalization-fitted IDs and cursors are saved.

`session.finalize()` does no learning. It exports the latest valid B state when
available, creates hash manifests under `bundles/`, strictly loads matching
modules on CPU, verifies source/dependencies/counters, then publishes a pinned
final snapshot. Repeated calls reuse an unchanged inference export and manifests.
An A1-only run is explicitly partial, with no generative inference artifact.
The final result reports complete_bundle_persisted and final_bundle_snapshot; an
explicit continue-on-failure policy never labels new unsaved bundles as persistently
verified. Inspect this result even when a previous verified snapshot still exists.

| View | Contents surviving a verified snapshot |
|---|---|
| inference | Standalone observation encoder/flow/T/D, typed scalar, graph, transforms, normalization/calibration, capabilities, source/config/gate provenance; full-state E and optimizers intentionally absent |
| diagnostic | Exact matching full E plus A1/A2/B canonical checkpoints, T/D/B, normalization/calibration, state schemas, dependency hashes, gate/validation/history/configs, reproducible procedural validation IDs, exact source ZIP |
| resumable | Diagnostic contents plus each available stage optimizer, scheduler, GradScaler, local torch sampling RNG/cursor, Python/NumPy/Torch CPU/CUDA/MPS RNG where applicable, successful/attempted/skipped/example counts, procedural dataset/split metadata, source/config identity |

The last two are views over the same canonical files, not duplicated weight ZIPs.
Optional `download_archive(root,kind)` makes a verified ZIP keyed by manifest hash
under downloads/ and reuses it. These download ZIPs are excluded from routine
snapshots. Downloading only the inference view again would omit E: retain the
Drive store or diagnostic/resumable download. A complete snapshot depends on
its objects, so copy the whole store rather than one manifest alone.

Functional resume is tested, including strict CPU exact replay after interruption.
Cross-device, library/kernel or FP16 replay is not promised bitwise identical.
Older checkpoints without ambient RNG can still load for inference/legacy resume,
but are rejected as a newly complete resumable bundle. Changed objectives/source
remain incompatible with exact resume. Use recorded source/config for the same
experiment; larger budgets require explicit extend and retain saved scheduler;
starting from weights is a separate new experiment, never an inference import.
Batch changes require explicit continuation and record old/new examples/updates.
New-run keep_examples conversion rejects warmup>=new budget unless the explicit
rescale_warmup policy is selected; e.g.1000 warmup/10000 updates →31/313.

## Exact next Colab run

Open `notebooks/colab.ipynb` from the new source package, not the old uploaded
notebook. Restart the runtime before changing source; old imported modules are
not reloaded silently. Upload ZIP in step1, optionally paste its handoff SHA256.
Setup verifies every source member and cached extraction, preserves compatible
PyTorch/CUDA rather than replacing it, and prints scientific source identity.

1. Step2: enable Drive, mount, choose DRIVE_RUNS_ROOT and local fast root. New run,
   save100 successful updates, minimum seconds0, last3, failure policy stop.
2. Step3: Medium, joint heat+wave, CUDA auto (native FP16+GradScaler on T4), Standard MSE
   (or an explicit region-balanced alpha.5 experiment).
   Baseline control uses FLOW_OBJECTIVE=baseline. Its paired treatment changes ONLY
   FLOW_OBJECTIVE=observation-aligned (the explicit table above). A2 physics remains0,
   no spectral term. Keep the printed unique RUN_ID.
3. Step4 optional ten-update preflight (two warmups separately); benchmarks are
   disposable. Step5 apply a reviewed batch result explicitly, inspect all budgets,
   actual family/state weights, losses, schedules, examples, device and save policy.
   Profile again if application invalidated it. Do not assume timing is accuracy.
4. Step6 enable training. For the first learning check use representation stage,
   inspect held-out field/constant-baseline quality, then progress only if appropriate.
   Failed gate remains a stop, never auto-overridden.
5. Step7 Save and verify complete run, including when partial/interrupted. Confirm
   verified snapshot ID/Drive destination and diagnostic/resumable availability.
   Download inference for the Mac, optionally diagnostic/resumable ZIP for backup.
6. Fresh runtime: same source in step1; mount in step2, RUN_MODE=resume, saved RUN_ID,
   optional SNAPSHOT_ID. Step3 restores source/config/dependencies/counters; step5
   reports recorded settings. Step6 resumes. Empty local root for conflicts.

To recover the exact source when only the Drive store remains: inspect a complete
snapshot JSON, find files['source/source.zip'].sha256, copy that object to a local
source.zip, verify its SHA and source manifest, then restart and upload it at step1.
Completion marker digest and object hashes must verify; the snapshot store helper
can perform that check from an installed compatible package. Do not train against
a mismatching source merely to get past this check.

Actual Colab/Drive/T4 acceptance for this revision (not run locally): perform a
pipeline-check representation run with save1; observe the first verified E/optimizer
snapshot; Stop; finalize; disconnect/reset runtime; reload same ZIP, mount and restore
RUN_ID into a fresh local root; inspect counts/scaler/source and resume. Then run
bounded preflight for the real requested Medium point counts/precision. Inspect
finite scaled/unscaled gradients, successful vs skipped updates, auxiliary timing,
peak CUDA memory and save status before committing a larger budget. Pipeline-check
quality is not a reason to bypass the gate or call its models accurate.

## Bounded ablations and evaluation

Use identical capacity, splits/seeds, normalization, observations, physical-time
policy, effective batch/optimizer/schedule and frozen A1/A2 hashes. Compare B FM
against B FM+endpoint likelihood; do not add A2 physics or change field loss in
that comparison. Inspect the failed representation gate first. A sensible *future*
first learning experiment is A1 XSmall 2000 updates, warmup100, held-out quality
review, then A2 2000 updates with paired supervision and the same review. Only if
both are adequate, compare B controls at 1000 updates/warmup100 using 3 fixed
seeds and a fixed held-out validation suite. This is a bounded development study,
not a sufficient accuracy budget; profile costs first. No such learning run ran
in this implementation task. Medium's missing original E/A2 still prevents its
matching oracle evaluation; no unrelated encoder was substituted.

Example configuration preparation, NOT execution:

```python
from dataclasses import replace
from pdeflow.colab import training_preset
from pdeflow.runtime import write_json
base = training_preset('xsmall','cuda','auto')
base = replace(base, ae_steps=2000,evolution_steps=2000,flow_steps=1000,
    training_options={**base.training_options,
        'autoencoder':{'warmup':100},'evolution':{'warmup':100},'flow':{'warmup':100}}).validate()
write_json('baseline-config.json',base.to_dict())
aligned = replace(base,joint={**base.joint,'observation_auxiliary':{
    'beta':.01,'minimum_s':.5,'sensor_budget':8,'discrepancy_relative':.05,
    'tolerance_absolute':.001,'tolerance_relative':.05,'warmup_updates':100}}).validate()
write_json('observation-aligned-config.json',aligned.to_dict())
# Separate A2 experiment: same representation, physics weight0 versus1e-4,
# query_points4; keep B and all field objectives fixed.
# Separate field comparison: MSE, RB alpha.5, or spectral_mse weight.1/grid16.
```

Compute is NOT equal merely because successful-update counts match. Endpoint
queries, physics higher derivatives and structured N² queries add measured cost.
`python scripts/observation_auxiliary_benchmark.py --device cpu --output <new.json>`
measures same-fixture B forward/backward overhead only (one full fixture batch,
all auxiliary endpoints active at s=.75) (not reference preparation,
optimizer, validation or persistence). `--config <recorded.json> --device cuda`
uses that architecture/batch on CUDA if memory permits, with explicit random
latent fixtures rather than claimed numerical reconstructions. Actual training
preflight is the end-to-end timing source; do not extrapolate tiny CPU ratios to T4.

Decide with held-out conditioning-sensor agreement AND independent held-out-sensor
RMSE/likelihood, initial/early/per-time u/v domain/RB/region/background metrics,
parameter error, periodic residual/invariant diagnostics, spectral detail/phase,
and equal-cost timing. Evaluate diverse fixed initial fields, signed features,
zero modes, known/unknown coefficients and noise/count shifts. Report ensembles'
spread, empirical coverage at declared levels and proper scores over independent
noisy observations; improvement in sensor fit alone is not calibration. Check
collapse/diversity loss when restricting s or increasing beta. Do not select
budgets using untouched test performance. Report no accuracy improvement before
these comparisons; original smoke models remain poor and the historical
conventional Fourier inverse remains superior in its recorded experiment.
