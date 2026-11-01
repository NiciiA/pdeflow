# Known physical measurement times

Implemented 1 November 2026 for the active periodic 2D heat/wave operator. This
changes NEW experiment conditioning distributions, not existing model accuracy.
An observation is `(x, y, physical_time, quantity, value, absolute_sigma)`. Time is
known conditioning. Reconstruction query time and generative flow time are
independent. A1 remains a complete initial-state encoder at t=0; A2 learns physical
evolution. No reference initial field enters inverse conditioning.

## What changed

Previously B already sampled continuous times, with a default mixture: 15% exact
initial, 35% log-uniform early, 45% whole-interval uniform, 5% exact final. Thus
training was not restricted to t=0. The playground default placed all observations
at the current display time. Training feature placement used a fixed t=.237 u map;
playground feature placement used the display-time map for other-time sensors.
Both were mismatched reference-assisted placement approximations. Measurement
values themselves were already evaluated at the sensor's actual time.

NEW runs resolve `CoordinateConfig.observation_time` to:

```json
{"policy":"continuous_uniform","stratified":true,"whole":1.0,"early":0.0,
 "initial":0.0,"early_min":0.0001,"early_max":0.2,"placement_groups":8}
```

For N valid observations, split the configured physical interval into N strata,
draw independently and uniformly inside each stratum, then randomly permute the
sensor times. Times are continuous floating values, not frame/bin centres. A
single sensor has one time and covers the interval across examples, not within
that example. Observation-count ranges, unconditional dropout and quantity IDs
retain their semantics. N is the TOTAL scalar observation count.

An optional explicit override is:

```python
{"observation_time": {"policy":"continuous_mixture", "whole":.65,
                      "early":.25, "initial":.1, "stratified":True}}
```

Component counts are multinomial (correct probabilities even for N=1), each
continuous component is jitter-stratified, then all times are shuffled. Early
fractions are log-uniform from early_min to early_max of the selected interval;
initial is exact t=0 and requires the selected interval to start at zero. Whole must be positive; probabilities sum to
one. No extra final-time atom. These are B policies; `time_sampling.observation`
is used only by the explicitly versioned historical policy.

## Values, spatial placement and cost

Shared `observation_times.py` serves local training, Colab and the playground.
SpectralState evaluates the requested quantity at each actual x/y/t, including
wave zero modes. Noise is independent absolute Gaussian sigma. Global training
quantity transforms normalize value/sigma; timestamps remain physical and the
coordinate encoder applies interval scaling. Network computation uses float32
coordinates; exported playground timestamps retain their original float values.

Random and even placement do not consult the reference. Feature placement keeps
background coverage (training .25; playground configurable .1–1). For each
quantity independently, partition the physical interval into at most 8 equal-width
temporal bins, then evaluate a bounded 32² placement map at the mean actual sensor
time of each occupied bin. Placement is an approximation; timestamps/measurement
values are NEVER rounded to that map time. At most 8 maps for heat, 16 for mixed
wave u/v per example. Quantity selection and bin maps stay reference-only. Masks
or full fields are not encoder inputs. Playground sampling.json records analysis
times, bin populations, spatial allocation and exact sensor-time extrema; training
records the policy, group bound and scientific source. Arbitrary reference fields
are not materialized as full space–time training arrays.

Registered spectral dynamic references support continuous evaluation. Imported
static fields and numerical trajectories without a declared validated temporal
evaluator are rejected here with an explanatory error; the playground no longer
silently re-solves an imported trajectory from its first frame. No interpolation
is currently promised for those imports. Use a supported spectral dynamic reference.

## Playground

Under **03 Take measurements → Measurement times**:

* **Random across the time interval**: edit start/end and total count, optionally
  expand Advanced temporal mixture, then explicitly click Take measurements.
* **Choose measurement times**: enter arbitrary distinct supported times, e.g.
  `.02, .2, .47, .83`. Blank per-time counts distribute the selected total cyclically
  (at most one extra per entered time); `1,2,3,1` explicitly resolves to 7 scalars.
* **Click points**: the visible Active measurement time initially takes the display
  time when entering placement. Each click stores that time. Changing display time
  afterward changes neither active placement time nor existing point timestamps.

Control changes affect future generation. Committed observations change only on
explicit Take measurements/import or edits that invalidate the reference itself.
The time summary shows count, min/max and an eight-bin histogram. Expand the sensor
inspector for exact coordinates, time, quantity, value and sigma. Markers within the
visible tolerance of display time are bright; others are faded or hidden using
Show other-time sensors faded. Fading means temporal distance only. Hover tooltips
show exact stored timestamps. Candidates are hollow, uncommitted spatial designs.

Advanced & details → Measurement details exports/imports JSON rows plus provenance.
Reimporting a saved record uses its rows verbatim, with no resampling. Missing times
are rejected. The historical API `pattern=current` is an explicit single-time mode
with required `time`; old patterns remain available for replay. New API modes are
`continuous`, `chosen`, and `timestamped_manual` (requires position_times).
Saved reconstruction requests and diagnostic cases include complete original rows.

## Full conditional path and diagnostics

Both u and wave v carry distinct quantity IDs. Physical x/y/t passes through
conditioning.prepare, masked ObservationEncoder coordinate features and the
conditional flow. Invalid padded sensors/timestamps are zeroed and excluded;
permuting observations preserves the set output. Known D/c disclosure is unchanged.
Endpoint likelihood calls `decode_points`: evolve generated initial latent separately
to each sensor time, decode its quantity at its coordinate, inverse-normalize the
residual. Frozen A2/decoder weights still propagate gradients to the generated
endpoint and flow. Latent velocity matching is unchanged. Endpoint likelihood is
optional; its recorded beta still controls activation.

Inference sensor predictions and metrics use exact sensor queries, independently
of display grids. metrics.json includes sensor_coordinates_and_times, quantity IDs,
per-sample predictions and the explicit evaluation policy. Diagnostic replay retains
rows and noise; no reference initial state is substituted. Timestamp support does
not mean an old checkpoint was trained with the new distribution or is accurate.

## Independent A2 target-time policy

A2 was already continuous. Its recorded default query mixture remains 25% exact
initial, 35% log-uniform early, 35% whole interval, 5% final. Uniform draws use 32
randomized strata across example phases, not frame centres. Targets evaluate the
same realization/parameter/quantity at actual ta/tb; odd phases may use continuous
ta∈[0,tb], even ones start at0. Exact identity T(t,t) is structural and tested at0;
identity cases can contribute zero evolution gradient. `nonzero_conditioned_v1` is
an explicit NEW A2 experiment (see a2-investigation.md), never an implicit change.
B sensor times and A2 target times are logged separately: b_observation_time_mean,
b_observation_exact_initial_fraction, eight B histogram fractions versus a2_ta,
a2_tb, a2_dt and A2 initial/early/intermediate/late fractions. Checkpoint/config
metadata records both policies. A2 reference targets are not animation snapshots.

## Existing runs and safe Colab transition

Engine6 requires recorded observation_time. Engine≤5 checkpoints without it resolve
to legacy_mixture_v1 and keep historical RNG/placement semantics. Strict resume
rejects a changed distribution. Source fingerprint validation remains enforced;
use the verified old snapshot's source to resume an old run. No automatic migration,
retraining, gate bypass or overwrite. New observation policy requires a NEW experiment.
Changing only UI evaluation settings never retrains a model.

Do not install/reload source in an active training kernel. Use one cooperative
interrupt, wait for its safe checkpoint/final sync, then run `session.finalize()`
and verify the complete persistent snapshot (instructions in a2-investigation.md).
Keep the old run and source snapshot. Restart the runtime, open the new notebook,
upload the new ZIP, choose RUN_MODE="new", a new display name/ID, and review
OBSERVATION_TIME plus the independent A2 settings. Leave training/benchmark flags
off until reviewed. Gate failure remains a stop. Actual CUDA/Colab execution of
this revision is untested.

## Evidence

`outputs/measurement-times-20261101/` contains focused CPU tests, notebook tests,
actual Chromium heat/random and wave-v/explicit-time inference, request/diagnostic
JSON, screenshots, manual/count/replay checks and packaging/preservation records.
Changing the display to .61 left the exact sampled rows unchanged and made no new
measurement requests. No substantial training or accuracy comparison was run.
See verification.md for exact checks and device limitations.

Manual values without a reference have a separate visible **Observed measurement
time** input in Measurement details. Adding a value stores that explicit time,
not the animation slider's current value. Wave imports accept u/v and reject
missing/nonfinite times. The real import/manual-value check is recorded in
browser-wave-import-final.log; browser-time-inspector.png shows the compact cards.

### Exact notebook reload

1. In the OLD runtime, interrupt cooperatively, wait for checkpoint/sync, run
   session.finalize and verify its complete persistent snapshot as shown above.
   Retain that snapshot ID, original source and diagnostic ZIP. Do not restart if
   verification failed.
2. Download the updated `outputs/measurement-times-20261101/colab.ipynb` and open
   that notebook in Colab. Restart the kernel before running its install cell.
3. In step1 leave REPOSITORY_URL and SOURCE_ARCHIVE blank, paste the hash from
   `pdeflow-continuous-measurement-times-source.zip.sha256` into
   EXPECTED_SOURCE_SHA256 and upload exactly the corresponding source ZIP when
   prompted. The cell creates a hash-specific extraction and verifies its manifest;
   an existing changed cache is rejected rather than overwritten.
4. Choose GPU/device/precision explicitly. For revised measurement-time training
   set RUN_MODE="new" and a new RUN_NAME. Step3 OBSERVATION_TIME defaults to
   continuous_uniform with stratification; leave A2 target settings independent.
   Inspect both printed policies. Keep RUN_TRAINING and benchmark flags false for
   configuration review. The representation gate remains authoritative.
5. For bounded Colab acceptance only, use PRESET="pipeline-check", allow the
   explicit CPU pipeline check if using CPU, and execute the existing guarded
   A1/A2/B pipeline procedure. Inspect config.observation_time, each stage's
   resolved-config and B/A2 separate logged time fractions. Two updates establish
   execution only and cannot validate distribution frequencies or model accuracy.
   Unit coverage/value tests supply those mathematical sampling checks. No CUDA
   result is inferred from this Mac's tests.
