# Continuous physical time and sensor placement

The normal heat2D playground and new local/Colab coordinate-model presets now
use the **dimensionless physical interval [0,1]** with unchanged diffusivity D.
This is more diffusion than evolving to0.15 with the same D. It is not a relabelled
old trajectory. For a pure time change t'=t/T, the same trajectory requires
D'=T D. The app makes no such automatic rescaling.

## Use the playground

From the repository with the existing Python environment and supported Node on PATH:

```bash
source .venv/bin/activate
python scripts/dev.py
```

Open http://127.0.0.1:5173 (API http://127.0.0.1:8000). Alternatively run
`pdeflow serve` and, in a second terminal, `cd frontend` then
`npm run dev -- --port 5173 --strictPort`.

1. Choose a field or paint it. Default **Time interval: 0 to 1** and D0.07 are
   physical choices. Older saved browser preferences retain their actual end
   time; enter1 explicitly if those preferences still show0.15.
2. Play, pause or scrub to e.g. **0.237**. The paused view evaluates that physical
   time using Fourier evolution. A continuous model decodes its **same generated
   latent** at that time; it does not resample a new reconstruction.
3. Choose **Random**, **Evenly spaced**, **Heat-focused** or **Click points**.
   Purple hollow markers preview candidates; white dots are committed observations.
   Set the scalar count and absolute noise sigma, then **Take measurements**.
4. **Heat-focused** balances retained hot/cold regions and reserves background
   coverage. It uses the reference field to choose positions. **Focus spread**
   controls how broad those regions are; exact background/settings are in Advanced.
5. **Click points** toggles a nearby sensor when clicked again; **Clear sensor
   points** clears them. Enter/remove points gives a keyboard-accessible alternative.
   The scalar count becomes the number placed. Drawing and sensor tools are separate.
6. Import a genuinely continuous checkpoint, enable experimental use for a tiny
   model, and **Run model**. No checkpoint is needed for simulation or sampling.

Changing time never rewrites committed observations. Changing only noise keeps
locations/time fixed unless you explicitly resample or change the placement.
The resample button changes the seed; deterministic even/manual designs do not
need it. Generate is explicit; changing placement only previews a new design.
Late responses are ignored/aborted when the field/time/design changes.

**Advanced & details** contains display/export frame count, measurement seed,
background fraction, chosen times and fixed/adaptive timing, provenance, observed
value entry and JSON import/export. Measurements-only mode accepts actual values
without truth; heat focusing is unavailable there. JSON accepts either the older
row array or a full exported record with `rows` and placement provenance. Invalid
finite/bounds/sigma values are rejected, not silently snapped to frames.

For multiple times, N always means **total scalar observations**. Fixed sensors
keep spatial locations designed at the current displayed time; chosen times repeat
for each sensor, with a partial last cycle if N is not divisible by the number of
times. Across-times supports scattered times (leave Chosen physical times blank) or explicit chosen times; optional
adaptive placement redesigns at each explicit chosen time. The UI and record
separate spatial sensors, sampled times and total scalars. Fixed+adaptive is rejected.

## Training contract and migration

New main heat2D presets explicitly set:

```json
{
  "temporal_mode": "continuous",
  "time_interval": [0, 1],
  "data_mode": "procedural",
  "animation_frames": 31,
  "observation_placement": {"random": 0.5, "even": 0.25, "heat": 0.25}
}
```

`time_sampling` configures encoder/query/observation policies independently.
Defaults below are fractions of **scalar space-time points**, not a spatial grid
multiplied by a frame count:

| Role | Atom at0 | Early positive | Uniform over interval | Atom at final time |
|---|---|---|---|---|
| Fixed reference encoder | .35 | .35 | .25 | .05 |
| Fresh reconstruction supervision | .25 | .35 | .35 | .05 |
| Conditioning observations | .15 | .35 | .45 | .05 |

Early positive t/T is log-uniform in `[1e-4,.2]`: draw r uniformly in[0,1),
then t=T exp(log(1e-4)+r log(.2/1e-4)). Its density is finite on that bounded
positive interval; zero is a separate atom. Probabilities must be nonnegative,
sum to1 and retain initial/full-interval support. For budgets >=8, category
counts use deterministic largest-remainder allocation and random point order;
within categories coordinates/times are random. Smaller observation sets use
categorical draws. Thus finite query budgets deliberately optimize the rounded
mixture mean MSE, **not uniform space-time MSE**. No importance correction is claimed.

Encoder coordinates/values are fixed by solution seed, split, index and role.
They contain continuously distributed times but remain identical whenever that
underlying solution is encoded, including latent moment calibration and frozen
flow-target creation. Query times/locations and observation variants refresh each
update using the checkpointed generator. Flow keeps the AE frozen. Inference
never receives the complete encoding, focus field, region centres or clean targets.

Each observation example chooses a configured spatial policy; half use one
sampled physical time and half use scattered times from the observation policy.
Heat-focused training designs spatial locations from a sampled design-time field,
recorded by this generator contract, then evaluates the actual sensor times.
Random/even policies also approximate arbitrary manual layouts; there is no
accuracy guarantee for every manually selected pattern.

A legacy config/checkpoint with no temporal mode remains **snapshot-trained**
using its original `times`. Floating-point decoder acceptance does not establish
continuous training or temporal interpolation quality. Its original domain,
interval, weights and legacy inference remain available; the normal playground
requires continuous support. The API independently rejects a legacy override
when `require_continuous=true`. A legacy0–0.15 checkpoint cannot cover0–1.
Create a **new run** to migrate; changing mode, interval, time/placement sampling
or encoder budget cannot silently resume an incompatible checkpoint.

Continuous identity/resume ignores `animation_frames` and legacy `times`; display
frames do not determine support or create a different training dataset. Meaningful
scientific source changes remain checked. Three capabilities are separate:
syntactic coordinate acceptance, declared continuous training support, and
**validated temporal accuracy** (currently false for all new tiny checks).
Existing1D/3D and fixed-grid models remain in the legacy workflow.

## References, precision and display

`heat_continuous.FourierHeat` evolves the finite nodal Fourier representation:
uhat_k(t)=uhat_k(0) exp(-D |k_phys|² t), with k_phys=2pi*n/L per axis. Constant
and negative modes, origins and lengths are retained; taking the real sum fixes
the even-grid Nyquist convention for off-node evaluation. No coarse-frame time
interpolation supplies scientific reference values. It is not the exact unknown
continuous field beneath an arbitrary drawing.

Reference values are CPUfloat64. Sparse Fourier sums use128-point chunks and
omit nonconstant coefficients <=1e-13 with an explicit available absolute
omitted-magnitude bound. The constant mode is always preserved. Original-node frames use FFT evolution with all coefficients. Training
inputs/model outputs use float32 (optional GPU autocast remains as documented).
Spatial coordinates are normalized for embedding, and physical time by the
actual interval; autograd therefore carries the domain/time chain factors into
physical derivatives. D remains the physical known diffusivity.

Time endpoint tolerance is1e-7; only roundoff within that band is canonicalized
at an endpoint. Materially out-of-range/nonfinite times fail. Observation positions
exclude duplicated periodic upper endpoints; queries may include their periodic
aliases. Query chunks preserve order across both space and time.

Playback uses linear interpolation of a bounded cached display sequence. Paused
scrubbing displays an evaluation-pending message until the requested Fourier/
fixed-latent decode returns. Display interpolation is not a solver accuracy claim.
Changing frame count does not retrain, change committed times or alter the PDE.
The model's generated latent is saved in each successful result and reused on the
recorded inference backend for subsequent physical-time decoding; old results without that saved state need a
new inference run. Saved historical scientific files are not modified.

## Placement policy details

Random uses continuous uniform positions. Even coverage uses balanced rows whose
heights are proportional to row counts, with cell centres in each row: exactly N,
spatial centroid at the domain centre, no duplicated periodic endpoints. It is
approximately evenly spaced when N has no suitable rectangular factorization.

Heat focus analyzes a bounded32² view at the design time, separately from the
64² reference used for measurement values. It smooths signed contrast u-mean(u)
at .025 times the minimum physical domain length; detects local magnitude extrema,
merges connected periodic plateaus, suppresses nearby centres within .18 min-length,
and retains at most8 regions above .12 of maximum contrast. Contrast <=1e-8 times max(1,max|u|) falls back to uniform coverage. Feature analysis is cached by reference
identity/time; sampling spread/background/seed changes reuse the same detection.

Default background fraction is.25 (ceil(alpha*N), at least1); remaining counts
are allocated equally across retained regions with seeded remainder order. When
possible every retained region receives a focused anchor within .707 of the kernel
width. Other points use wrapped Gaussian jitter with width .08 min-length by
default, never below .04. Excess regions are covered by a seeded balanced subset
and the limitation is reported. Focused points try minimum separation .005
min-length against already placed points for12 attempts, then explicitly report
relaxation. Uniform background is not repelled. This is not a guarantee to detect
every physical source, optimal design or blind active sensing. Positions themselves
can convey reference information even though no scoring field enters inference.

## Training, validation and comparison

Use the updated Colab notebook and [Colab guide](colab-training.md). Large remains
width512 / tokens128 / layers8 / heads8, 65,536 training examples and100,000 updates
per stage; XL capacity/budgets remain selectable. Current presets are
`configs/coordinate/colab-*.json`; a local tiny CPU example is
`configs/coordinate/heat-continuous-smoke.json`. References retain initial grids,
coefficients and fixed encoder points in a bounded cache, not a full dense
animation tensor. Training normalization is fitted only to recorded training
**initial fields**, with one global transform shared across time.

For a bounded CLI execution check (fresh output directories):

```bash
pdeflow coordinate-train --config configs/coordinate/heat-continuous-smoke.json \
  --stage autoencoder --output outputs/my-continuous/autoencoder
pdeflow coordinate-train --config configs/coordinate/heat-continuous-smoke.json \
  --stage flow --autoencoder outputs/my-continuous/autoencoder/autoencoder.pt \
  --output outputs/my-continuous/flow
python -m pdeflow.coordinate.temporal_validation \
  --checkpoint outputs/my-continuous/flow/coordinate.pt \
  --output outputs/my-continuous/placement-comparison.json \
  --device cpu --subset 6 --count 25 --noise .01 --seeds 91 92 93
```

The comparison uses the same held-out validation solutions, N, measurement time
0.237 of the interval, noise realization policy, model/integration/ensemble settings
and multiple seeds for all three policies. Whole-domain individual and ensemble
errors are separate. Evaluation needs the run's data recipe/AE dependency; the
inference export alone needs neither. It does not choose a winner from test data.

AE and flow validation report t=0, early(0,.1], intermediate(.1,.6], late(.6,1),
and t=1 errors separately; a uniform11-time trapezoidal spatial-mean MSE is reported
alongside absolute, mean-subtracted and guarded-relative errors. AE additionally
reports the declared training-weighted objective and initial/spectral distortion.
The representation gate uses the **uniform** metric against a training-fitted
constant, not a claim of acceptable early-time quality. Off-frame times include
.001,.01,.037 and.237; report files declare exact subsets/query shapes. Test stays
untouched unless explicitly requested for final evaluation.

See [verification and screenshots](continuous-heat-verification.md). Only tiny CPU and actual MPS training are verified; CUDA/actual Colab
execution and useful temporal accuracy remain unverified. Next: a bounded representation-quality study stratified by time and
field family before substantial conditional-flow training.
