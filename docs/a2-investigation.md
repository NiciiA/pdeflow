# A2 investigation and safe diagnostics

The reported live run `20261101T122421-57f94d11b0a` (planned A2 2500) is not present
in this checkout. Its screenshots/run artifacts were not supplied as accessible
files here. Its actual objective, LR history, weight changes and accuracy therefore
remain unconfirmed. Variable stochastic batch loss alone does not establish a
plateau. The source investigation confirmed reporting bugs and separately tested
whether the implemented evolution can learn nontrivial dynamics.

## Confirmed corrections

* A2 notebook previously selected frozen `representation-validation.npz` and its
  first physical frame t=0. Now A2 publishes current learned T/E/D predictions at
  nonzero .237 times the interval, on fixed held-out heat u, wave u and wave v
  examples. Captions contain stage/family/quantity/time/example/weight update/preview
  update/RMSE. Reference and prediction share limits. A missing or mismatched A2
  preview is reported unavailable; there is no fallback to an A1 image.
* The live history exposed only the last200 rows. Validation could disappear by
  update1300 if last evaluated at1000, and a lone point had no marker. Full local
  history merges the durable checkpoint history, append-only journal and live tail;
  moving average is labelled, and even one validation point has a visible marker.
* “Fixed” validation previously sampled fresh queries from the current training RNG
  and restored it afterward. Now a separate fixed RNG/phase schedule repeats actual
  examples/times/queries. Validation and preview restore RNG and every module mode.
  The objective remains training-weighted, not uniform physical space–time accuracy.
  Historical rows without the new label are marked legacy resampled validation.
* Component logs previously reflected a last case/microbatch. New detached per-case
  reductions reflect the differentiated domain/spectral/alignment/composition/
  physics sum. Actual optimizer ownership, sampled unscaled/clipped gradients and
  representable parameter changes are measured at bounded checkpoint cadence.

A2 uses raw frozen complete-state encoding, trainable graph/evolution, frozen but
DIFFERENTIABLE decoder. Optimizer owns exactly trainable graph/evolution tensors.
No detached decoder output or stale prediction cache was found. Frozen E/D/scalar
weights remain unchanged while evolution parameters change. Known D and wave c
routing, c² residual, global per-quantity normalization and exact identity checks
pass. Continuous A2 targets were already implemented; the new B measurement-time
policy is a separate feature (measurement-times.md).

## Bounded scientific findings

Evidence: `outputs/a2-investigation-20261101/`. The before/after control used identical
configuration and LR1e-30 (no representable weight updates). Training losses were
identical `[10.325526,7.300939]` and max weight difference0. Old “fixed” validation
changed `7.202840→6.289010` with unchanged weights; corrected validation repeated
`7.609698→7.609698`. This isolates the reporting error without changing training.

With fixed nontrivial heat/wave trajectories, 120 production-sampled tiny A1 updates
then200 fixed-case A2 updates reduced train normalized field MSE .805699→.713671.
Its held-initial baseline was .721978: learned/held ratio1.11596→.98849. Poor frozen
representation remained a major limitation; wave v was nearly identity on held-out
cases. An explicitly named control fitting A1 on the same two cases for200 updates,
then A2 for200, reduced .692499→.465874; held .566268, ratio1.22292→.82271.
Independent validation ratio .97717 is only a small aggregate gain, not sufficient
accuracy. At wave-v endpoint, learned RMSE .7166 was WORSE than held-initial .5322.
Initial wave-u/v RMSE .8323/.5287 remained poor. All are tiny diagnostic fixtures,
not evaluated pretrained models or representation-gate passes.

With spectral frequency power0, Parseval makes the phase-preserving normalized
complex FFT error equivalent to spatial grid MSE on that same endpoint-excluded
grid. It is not frequency-selective supervision. Domain/spectral gradient cosine
in the fixed-A1 control changed .99859→.10086; there is no evidence here of a
systematic gradient conflict. This does not diagnose the unavailable Colab run.
Physics probes differentiate actual decoded evolution, including time derivatives.

A2 default query mix has25% exact initial targets, whose identity can give zero
operator gradient when physics is off. This is declared historical behavior, not a
proven cause of the reported plateau. Optional NEW experiment:
`{"evolution":{"time_policy":"nonzero_conditioned_v1"}}` conditions the existing
mixture on nonzero target times, retains early/full coverage, leaves A1/validation
identity checks intact, and is incompatible with same-stage resume. It does not
silently alter an active run. Fix representation quality and inspect fixed physical
metrics before enlarging A2 or launching the next substantial run.

## Obtain the actual run's diagnostics without reloading its source

Download/upload `src/pdeflow/evolution_export.py` to Colab as `evolution_export.py`.
It is self-contained stdlib+torch orchestration and does not import/install new
PDEFlow scientific modules. The equivalent ready cell is docs/a2-current-run-export.py.
In the OLD kernel, after a safe pause:

```python
from pathlib import Path
namespace = {"__name__":"pdeflow_readonly_export"}
exec(compile(Path("evolution_export.py").read_text(), "evolution_export.py", "exec"), namespace)
report = namespace["export_a2_diagnostics"](
    session.root, Path("/content/a2-diagnostic-export.zip"),
    persistent_root=session.persistent)
print(report)
from google.colab import files
files.download(report["path"])
```

Use a fresh destination if it already exists. It copies current configs/history/
preview and a bounded CPU diagnostic weight export from the latest atomic A2
checkpoint, including MATCHING E/T/D and transforms. No optimizer mutation or
training. Live row counters can be ahead of the last safe checkpoint; the export
labels this. Persistent manifests are listed as candidates, not falsely marked
verified. Maximum copied weight state is512MiB; no dataset/full optimizer download.

On a compatible extracted source, inspect the exported checkpoint read-only:

```bash
python -m pdeflow.cli evolution-diagnose --checkpoint /path/to/a2-diagnostic.pt \
  --output outputs/a2-actual-diagnostic.json --device cpu --per-family 1 --resolution 16
```

Optional repeatable `--earlier /path/to/earlier.pt` compares the same split,
normalizations and hashed frozen A1. Model/reference mathematics must match the
recorded critical source files; mismatches are rejected. Outputs contain physical
MSE/RMSE/relativeL2 per quantity/time, initial held-constant baseline, reference-state
reencoding diagnostic (not a guaranteed error floor), invariants and strong residuals.
RelativeL2 floor is `(global training quantity scale*1e-6)^2` in mean-square units.
Physical aggregate uses trapezoidal time weights over declared fixed evaluation
times; per-time records are retained. No test-split tuning.

## Safe stop and source transition

In the OLD notebook use one Interrupt execution action (do not terminate runtime).
Wait for the engine to reach its safe boundary and session final sync to finish.
Then, with its ORIGINAL installed package:

```python
final = session.finalize()
print(final)
from pdeflow.persistence import latest, verify
snapshot = latest(session.persistent)
assert snapshot is not None
checked = verify(session.persistent, snapshot["id"])
assert checked["files"]["bundles/resumable.json"]["sha256"]
print("Verified persistent snapshot:", checked["id"], checked.get("progress"))
```

A successful verification hashes the manifest's persisted objects. If saving or
verification fails, preserve the local runtime and repair storage before restarting.
No Drive verification was executed from this Mac. Keep run/source/checkpoints and
snapshot IDs; download the diagnostic export before switching source. Resume the
old run only with source/source.zip from that exact verified snapshot. New source
and especially changed B observation distribution require a deliberately NEW
experiment; do not enable allow_source_change to evade compatibility checks.

The new source ZIP/notebook can be installed after restarting the kernel in a fresh
extraction directory. See measurement-times.md and SOURCE_PACKAGE.md. The
representation gate remains enforced; smoke models are not accurate. CPU and
bounded actual MPS execution were recorded during this A2 investigation; CUDA,
actual Colab FP16 and the unavailable current-run learning outcome remain untested.
