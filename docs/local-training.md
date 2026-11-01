# Reproducible MacBook training

Use the existing Python 3.12 environment. Float32 MPS training, endpoint/PDE
and observation gradients, save/resume and CPU reload were actually executed
at both new intended model/grid/time/batch layouts. These bounded checks used
12 instances (8/2/2), two updates, and two guided generation steps; they do not
measure trained reconstruction quality. CUDA and actual Colab are untested.

## Dataset and model profiles

| Config | Independent train/validation/test | Spatial grid | Saved τ | Storage | Model / batch | Max updates |
|---|---:|---|---|---|---|---:|
| cpu_2d (smoke) | 18 / 6 / 6 | 8² | 0,.05,.1,.15 | Original float64 NPZ | width 12, 2 blocks / 4 | 30 |
| mps_2d_pilot | 2000 / 200 / 200 | 32² | 0,.01,.03,.06,.1,.15 | float32 NPZ shards | width 32, 3 blocks / 16 | 5000 |
| mps_2d_standard | 10000 / 1000 / 1000 | 32² | same six times | float32 NPZ shards | width 32, 3 blocks / 16 | 20000 |
| mps_3d_pilot | 1000 / 100 / 100 | true 16³ | same six times | float32 NPZ shards | width 16, 2 blocks / 2 | 3000 |

All fields are scalar u, one channel. New profiles use exact Fourier heat
reference evolution on periodic unit boxes, known D uniform in [.03,.12],
Gaussian Fourier initial fields with spectral bandwidth uniform [.6,2.4],
amplitude multiplier uniform [.3,2], and offset uniform [-1,1]. Gaussian mode
coefficients independently vary in every spatial dimension. These are signed
heat fields, not nonnegative chemical concentrations. More masks/measurements
are not counted as independent instances. Larger counts are engineering targets,
not a guarantee of statistical adequacy or reasonable full-run duration.

Each complete instance has a deterministic 128-bit child seed and content/config
ID; a separate seeded permutation allocates complete instances before sensor
masks. generation.json and test_manifest.json are fixed before materialization.
Each completed shard gets an atomic checksum sidecar. Rerunning an interrupted
generate with unchanged config resumes completed shards, checks their hashes
and the held-out identity, and recomputes unfinished shards. Completed datasets
reject overwrite. Different configs require different directories. Held-out
manifests are immutable by contract and validated on load, without changing OS
permissions. File integrity is checked when a shard is first read.

Only one shard is generated at a time. The loader keeps at most two decompressed
shards and training materializes only a batch on MPS. Normalization uses a
streaming mean/variance over sorted training indices only, retaining original
training-ID provenance; validation/test members never fit transforms. Validation
is batched. The original tiny NPZ format is preserved and remains float64 with
its original sampling/split rules; use sharded_npz for configurable distributions,
splits and storage dtype. The lazy Python reader is synchronous (zero data-loader
workers); runtime CPU threads=2. Random batch access may incur repeated shard
I/O. No claim of optimized large-dataset throughput is made.

Adam uses constant learning rate .001, global gradient clipping at 10, float32
full precision, and validation/checkpoint every 100 updates. No scheduler,
precision scaler, gradient accumulation or data-loader multiprocessing is used;
scheduler/scaler states remain explicitly null. Fixed validation RNGs are
independent of training streams. Batch/source/generation and backend RNG states
are persisted. Exact CPU same-source/config resume is tested; cross-device
bitwise equality and deterministic MPS kernels are not promised.

## Measured estimates and preflights

Four-instance probes separate manifest-planning overhead, reference evolution
and compression. Actual larger datasets were not generated during this task.
Numbers below are decimal MB, excluding small manifests/sensors and runtime
memory. Projection includes measured planning plus extrapolated reference and
compression; disk writes, checksums and filesystem overhead are not measured.

| Profile | Raw bytes / MB | Estimated compressed MB | Reference ms/instance | Projected seconds |
|---|---:|---:|---:|---:|
| mps_2d_pilot | 58,982,400 / 58.98 | 54.19 | 1.08 | 4.84 |
| mps_2d_standard | 294,912,000 / 294.91 | 270.95 | .92 | 22.70 |
| mps_3d_pilot | 117,964,800 / 117.96 | 106.94 | 1.67 | 6.69 |

Earlier estimates included planning time in per-instance cost; the corrected
*-final.json probes separate those terms. These remain small-probe projections,
not actual full-dataset timing. Raw formula: N × 1 × T × product(grid) × 4.
At shard_size=32 the 2D/3D raw value buffers are about .79/3.15 MB respectively;
normalization uses temporary float64 tensors and caches/batches add memory.

Actual 2D MPS preflight: 6.18 s, process RSS high-water 421,183,488 bytes;
post-guidance MPS tensor allocation 247,040 bytes, driver allocation 96,403,456.
True 3D: 4.28 s, RSS 411,140,096; post-guidance tensor allocation 136,704,
driver allocation 36,765,696. MPS figures are point-in-time values after
intermediate tensors were freed, not peak training memory. CPU 2D preflight
also passed. Hardware: macOS 15.2 arm64, Python 3.12.3, PyTorch 2.5.1.
No silent CPU fallback occurred; CPU reference generation and CPU reload are
explicit stages. Out-of-memory errors require an explicitly revised config;
the code never silently changes physical grids or time windows.

## Exact launch, resume, discovery and reconstruction

From the repository root, use new output folders. The examples below have not
launched the full datasets or long training automatically.

```bash
source .venv/bin/activate
CFG2="configs/experiments/mps_2d_pilot.json"
RUN2="$PWD/outputs/local-mps-2d-pilot"
DATA2="$RUN2/data"
TRAIN2="$RUN2/training"

pdeflow hardware
pdeflow estimate-data --config "$CFG2" --samples 8 --output "$RUN2/estimate.json"
pdeflow preflight --config "$CFG2" --output "$RUN2/preflight"
pdeflow generate --config "$CFG2" --output "$DATA2"
# Optional existing-data validation and bounded training before the long job:
pdeflow preflight --config "$CFG2" --data "$DATA2" --output "$RUN2/preflight-existing-data"
pdeflow train --config "$CFG2" --data "$DATA2" --output "$TRAIN2"
```

Generation is explicitly CPU reference solving and does not require MPS. Training
fails clearly if MPS is unavailable. To check the workflow on CPU, pass
--device cpu to preflight/train. To choose the 12,000-instance standard profile,
set CFG2=configs/experiments/mps_2d_standard.json and use a distinct RUN2 before
executing the same commands. Do not reuse a dataset under another config.

Stop training with Ctrl-C. Resume the latest complete checkpoint (updates since
its last atomic write are lost). In a new shell first restore the variables
above to the existing run. To guarantee a first bounded checkpoint you may use
--stop-after 100 on the initial train invocation, then resume to the configured
maximum. Resume requires unchanged source, exact config, dataset/splits and
normalization; old-source exports remain usable for inference but resuming old
training code under changed source is rejected deliberately.

```bash
pdeflow train --config "$CFG2" --data "$DATA2" --output "$TRAIN2" \
  --resume "$TRAIN2/checkpoint.pt"
```

Every save writes checkpoint.pt and portable inference.pt; no separate export
command is required. History records actual training and fixed-RNG validation
CFM losses, and manifests record steps/time/RSS. Test error is not used for
checkpoint or hyperparameter selection. These commands do not establish an
accuracy improvement; evaluate held-out reconstruction and conventional baselines
separately with validation-frozen settings.

```bash
export PDEFLOW_MODEL_ROOTS="$TRAIN2"
# Node >=22.12/npm must be on PATH (Node 24.19.0 tested).
python scripts/dev.py
# Open http://127.0.0.1:5173. Insert Heat, choose the new checkpoint and
# review Use checkpoint layout; provide compatible observations and validate.
```

Restart the API when changing configured roots. Model discovery is automatic
within those roots; no upload or separate registration command is needed.
From another activated terminal with the same variables:

```bash
RECON2="$RUN2/reconstruction-$(date +%Y%m%d-%H%M%S)"
pdeflow reconstruct --config "$CFG2" --device mps \
  --checkpoint "$TRAIN2/inference.pt" --problem "$DATA2/test_problem.json" \
  --observations "$DATA2/test_observations.json" \
  --reference "$DATA2/test_reference.npz" --output "$RECON2"
pdeflow evaluate --result "$RECON2/observation_pde"
```

Omit --reference for standalone inference, or choose --device cpu for portable
reload. The three guidance comparisons and Fourier baseline use the same known
problem and measurements; references are loaded only afterward for evaluation.

True 3D uses its own dataset/config/output, with no grid reduction:

```bash
CFG3="configs/experiments/mps_3d_pilot.json"
RUN3="$PWD/outputs/local-mps-3d-pilot"
DATA3="$RUN3/data"
TRAIN3="$RUN3/training"
pdeflow estimate-data --config "$CFG3" --samples 8 --output "$RUN3/estimate.json"
pdeflow preflight --config "$CFG3" --output "$RUN3/preflight"
pdeflow generate --config "$CFG3" --output "$DATA3"
pdeflow train --config "$CFG3" --data "$DATA3" --output "$TRAIN3"
# After interruption:
pdeflow train --config "$CFG3" --data "$DATA3" --output "$TRAIN3" \
  --resume "$TRAIN3/checkpoint.pt"
export PDEFLOW_MODEL_ROOTS="$TRAIN2:$TRAIN3"
```

The notebook remains a launcher for the same package; new sharded profiles work
through generate/train without notebook numerical code. For Colab use an explicit
CPU/CUDA runtime override rather than requesting MPS there. No actual Colab or
CUDA run was performed. The next executable family is the complete conservative
2D/3D Darcy specialist in darcy.md, followed by substantive chemistry.

The historical pdeflow.experiments budget-study runner intentionally remains
limited to its small original NPZ profile and rejects sharded profiles before
generation. Use the documented generate/train/preflight path for larger data;
a memory-bounded multi-case large-profile benchmark runner is separate work.

## Second executable specialist: Darcy

The separate darcy_mps_2d/3d and darcy_colab_2d/3d profiles now configure real
Darcy data/training, rather than reusing heat physics. Their complete commands,
counts/layouts, resume rules and checkpoint registration are in
[multispecialist.md](multispecialist.md). The larger full runs were not launched.
Saved two-update MPS preflights verify32² and true16³ layouts; CUDA/actual Colab
remain untested. The estimate-data command remains a heat-reference compression
probe; it must not be used to estimate Darcy costs. Darcy CG timings and
iterations are in its generated shard sidecars and preflight manifests.

## Current online/hybrid and chemistry workflow

The prior sections record offline heat/Darcy workflows. New common physical
providers and the implemented chemistry specialist are documented in
[online-training.md](online-training.md), with exact --evaluation commands,
MPS/Colab profiles, resume, preflight and registration. Online/hybrid do not
require --data or full training-set generation. The Darcy inverse is now
implemented; its bounded study and limitations are in darcy-study.md.

## FK inference after training

FK uses the same exported independent-Gaussian linear-CFM model; no extra FK
training objective or checkpoint is required. Current exact-grid8²/8³ inference
profiles are configs/experiments/fk_*.json, with explicit MPSfloat32 variants.
Validate the full checkpoint/layout first; larger trained layouts need their own
matching public config plus inference.fk block. FK/importance select real
Euler–Maruyama stepping and weighted inference; the existing gradient/refinement
paths retain separate semantics. See fk-verification.md for exact local/MPS
commands, Colab transfer and current negative/limited scientific evidence.
CUDA/actual Colab FK execution is untested; training resume is not FK resume.

## Simplified ordinary operation — 25 October 2026

This guide retains explicit data/configuration research workflows. Everyday
online training now uses `pdeflow train --preset heat-2d --device mps`, then
`pdeflow resume latest` and `pdeflow evaluate --run latest`. The wrapper prepares
fixed held-outs and frozen training-only calibration and publishes exports
itself. No dataset generation or PDEFLOW_MODEL_ROOTS is required for ordinary
use. See everyday-usage.md for the distinct preset table and exact commands;
the older offline profile table above is not the everyday preset table.
Completed-run extension adds to the recorded total with --additional-steps;
constant Adam learning rate/state are preserved. Full new pilot budgets remain
unexecuted. Current backend/layout evidence is in everyday-verification.md.
