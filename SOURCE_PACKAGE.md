# PDEFlow mesh-family source package

Extract and work inside pdeflow/. README.md and docs/mesh-family.md describe the
active new M1–M5 family. notebooks/colab.ipynb is the thin single-candidate/suite
workflow; notebooks/legacy-factorized-colab.ipynb preserves the preceding original
A1/A2/B notebook. The unrelated older colab-legacy.ipynb remains preserved in the
checkout and excluded under the original packaging policy.

Included: installable package source, frontend source and npm lockfile, project and
dependency metadata, configurations, tests, local launch/benchmark scripts, active
and retained factorized notebooks, documentation/attribution, .nvmrc and this file.
SOURCE_MANIFEST.json records every included member's bytes/SHA256; the adjacent
.sha256 records the ZIP. Verify CRC plus manifest, not just filename. No private
repository/account/Drive path is required; notebook paths are configurable.

Excluded: outputs, .git, .venv, node_modules, caches/builds, datasets/checkpoints,
secrets, unrelated personal files and user reference PDFs. Existing checkpoints,
scientific datasets/results and source archives stay in their original locations.
No pretrained model is bundled or automatically substituted. The deterministic
reference recipe generates TRAIN/validation/test data locally; no external dataset
download is required. Train through documented CLI/UI/Colab, or import your own
compatible standalone export. Source-only packages cannot resume a training run
without its separately verified resumable artifact/snapshot.

New models have explicit mesh architecture/schema/physical-state contracts; no A1
encoder exists. Inference exports include every actual conditioner, generator,
evolution and decoder used, transforms/capabilities/provenance, without needing data.
M1 is initial-only. Diagnostic/resumable bundles include source/config, dataset
identity/splits/scales, actual modules and strict prerequisite hashes, training state,
optimizer/scheduler/scaler/RNG/counters/history. Resume requires recorded scientific
source/config. Never rename legacy weight keys or load random new modules as a
compatible checkpoint. Frozen dependencies have canonical files and are deduplicated
by persistent object hashes.

On Colab stop/save old training first, start a fresh runtime, open the updated
notebook and upload this ZIP in step1. Optional EXPECTED_SOURCE_SHA256 comes from
its adjacent hash. Existing source caches/snapshots are preserved. Resume reloads
that run's ORIGINAL source/source.zip. Do not change source under active training.
Step4 preflights the requested workload; step6 is explicit and off by default.
CPU/MPS pipeline and fresh-source local notebook execution do not verify CUDA,
live Colab or mounted Drive. Tiny models are scientifically unevaluated.

## Stability release

The updated notebook calls shared live monitoring while training blocks, rather
than printing only when run returns. scripts/collect_mesh_failure.py is a read-only
single-run collector that can be executed from ZIP bytes without installing new
scientific source under an old kernel. scripts/replay_mesh_checkpoint.py compares
one batch under the original matching source without optimizer steps. The exact
audited corrected-source transition is pinned in mesh/patches/stability_v1.json,
included in package data; it preserves parent checkpoint/source bytes and all
training state in a NEW M1 run. See docs/mesh-training-recovery.md before recovery.
Do not infer a healthy remote checkpoint or CUDA stability from CPU/MPS tests.

## Retained legacy performance notes

Performance update —31 October2026: docs/training-performance.md documents native
precision selection, bounded reference/frozen caches, timing semantics and the
same-config before/after Colab benchmark. scripts/benchmark_training.py and
scripts/benchmark_references.py are included. The original baseline ZIP is supplied
separately alongside the updated source ZIP; generated benchmark models/results
remain excluded. Open the updated notebook, restart the Colab kernel, upload the
updated source ZIP, and keep RUN_TRAINING=False while profiling. Existing source
caches and scientific runs must be preserved. A new source fingerprint does not
silently authorize changing old-run precision/source on resume.

Actual-batching update: docs/batched-training.md and scripts/benchmark_microbatches.py
provide a fixed-effective-batch accumulated versus8/16/32 cold/warm experiment.
The updated Colab cell4 opts in with RUN_MICROBATCH_SWEEP; default training remains
off. Original requested configuration is preserved; effective-batch overrides
belong only to the explicit comparison. Source packages exclude all benchmark
weights/data and previous output/source archives. Download the corresponding new
notebook, restart the kernel, then install the new archive before benchmarking.

Batch application update: docs/batch-settings.md describes the active
Configure → Benchmark optional candidates → Select/apply → Review → Train sequence.
The notebook's Apply selected batch settings cell/button rebinds the actual Session
through the shared public API; no manual internal attribute edits are needed.
Candidate1024 is configurable, not promised to fit. Benchmark results, application
and actual training are distinct states. Disposable benchmark binaries are excluded
from source archives and future persistent snapshots; result metadata is retained.

FP16 anchor recovery update: docs/anchor-fp16-recovery.md traces the supplied
failure, documents the bounded successful-update anchor rule, and gives exact
restart/source-reload instructions plus an original-Medium-job T4 verification
cell. Open the updated notebook to see anchor failures/tracebacks directly.
Architecture/objectives/point counts are unchanged. Historical checkpoints and
experiments are excluded from this archive and must remain in their existing
locations; new source does not silently authorize old-run source changes.

Reconstruction diagnosis update —1 November2026: docs/reconstruction-diagnosis.md
contains the actual imported Medium inventory, immutable case/oracle CLI, exact
sensor/conditioning controls, optional bounded diagnostic head and latent-correction
commands. Existing model and case artifacts are deliberately separate from source.
Keep the original matching evolution/evolution.pt for oracle E/A2 checks: portable
inference exports omit E. New exports retain actual stage/gate provenance. No old
hash, checkpoint or gate is edited. The default noiseless training fraction remains0;
only explicit new experiments change it. CUDA/Colab execution of these additions
remains unverified locally; restart/reload source before using the new CLI there.

Current notebook: eight steps, verified source, pre-training persistence probe,
unique run ID, first/cadence/stage saves, complete source/encoder/optimizer/RNG
snapshots, final verify/export and recorded-config restore. Keep the entire
content-addressed Drive run store, or a diagnostic/resumable download, not only
the inference view. New objectives are experimental; see
docs/observation-aligned-training.md and docs/observation-aligned-verification.md.

Current new-run field-loss default: Standard MSE. Region-balanced MSE remains an
explicit experimental choice (alpha.5 starting value). See
`docs/standard-mse-workflow.md` for UI/CLI/Colab selection and recorded-run
compatibility. Historical source archives retain their original defaults.

Known-time/A2 update —1 November2026: new runs use observation_time continuous_uniform,
independent from A2 target times; old runs retain recorded/legacy distributions.
docs/measurement-times.md and docs/a2-investigation.md describe exact configuration,
fixed current A2 previews/validation, read-only live-run export and safe transition.
Never replace source beneath a running trainer. Verify its safe-stop snapshot first;
resume with that snapshot's original source or create a NEW experiment after restart.
The source package includes the self-contained diagnostic exporter and refreshed
8-step notebook, .nvmrc and SOURCE_PACKAGE.md. No checkpoint/data/accepted output
is included. After extraction verify SOURCE_MANIFEST.json hashes before installation.
The current delivery archive is outputs/measurement-times-20261101/
pdeflow-continuous-measurement-times-source.zip on the originating checkout, with
adjacent .sha256 and notebook. This path is an output, intentionally excluded from
itself. Follow docs/measurement-times.md → Exact notebook reload for fresh-kernel
installation and independent A2/B policy review without changing an old run.

The November1 spatial-conditioning revision additionally includes B0–B3 selectors,
versioned direct sensor attention, independent wave kinematic objective, explicit
quantity weights and shared bounded comparison tools. Follow
`docs/spatial-conditioning.md` for safe Colab source reload, exact frozen-parent
copy/warm-start commands and outstanding T4/Drive acceptance. Source packages do
not contain the downloaded parent checkpoints or immutable diagnostic cases:
obtain your own verified complete resumable artifact, or generate new shared
validation cases through the supplied probe tool. All models used here remain
research models; tiny probes do not establish improved reconstruction accuracy.
