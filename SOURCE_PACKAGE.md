# PDEFlow operator-aware heat/wave source package

Extract and work inside pdeflow/. Follow README.md and notebooks/colab.ipynb for
the active operator-aware physical-state v2 architecture and A1 → A2 → B workflow.
Historical trajectory-model instructions require their original source; their
checkpoints are preserved and incompatible with this model.

Includes package source, frontend source and npm lockfile, dependency/project
metadata, model/data configurations, tests, local launch/verification scripts,
the active Colab notebook, documentation and attribution. SOURCE_MANIFEST.json
lists every included file's byte count and SHA256. The adjacent .sha256 file
records the ZIP hash. The legacy notebook is deliberately excluded.

Excludes outputs, .git, .venv, node_modules, caches/builds, datasets, checkpoints,
secrets, unrelated personal files and user scientific reference PDFs. No accepted
experiment output is moved into the archive or deleted. Historical artifact paths
in documentation describe the originating checkout; they are not promised ZIP
members. The supplied design PDFs remain with their owner and are not redistributed.

The shared trainer generates reference data deterministically on demand; no external
dataset download is required. For a new run, use the managed training command or
Colab workflow in README.md. Start with representation validation. A failed gate
must be investigated; a two-update pipeline check is unsuitable for accuracy.
The archive intentionally contains no pretrained weights. Generate pipeline-check
weights with the documented smoke configuration, train a properly evaluated model,
or obtain a compatible v2 coordinate.pt export from your own prior run. On the
originating checkout, outputs/operator-wave-20261030/final-cpu/flow/coordinate.pt is
a preserved two-update-per-stage smoke export, not an accurate pretrained model.

Portable inference exports contain graph/observation/context encoders, conditional
flow, learned evolution, spatial/scalar decoders, transforms and provenance. They
need no external A1/A2 file or dataset at inference. Resume requires the original
stage directory, optimizer/RNG state, hashed dependencies and compatible scientific
source. Never convert old weights by renaming state keys.

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
