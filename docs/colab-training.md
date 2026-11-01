# Colab: shared heat/wave operator v2

The active eight-step notebook is `notebooks/colab.ipynb`. Current save/restore, objective controls and exact instructions: [observation-aligned-training.md](observation-aligned-training.md). This supersedes the older cell sequence below. It installs this package and calls
its shared trainer; it contains no separate neural model or reference solver.
The previous trajectory-based models/notebook are historical and incompatible.

1. Choose a GPU runtime. Supply the current source ZIP or your own repository URL
   and exact revision. Credentials belong in a Colab secret, never in a URL.
2. Configure fast local and persistent roots. Drive mounting is optional and needs
   your interactive authorization. Runtime-local caches are disposable.
3. Choose FAMILIES=joint (balanced heat/wave), heat or wave. Choose capacity and
   precision independently of budgets. New runs use Standard MSE; explicit region_balanced_mse retains alpha=.5;
   MSE/weighted MSE/MAE remain explicit alternatives. Resume retains recorded loss.
4. Inspect actual training-only previews and real three-stage preflight. It uses
   disposable tiny update counts with your requested architecture/query budgets;
   it does not silently shrink capacity. Meta counts do not predict GPU fit.
5. Set RUN_TRAINING=True. `both` manages A1 → validation gate → A2 → B and paths.
   `representation` stops after A1 validation; `evolution` completes A2; `flow`
   requires a completed A2 or creates it from the chosen completed A1. Frozen
   dependencies are hashed. Gate overrides are explicit and recorded.
6. Interrupt once or call `session.stop()` for an optimizer-boundary checkpoint.
   Reconnect with the same source, name and recorded configuration, restore the
   persistent run, profile your allocated device, then choose resume. Extend only
   a declared increased budget; changing representation/operator requires a new
   downstream generation run. Never resume a v1 checkpoint as v2.
7. Export/download `export/coordinate.pt` and its JSON. Import the PT in the Mac
   Heat/Wave playground. It includes graph, observation/context encoders, coupled
   flow, learned evolution, spatial/scalar decoders, transforms and provenance.
   It excludes optimizer, training-only state encoder, training data and external
   path dependencies. CPU-first reload supports explicit CPU/MPS remapping.

A1 trains state encoder/decoder and scalar heads. A2 freezes these and trains the
shared PDE graph/evolution modules on complete states and composition. B freezes
both and trains conditional initial-state/scalar generation. Optional decoded
auxiliary terms are off by default; post-generation correction is separate.

Large is65,536 /2,048 /2,048 realizations; width768/depth16/heads12/384 TOTAL slots;
100k A1 +100k A2 +100k B updates. XLarge is larger. Inference/training counts and
normalization/loss/mixture details are in `operator-wave-model.md`. A1/A2 and default or new endpoint-likelihood B use actual mixed-family neural
microbatches with per-example/per-quantity reductions. Explicit accumulated control
and legacy decoded auxiliaries remain serial; configured capacity alone does not
predict throughput. CUDA AMP/backend performance,
memory headroom and useful accuracy require actual hardware profiling.

Persistent snapshots are content-addressed and verified. Incomplete uploads are
ignored; retention only affects owned snapshots. A runtime disappearance loses
updates after the last successful sync. Source/configuration changes reject strict
resume; allow_source_change is a deliberate non-exact-reproduction override.

Historical notebook cells were run locally on CPU through A1/A2/B. The new
eight-step notebook was executed locally through a two-update A1 run, failed
representation quality gate (not overridden), partial complete-artifact save and
fresh-runtime restore. Separate library tests exercise A1/A2/B objectives and full
artifacts; do not confuse them with a scientifically adequate model. This is **not actual Colab, Drive or CUDA verification**. See
`operator-wave-verification.md`; no substantial training was launched.
