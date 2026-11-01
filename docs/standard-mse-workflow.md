# Standard MSE and experimental region-balanced MSE

For NEW runs, Standard MSE is now the shared default in CoordinateConfig,
training presets, source JSON profiles, the training UI, managed CLI and Colab.
This supersedes the previous new-run RB recommendation; historical verification
records describe what ran at their recorded time and were not rewritten.
Architecture, capacity, physical/family/time distributions and unrelated losses
are unchanged. This simpler baseline does not establish improved reconstruction.

## Selection and stages

Open Train model → Field reconstruction loss. Choices are **Standard MSE —
default** and **Region-balanced MSE — experimental**. Other existing WMSE/MAE/
spectral choices remain supported. Only RB shows alpha and collapsed advanced
region controls/preview. Alpha starts at .5; it is not an optimized value.

“Adds equal emphasis across detected features. Experimental; may trade whole-field
accuracy for feature accuracy.” Training loss controls learning. Display metrics
only change evaluation.

A1 encoder/decoder reconstruction and A2 decoded evolution use the selected field
objective, with the existing global per-family/quantity normalization. A2 keeps
its paired latent/composition terms. B's latent FM velocity loss, endpoint
observation auxiliary, optional physics terms, inference-time correction and
evaluation/error-map selection are independent and unchanged. No spatial region
weights are assigned to latent tokens.

Standard MSE estimates the mean squared error over the original uniform spatial
query measure at each supervised time; average time groups, active quantities
and realizations with existing documented weights. The physical-time initial/
early/uniform/final mixture is unchanged, and is not uniform space-time MSE.
Predictions/targets are in the same normalized units for training; physical metrics
invert normalization separately. Zero-weight invalid points are masked BEFORE
subtraction, so padded NaNs cannot leak into either loss or gradients. No valid
nonfinite values are concealed.

RB is unchanged: `(1-alpha) domain_MSE + alpha mean(region_MSE)` per supervised
physical time/solution, then the existing quantity/batch reductions. Regions are
reference-only, signed, periodically connected, area/contrast filtered. Whole-domain
queries are unchanged; fixed additional region strata retain their estimator
weights, alpha0 and no-region fallback. There is no query dropping/reallocation.
See field-reconstruction-losses.md for exact segmentation and quadrature.

Standard MSE now skips analysis fields, connected components, extra regional targets,
region queries and the training region-cache object. Explicit region evaluation/
preview diagnostics still work when requested; they are separate from training.
The resolved summary reports both domain and additional region-query counts.
For the pipeline preset these are32+0 for Standard,32+256 for RB per active
quantity/time; changing loss never reallocates the removed256 to domain queries.

## CLI and Colab

Inspect a new configuration without training:

```bash
pdeflow training start --plan --device cpu --field-loss mse
pdeflow training start --plan --device cpu --field-loss region_balanced_mse --field-alpha 0.5
```

Remove `--plan` only when ready to train. Development MPS examples:

```bash
pdeflow training start --problem heat-wave-2d --budget development --device mps --field-loss mse
pdeflow training start --problem heat-wave-2d --budget development --device mps --field-loss region_balanced_mse --field-alpha 0.5
```

These commands launch NEW runs, not retraining existing checkpoints. No such
500-update development run was launched for this implementation. Existing gates
still apply. Resume keeps the recorded choice with `pdeflow training resume latest`;
there is no loss-override flag on resume. Scientific source compatibility remains
strict; use the recorded source for old runs rather than silently migrating them.

For low-level configuration creation:

```bash
python scripts/joint_model.py config --preset xsmall --family joint --device mps --field-loss mse --output outputs/my-standard-config.json
python scripts/joint_model.py config --preset xsmall --family joint --device mps --field-loss region_balanced_mse --field-alpha 0.5 --output outputs/my-region-config.json
```

`coordinate-train --config ...` uses that explicit configuration. `--field-alpha`
requires explicit region-balanced selection. Source profiles now describe new MSE
experiments; do not replace a recorded run's configuration with an updated profile.

Open the updated notebooks/colab.ipynb. Step3 defaults `FIELD_LOSS="mse"`.
Select `"region_balanced_mse"` explicitly for RB; FEATURE_BALANCE=.5 is used only
there. Step5 prints the concise label and resolved query counts. Both choices call
the same repository trainer. The resume branch reads the stored configuration;
it does not merge today's loss default. Save/verify and complete artifact recovery
remain as described in observation-aligned-training.md.

## Recorded compatibility

Run plans/summaries, run.json, checkpoints and portable exports record the actual
field_loss plus resolved_field_loss (applicable settings, stage scope and costs).
The pre-training message is `Field reconstruction loss: Standard MSE` or
`Field reconstruction loss: Region-balanced MSE, alpha=...`.

Checkpoint schema operator_physical_state_v2 with engine<=4 and absent field-loss
records has documented MSE/legacy estimator semantics. Explicit recorded loss,
including top-level historical metadata, wins. Engine5 missing both records, or
conflicting config/checkpoint records, is genuinely ambiguous and rejects with
an explanation. No checkpoint is rewritten by loading. Resume still rejects
changed loss/settings and source identities; a loss change is a clearly new
experiment. An explicitly empty legacy config retains its legacy estimator rather
than becoming today's grouped preset. Session restore rejects absent ambiguous
loss configuration rather than guessing. All existing outputs remain unchanged.

## New verification (1 November 2026)

Evidence: outputs/standard-mse-20261101/. Only tiny optimizer budgets were used.

- Focused new/default/API/math/legacy/resume/field tests:37 passed,1 CUDA skip;
  one new test assertion initially attempted NumPy conversion of a gradient tensor.
  The corrected hand-calculated mask test separately passed (4.5 normalized loss,
  exact expected gradients). No numerical tolerance was loosened.
- Batching, configuration and joint-stage regressions:37 passed,1 CUDA skip.
- Frontend selector/training/evaluation tests:12 passed; TypeScript/build passed;
  ESLint0 errors/3 existing fast-refresh warnings.
- Actual notebook:3 passed (syntax plus real eight-cell execution for BOTH losses,
  tiny CPU A1=2, partial artifact verification and fresh-runtime restore). Real
  IPython, no mocked trainer. notebook/<loss>/ contains logs and stage checkpoints.
- Actual rendered browser:1 passed; both selections launched real isolated CPU
  A1/A2/B=2/2/2 pipeline checks, saved actual objectives/alpha and query counts.
  No manual gate override was enabled; the existing disclosed pipeline-check
  exemption remains unchanged. These are poor execution fixtures, not scientifically
  adequate models or evidence that either objective wins. browser/ has screenshots
  and actual-runs.json; browser-workspace/ retains configs, checkpoints and logs.
- The task-only API18084 and frontend5184 were stopped cleanly. Existing app services
  and runs were not stopped. CUDA/actual Colab/Drive execution remains untested;
  the affected field-loss/batch regressions include actual available MPS checks.

The suites overlap. No substantial training, test-based model selection or new
scientific accuracy study occurred. Source-package and preservation verification
are in source-package-verification.json and preservation-final.json in that root.

Load the new source ZIP in a RESTARTED Colab runtime, open its notebook, upload
that ZIP in step1 and paste the adjacent SHA256 if desired. Do not run an old
notebook or retain imported modules from the old source; cached-extraction and
scientific identity checks reject stale/mismatched source. Resume historical runs
with their original source archive and recorded objective. A controlled future
comparison must hold architecture, splits, time/observation policy, optimizer and
budgets fixed and report RB's additional compute. Standard MSE is not claimed to
fix the current poor reconstructions.
