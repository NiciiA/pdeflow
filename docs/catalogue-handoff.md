# Training catalogue and native 512² handoff

From the repository root, activate the existing environment and use supported
Node22.23.3/.nvmrc on your PATH. No bundled Codex runtime path is needed.

```sh
source .venv/bin/activate
pdeflow serve
```

Second terminal: `cd frontend` then `npm run dev -- --port 5173 --strictPort`.
Open http://127.0.0.1:5173 (API http://127.0.0.1:8000). Models & Training discovers
CLI runs and polls real manifests/ownership, with an immediate refresh action.
Closing that page does not terminate training. Startup and planning never train.

## Train one native512 heat model

```sh
pdeflow train --preset heat-2d-512 --device mps
pdeflow status latest
pdeflow resume latest
pdeflow evaluate --run latest --device mps
```

The substantive profile is10000 updates,batch1,73,939-parameter direct-field
periodic U-Net,float32,unit512²,T3=[0,.05,.1]. Known D is conditioned in [.03,.12].
Online references/calibration are real; no full training dataset is saved. Scaling
is frozen from training-only calibration. Validation/checkpoints every100 updates.
Codex did not run this full budget. Resource/method execution, not full trained
quality, was verified on MPS. Keep Python source unchanged between interruption
and resume; old optimizer states still require their recorded source. Old inference
exports load independently of that rule.

## Full, missing, retrain and resume

```sh
pdeflow catalogue
pdeflow catalogue --json
pdeflow coverage
pdeflow train-all --device mps --plan
pdeflow train-all --device mps
pdeflow train-all --device mps --missing --plan
pdeflow train-all --device mps --missing
pdeflow train-all --include heat-2d-medium --force --device mps
pdeflow suite-status latest
pdeflow resume-suite latest
pdeflow resume-suite latest --retry-failed
```

The default full/all catalogue has11 targets: the original six heat/Darcy/chemistry
2D/3D layouts, medium64²,large128²,native512²,native1024² heat and the distinct
colab64² CNN profile. Explicit local-pilot/quickstart/capacity-local subsets remain.
Execution is sequential. Missing/reuse checks actual completion, budget, architecture,
distribution and validated export shapes; a smoke does not satisfy a full target.
Inference guidance/step counts do not define another trained prior. Source/seed/
device rules and target details: training-catalogue.md.

Your existing six-target suite completed its original3000/1500 budgets; it does not
need resuming to obtain new targets. suite-status shows its frozen original success
and targets absent from that snapshot. A new --missing suite adds the uncovered
work; it never edits the old manifest. CUDA/actual Colab and complete expanded-suite
resource fit remain untested. No grid/physics is silently reduced after failure.

## Use the 512² workbench and CLI

After your compatible checkpoint exists, insert Heat or click Load512² heat example
and review/apply the diff. Keep domain/physical times exact; D remains a known
condition, not a separate checkpoint for every value. In Reference experiment,
expand Analytical example; review u0 and full u_ref separately, insert/edit, verify,
solve, sample observations, then reconstruct. Full initial/reference fields stay
out of inference. In Reconstruction use one sample/two generation steps for a
first bounded check. Expand Feedback controller and budgets and explicitly choose
Use requested native grid budget (262144), or disable feedback. Never confuse
that reviewed limit with a full GPU-memory estimate. Reference/error appears
automatically after sampling and survives reopening.

```sh
pdeflow experiment --problem configs/reference/heat-512.json --model auto --inference configs/inference/heat-512.json --output outputs/my-512-experiment --device mps
pdeflow compare --result outputs/my-512-experiment/reconstruction --reference outputs/my-512-experiment/reference --output outputs/my-512-comparison.json
pdeflow models
```

Use a fresh output directory each time. The12-step/one-particle inference file is
an explicit user-launchable configuration, not an extra validated model-quality
claim. Run history provides copyable low-level reconstruction/evaluation commands
with its actual config/checkpoint/fixed-validation input paths.

To inspect the already-created bounded QA512 model and actual saved comparisons
without more training, serve this workspace instead:

```sh
pdeflow serve --workspace outputs/catalogue-512-20261026/workspace
```

This workspace links your existing exports read-only and contains a10-update512²
verification export. It remains partial against10000; it is not a scientifically
accurate model. Its exact locations, reference/sampling/jobs and screenshots are
under outputs/catalogue-512-20261026/. CLI/header source hashes distinguish stages;
use the current source for new runs and the recorded source for old optimizer resume.

## Storage and safe cleanup

```sh
pdeflow storage
pdeflow storage --refresh
pdeflow storage --json
pdeflow cleanup --preview
pdeflow pin local/suites/20261026T171753-local-pilot-281503ff
pdeflow pin local/suites/20261026T171753-local-pilot-281503ff --unpin
```

Pin paths are relative to the default repository outputs root; add --root PATH for
another managed root. The UI shows sizes/types/timestamps, run/suite, active/pinned/
referenced/protected status, and previews exact affected paths. Cache totals are
logical public-artifact sizes; quarantine remains on disk outside those totals.
No scientific outputs, QA, exports, resumes, provenance or unknown files are cleanup
candidates. Actual task previews contain zero eligible files; no real apply occurred.

A producer or a user can explicitly declare a known disposable cache, rather than
inferring ownership from its name. Optional CLI declaration (replace the path with
a cache you reviewed): `pdeflow cleanup --register-cache RELATIVE_CACHE_PATH`.
This only writes metadata; recent(<24h), changed, active, pinned, referenced, symlink
or tracked files still cannot be applied. Then preview again, review every path,
and only explicitly apply that exact digest: `pdeflow cleanup --apply PLAN_ID
--confirm REVIEW_DIGEST` (same --root if configured). It rechecks current state
and serializes against pin/affected-run ownership, then moves into recoverable
.storage/quarantine with recovery.json. There is no permanent-delete operation.
No automatic historical pruning or source-archive regeneration was added.

## Investigate model quality

```sh
pdeflow diagnose-learning --checkpoint outputs/local/runs/20261026T171753-local-pilot-281503ff-heat-2d/training/inference.pt --output outputs/my-learning-diagnostic --device cpu --updates 100 --cases 2
```

The command copies the model for fixed-pair overfit and never changes your export.
It includes a tractable CFM toy, physically conditioned prior statistics/spectra/
decay and three validation seeds at4/8 integration steps versus the conventional
baseline, with actual parameter count and measured compute. It is bounded to small
compatible layouts; it never downsamples a larger checkpoint secretly. Prior fit
is not paired reconstruction error. Current evidence:0.09105→0.00903 overfit loss,
but obs+PDE unobserved relativeL2=1.24281/1.28308 at4/8 steps versus Fourier0.83737.
No winning budget was selected; no test set or long study was used. Earlier matched
capacity and training-budget negatives remain preserved and separate.

Seven supplied lecture PDFs are mapped by physical page in generative-foundations.md.
The actual model is direct-field CFM; AE/VAE, autoregression, discrete likelihood-
trained flows and diffusion remain background/optional rather than new implementations.
No KL/ELBO term or exact density/posterior claim was added.

Verification/evidence/failures/backend limits: catalogue-verification.md. Next
scientific slice: validation-designed repeated-seed prior/reconstruction diagnostics,
controlled guidance/observation geometry and native baseline convergence, followed
by untouched testing. Do not attribute poor quality solely to model size or budget.
