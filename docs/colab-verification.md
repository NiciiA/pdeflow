# Colab workflow verification — 27 October 2026

This record covers the new substantial heat2D Colab workflow. All runtime checks
below were performed in this task on the Mac, **not inside Google Colab**. Only
bounded CPU optimization was run. Existing historical verification documents
remain historical evidence; their CUDA/Colab gaps are not filled by this work.

Evidence directory: `outputs/colab-training-20261027/` (excluded from source ZIP).
Operating instructions and exact presets: [colab-training.md](colab-training.md).

| Check performed this session | Result | Saved evidence / limits |
|---|---|---|
| Affected shared coordinate/local/Colab Python checks | 34 passed, 1 skipped | `shared-corrected.log`; actual CUDA test skipped because no CUDA |
| Notebook metadata/syntax + all six actual cells in order + safe setup rerun | 2 passed | `notebook-final.log`, `notebook-run-all.log`; real archive extraction/editable installation, IPython display, profile, AE2/flow2, gate setting, persistent filesystem and export |
| Legacy notebook source/bootstrap structure | 1 passed, 1 deselected | `legacy-notebook.log`; preserved original notebook, not a new legacy numerical run |
| Frontend tests | 49 passed in 10 files | `frontend-tests.log` |
| TypeScript + production build | Passed | `frontend-build.log`; existing large legacy chunk warning remains |
| ESLint | 0 errors, 3 existing component fast-refresh warnings | `frontend-lint.log` |
| Real rendered playground numerical integration | 1 passed | `browser.log`, `playground-notebook-import.png`; notebook-generated export imported, eight measurements submitted, actual CPU numerical reconstruction rendered |
| Large gallery / twice-finer reference comparison | Six families generated on CPU | `large-reference-gallery.png/.json`; float64 common-node RMSE3.31e-17–8.77e-17; shared spectral reference, not independent solver or learned-model evidence |
| Large / XL configuration and model sizes | Validated; parameters counted on meta device | `large-summary.json`, `extra-large-summary.json`; no large weights allocated or optimization executed |
| Dependency environment | Existing CPU stack accepted | `environment-check.json`; no PyTorch/CUDA replacement |
| Whitespace / source archive integrity | Final packaging record below | `whitespace.log`, `archive-verification.json` |

Commands (repository root, existing Python environment):

```bash
.venv/bin/python -m pytest tests/integration/test_colab_training.py \
  tests/unit/test_coordinate.py tests/integration/test_coordinate_pipeline.py \
  tests/integration/test_training_workflow.py -q
# For real notebook execution, set this to an isolated environment with IPython
# and the supported real core dependencies; without it the execution test skips.
PDEFLOW_NOTEBOOK_PYTHON=/path/to/isolated/bin/python \
PDEFLOW_NOTEBOOK_EVIDENCE=outputs/my-notebook-evidence \
  .venv/bin/python -m pytest tests/integration/test_colab_notebook.py -q
cd frontend
npm test
npm run build
npm run lint
# With API/Vite running and notebook-run/export/coordinate.pt at the saved
# evidence path used by this focused test:
npx playwright test e2e/colab-export.spec.ts --workers=1
```

The real notebook test changed only form/path settings: pipeline-check, explicit
CPU, filesystem persistence instead of Drive, explicit gate override permission,
training enabled and browser download disabled. It executed source setup, data,
preflight and both stages in a fresh isolated Python process; it did not replace
cells with mocked trainer calls. All six cells compile and have empty saved
outputs. Notebook GUI forms, upload/download dialogs and Drive authorization were
not exercised in Colab. The base local environment was preserved; an isolated
IPython8.37 environment used the existing core dependency libraries read-only.

Focused Python checks include deterministic split/normalization identities,
JSON-order-stable recipe identity, unique equal-time queries, width/head checks,
exact uninterrupted-versus-resumed CPU weights/RNG, weighted microbatch agreement
(including an incomplete group), activation-checkpoint gradient agreement,
precision policy rejection, deliberate budget/scheduler continuation, meaningful
source compatibility and explicit override, gate failure/override, interruption,
partial export, deletion of disposable test-local state then verified restore,
corrupt/incomplete persistent uploads, retention and frozen dependency protection.
An exported checkpoint is copied alone into a new directory, loaded in a clean
CPU subprocess, and reconstructs only from measurements/public metadata. Real
API import is exercised, including a partial state. The actual browser test
asserts no initial/reference field or hidden known-field payload enters the
reconstruction request. Mocked precision policies are not GPU evidence.

## Scientific interpretation

The saved notebook example has two AE and two flow updates. Its representation
validation MSE is **0.1579427063**, versus constant predictor MSE **0.1667221206**,
a ratio of **0.9473410350** on six validation solutions. Beating that constant
baseline is a minimal gate only. The rendered comparison is visibly poor
(field RMSE about0.369 for that synthetic browser case). No new model is trained
enough to assess useful scientific accuracy. Individual and ensemble-mean errors
remain separate in `notebook-run/flow-validation.json` (six solutions × two counts
× two noise levels, two particles, two Heun integration steps, 8² output here).
No test split is used in these gate/training/validation checks.

Large and XL have respectively AE/flow parameter counts 3,235,329/18,675,200 and
7,261,441/60,880,128. Storage estimates are arithmetic, not GPU-memory or runtime
measurements. Single-grid training plus coordinate queries establishes execution
flexibility, not learned resolution generalization. Previous conventional Fourier
inverse results remain better; no winning budget is selected using test quality.

## Failures found and repaired

Initial isolated notebook setup lacked the actual core dependencies; that test
environment was corrected without reinstalling the project environment. Actual
cell execution then exposed editable-install imports missing from the already
running kernel. Setup now adds the chosen source path explicitly and refuses
switching source after modules are loaded. Cache verification protects reruns; the actual setup cell was rerun after model imports and a modified disposable source cache was rejected without overwriting it.

The shared stage-driver refactor passed in-process tests but failed the real
backend-restart/detached-worker regression with `NameError: coordinate_stages`.
The module entry point had preceded its definition. Moving that entry point to
the bottom fixed the real worker; the affected 34-test set then passed. Failed
logs remain (`notebook-check*.log`, `shared-final.log`); the current success logs
are `notebook-final.log` and `shared-corrected.log`. No tolerance was relaxed.

A final recovery review also corrected ordinary resume after a budget extension
(the original scheduler horizon must survive subsequent resumes), and partial
export during an interrupted extension (use the latest saved boundary, not an
older completed export). The regression now extends, interrupts, exports the new
partial state, resumes normally and reopens the completed extension. The current
34-test and six-cell success logs include these corrections.

## Remaining hardware and runtime work

Actual CUDA BF16/FP16/scaler behavior, selected fused kernels, GPU gradients,
activation memory, timing, OOM handling and full-run validation headroom remain
unverified. MPS was not run for this continuation. CPU accumulation/checkpointing
checks do not establish GPU performance or exact cross-device reproducibility.
Actual Colab runtime installation, private-secret checkout, Drive authentication,
interrupt/disconnect and remote filesystem recovery require a real Colab session.
Filesystem corruption/recovery tests simulate partial storage on a local
filesystem; they do not certify Google Drive behavior. Optional compilation is
explicitly unsupported. There is no automatic capacity reduction after OOM.

Recommended next bounded task: run the requested architecture's disposable
three-update-per-stage profile on the actually assigned Colab GPU, inspect
memory headroom and validation cost, then train representation only for a
predeclared bounded pilot. Inspect physical, initial and spectral validation
errors across all six families before funding substantial flow training. Keep
the test split untouched while choosing that budget. A larger model or gate
ratio below1 does not itself imply accurate reconstruction.

## Preservation and delivery

`preservation-before.json` recorded zero existing `.pt` files at task start and
the Git index fingerprint. No previously present scientific outputs/checkpoints
were removed or overwritten. New tiny verification weights remain under this
session's evidence directory; they are pipeline checks. Source files retain
existing staged/unstaged/untracked user changes; no commit, reset or staging was
performed. The source archive excludes outputs, environments, caches, datasets,
weights, secrets and personal reference files. It includes both notebooks,
Python/frontend source, npm lockfile, project metadata, configs, scripts, tests,
documentation and a SHA256 manifest. Historical output paths in documentation
refer to the original machine and are not bundled.

Final archive: `outputs/colab-training-20261027/pdeflow-colab-source.zip`.
Its open/hash/expected-file inspection is saved in `archive-verification.json`.
Task-owned API/Vite verification services are stopped after browser checks;
`preservation-after.json` and `process-check.txt` record final preservation and
process state. No unrelated service is stopped.
