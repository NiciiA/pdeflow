# Versioned sequential training suites

Current authoritative catalogue, state definitions, coverage and commands:
[training-catalogue.md](training-catalogue.md). The default train-all now means
full/all: all 11 advertised targets, including designated large/high-resolution
heat and 3D layouts. Planning never starts training.

```sh
pdeflow catalogue
pdeflow train-all --plan
pdeflow train-all --device mps
pdeflow train-all --missing --device mps --plan
pdeflow train-all --missing --device mps
pdeflow train-all --suite quickstart --device cpu
pdeflow train-all --suite local-pilot --device mps
pdeflow train-all --suite capacity-local --device mps --plan
pdeflow train-all --include heat-2d-512 --force --device mps
pdeflow suite-status latest
pdeflow resume-suite latest
pdeflow resume-suite latest --retry-failed
pdeflow coverage
```

Use the same --workspace when selecting custom persistent storage. Full uses the
11-target catalogue; local-pilot and quickstart explicitly retain the original six
layouts, and capacity-local selects medium64² and large128². Quickstart reduces
budgets/batch only. A utility 8² smoke is not a substitute for these layouts.

Every launched suite freezes its complete target list/configs and catalogue
version. Include/exclude, explicit per-member budgets, fail-fast/default versus
continue-on-error, advisory locks, safe-boundary interruption and atomic resume
remain supported. The first six seed indices remain unchanged; all later indices
are stable and inspectable. Completion/reuse validates actual successful effort
and loadable exports, not just a filename. New targets never silently mutate old
suite manifests. Status reports original completion and current targets absent
from the snapshot; current cross-run coverage is separate.

The user's original 3000/1500-update local suite completed its original six
targets and remains unchanged. Historical bounded CPU/MPS scheduling evidence
is preserved in reference-verification.md. New tests, native512 MPS execution,
UI/CLI monitoring and storage evidence are in catalogue-verification.md. No full
11-target suite was launched during this implementation. CUDA/actual Colab remain
untested; command/configuration support is not hardware verification.
