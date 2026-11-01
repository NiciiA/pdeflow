# Historical source record — superseded 30 October2026

Generative commands in this record require their original source. Current usage is in the repository README and joint-latent-model.md.

# PDEFlow source package

Current heat2D workflow: `notebooks/colab.ipynb`, `docs/continuous-heat.md` and
`docs/colab-training.md`. Training support is continuous physical[0,1]; display
frames are separate. Upload this current source ZIP, not an older snapshot ZIP.
Historical multi-specialist examples below remain available through Legacy tools.

Extract the ZIP and work inside its `pdeflow/` directory. This package includes
Python source, React/TypeScript frontend source and npm lockfile, project and
dependency metadata, configurations, tests, local launch/verification scripts,
the Colab notebook, documentation and shadcn attribution. `SOURCE_MANIFEST.json`
lists every payload file's size and SHA256 (the manifest does not hash itself).

The archive deliberately omits `.git`, `.venv`, `node_modules`, hidden caches,
generated build/egg-info metadata, runtime outputs, datasets, checkpoints,
private files and references. No external reference code is bundled. Accepted
experiment outputs remain on the originating machine; they are not needed to
install the package. Paths to those historical results in the documentation
describe that machine and do not imply the outputs are inside this ZIP.

Use Python3.12 and Node22.23.3 (.nvmrc), then the ordinary workflow:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[workbench]"
pdeflow setup
pdeflow train --preset quickstart-heat-2d --device cpu
pdeflow evaluate --run latest
pdeflow serve
# Separate terminal: cd frontend && npm run dev
```

The package automatically prepares fixed evaluation and training-only calibration,
trains online without a full saved training dataset and discovers its exports.
Use heat-2d/darcy-2d/chemistry-2d --device mps for actual longer pilots (not run by
setup). Resume with `pdeflow resume latest`. For headless use setup --headless;
Node is optional. See docs/everyday-usage.md. No accepted QA directory or
PDEFLOW_MODEL_ROOTS is needed. Keep an already usable .venv; if a pip-less uv
venv needs pip, run `python -m ensurepip` inside it rather than recreating it.

The following explicit-config examples are retained advanced alternatives.
To obtain a compatible model, either copy your own exported `inference.pt`
and its public config/problem/observation files, or explicitly generate a small
demo. This performs real, bounded training; opening the UI does not train:

```bash
pdeflow smoke --config configs/experiments/cpu_2d.json --output outputs/demo-2d
export PDEFLOW_MODEL_ROOTS="outputs/demo-2d/training"
python scripts/dev.py
```

Open http://127.0.0.1:5173 and follow `docs/multispecialist.md`. For other dimensions,
use `cpu_1d.json` / `cpu_3d.json`, distinct output folders, and add their training
folders to `PDEFLOW_MODEL_ROOTS` separated by the platform path separator.
The smoke command generates observations and evaluation truth under `data/`.
These tiny models demonstrate execution and are not scientifically accurate.
The current UI starts empty: insert Heat and let equation-driven routing select
the compatible checkpoint; layout suggestions require explicit review.
For the real second family, run:

```bash
pdeflow smoke --config configs/experiments/darcy_cpu_2d.json --output outputs/demo-darcy-2d
pdeflow smoke --config configs/experiments/darcy_cpu_3d.json --output outputs/demo-darcy-3d
export PDEFLOW_MODEL_ROOTS="outputs/demo-2d/training:outputs/demo-darcy-2d/training:outputs/demo-darcy-3d/training"
python scripts/dev.py
```

Insert Darcy, explicitly set its known mean-zero forcing, then generate sensors
and reconstruct. Use dimension3 for the genuine static3D checkpoint.
The Darcy inverse baseline is implemented; smoke checkpoints do not establish
scientific accuracy. See docs/darcy-study.md for the bounded negative study. See docs/symbolic-editor.md for the mathematical
request and docs/local-training.md for larger sharded datasets and MPS commands.
CLI reconstruction/evaluation and Colab export instructions are in README.md
and docs/handoff.md. The exact historical pilot can be generated with:

```bash
python -m pdeflow.experiments --config configs/experiments/local_budget.json \
  --output outputs/my-budget-study
```

This is a one-seed pilot; reproducing execution does not guarantee bitwise
agreement across devices. CUDA and actual Colab execution are untested.
Repackage the current source using:

```bash
python scripts/package_source.py --output outputs/pdeflow-source.zip
```

The third physical specialist is real A+B reversible C chemistry. To generate
a small checkpoint from this source without a saved training set, run
`pdeflow smoke --config configs/experiments/online_chemistry_2d.json --output outputs/demo-chemistry-2d`
(and online_chemistry_3d.json for true3D in a distinct root). Add its training
folder to PDEFLOW_MODEL_ROOTS. docs/online-training.md gives prepare-evaluation,
online/hybrid/offline training, resume, reconstruction and Colab commands.
Numerical clean references and fixed evaluation remain necessary even when no
full training dataset is saved. All runtime outputs/weights/data remain omitted
from this archive. The legacy notebook skips full training-data generation for
online/hybrid and persists fixed evaluation and checkpoints explicitly; it is
available as `notebooks/colab-legacy.ipynb`.

The process-diagram continuation adds the optional bounded feedback controller
and one global2× periodic nodal heat refinement level. See
`docs/process-audit.md` and `docs/process-verification.md`. The source archive
omits the verification outputs/checkpoints referenced there. Generate fresh
bounded models with process_heat_2d.json / process_darcy_2d.json /
process_chemistry_2d.json and process_heat_fine_smoke.json using `pdeflow smoke`
and distinct output roots. Fixed evaluation inputs are under evaluation/test/0/.
Refinement starts a fresh compatible fine-grid inference from the same actual
measurements; it is not interpolation credited as physical recovery. Default
process thresholds are unconfigured, so ordinary demo completion reports budget
exhaustion. The later FK continuation implements a specified EM/systematic particle path
and verifies it on CPU/actual MPS; FK plus refinement/retries are rejected. See
docs/fk-mathematics.md and docs/fk-verification.md. No scientific accuracy,
CUDA or actual Colab execution is implied by this archive.

## Generating an FK demo without accepted outputs

FK needs no additional training objective. After generating outputs/demo-2d
above, derive a matching inference config from the same public smoke profile:

```bash
python - <<'PYCONFIG'
import json
from pathlib import Path
c=json.loads(Path('configs/experiments/cpu_2d.json').read_text())
c['inference'].update(particles=8,steps=8,fk={'chunk_size':4,'ess_fraction':0.7,'resample_every':2,'observation_weight':0.001,'physics_weight':0.0001},controller={'max_particles':8,'max_network_evaluations':128,'max_sampling_steps':8})
Path('outputs/demo-2d/fk-config.json').write_text(json.dumps(c,indent=2))
PYCONFIG
pdeflow reconstruct --config outputs/demo-2d/fk-config.json --mode fk \
  --checkpoint outputs/demo-2d/training/inference.pt \
  --problem outputs/demo-2d/data/test_problem.json \
  --observations outputs/demo-2d/data/test_observations.json \
  --output outputs/demo-fk
pdeflow evaluate --result outputs/demo-fk/fk \
  --reference outputs/demo-2d/data/test_reference.npz
python -m pdeflow.fk.toys --kind gaussian --output outputs/demo-fk-gaussian
```

The config-generation/CLI route is validated against the same packaged schema;
this archive workflow explicitly trains a demo only on user launch. No datasets,
weights or personal scientific reference files are bundled. Copy compatible
exports and inputs from your own saved runs to avoid retraining. FK snapshots
are inspection artifacts, not resumable inference checkpoints. Terminal weights,
ancestry and actual target assumptions remain part of the result.

New reference/suite commands are included. After the compatible8²
quickstart-heat-2d export exists, run:

```sh
pdeflow experiment --preset heat-2d --model auto --output outputs/my-experiment --device cpu
pdeflow train-all --plan
pdeflow train-all --suite quickstart --device cpu
pdeflow resume-suite latest
```

Reference-only work needs no checkpoint:
`pdeflow reference --problem configs/reference/heat-2d.json --output outputs/my-reference`.
See docs/reference-experiments.md and docs/training-suites.md. Full suite pilots
use explicit larger grids; they are not required to open a compatible8² demo.
All generated reference fields, noisy measurements, clean evaluation-only samples,
checkpoints and results stay in configurable runtime output directories outside
this source archive. Numerical seeds/parameters and regeneration commands are
included; historical results can be copied separately without overwriting them.

The latest source also includes explicit automatic/registered model selection,
periodic multiscale U-Net capacity presets, the CFM mathematical audit, bounded
native-resolution probes and linked point/generation workspace. See
`docs/flow-workspace-handoff.md`. No trained1024² model or validated high-resolution
inverse result is bundled or claimed. Generate compatible checkpoints with the
specified managed presets; use exact matching reference layouts in configs/reference/.
The supplied reference animation and inspection frames remain outside this archive.

## Substantial heat2D Colab workflow

The main `notebooks/colab.ipynb` now trains the coordinate latent heat2D model
through the same scientific trainer and stage driver as local training. Upload
this notebook into Colab, select a GPU runtime, upload this source ZIP in cell 1,
connect persistent storage, choose Large or Extra large, profile, then explicitly
set RUN_TRAINING=True. See `docs/colab-training.md` for settings, recovery,
quality-gate handling, partial/completed exports and local playground import.
The source ZIP contains the procedural data recipe and presets, not a full
training set. Regeneration uses deterministic per-split solution seeds. Historical
scientific outputs remain outside the archive. This continuation performed CPU
pipeline checks only; no Large training or actual Colab/GPU/Drive run is claimed.
