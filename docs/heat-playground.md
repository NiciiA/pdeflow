> Historical implementation record. The active heat/wave physical-state v2 workflow is documented in [operator-wave-model.md](operator-wave-model.md) and the current README. Previous whole-trajectory checkpoints are preserved and incompatible.

# Heat playground

Current continuous-time/sensor workflow: [continuous-heat.md](continuous-heat.md),
with [new verification/screenshots](continuous-heat-verification.md). New main
profiles use physical[0,1]; old snapshots retain their original capability.

The default application is a 2D, periodic, constant-diffusivity scalar heat
playground. The broader 1D–3D reconstruction, Darcy, chemistry, fixed-grid and
steering research implementation is preserved behind **Legacy** and its CLI.
This is a change to the default experience, not a universal model claim.

## Open and play

From the repository root, in terminal 1:

```sh
source .venv/bin/activate
pdeflow serve
```

In terminal 2, from the repository root:

```sh
cd frontend
export PATH="/opt/homebrew/bin:$PATH"  # this Mac's supported Homebrew Node
npm run dev
```

Open <http://127.0.0.1:5173>. The API listens on <http://127.0.0.1:8000>.
Node 22.23.3 is tested; `.nvmrc` also supports selecting it with nvm.
For first installation, follow README's Python 3.12 setup and run `npm ci`
inside `frontend`. Opening the application does not start training.

1. Press **Play**, move D or choose a starting field. Restart replays that same
   field; playback speed changes animation pacing, not physics.
2. Pause at a physical time, set N and absolute noise sigma, then **Take
   measurements**. **Resample** increments the measurement seed. Existing
   observations retain their times when you scrub or play.
3. **Run model** uses a compatible resolution-flexible 2D heat checkpoint.
   Existing models are found across project output workspaces. A model can also
   be imported using **Import model** and a completed `coordinate.pt` export.
   The file is validated and copied into the configured workspace, leaving its
   source intact. Barely trained models require **allow experimental use**.
4. Results show reference, model and absolute error at the same physical time.
   Reference and model share one color scale across the entire trajectory.
   Error has a separate labeled scale. More query pixels do not establish more
   learned detail. With no model, reference simulation remains available.

**Train model** opens a separate view, defaulting to heat-2D,
resolution-flexible, **Development run**, Auto device. Examples vary separately
from the visible experiment. Both learning stages, validation gate, stop and
resume reuse the existing orchestration. A Development run is an experiment,
not demonstrated sufficient training. No new playground training was launched.
Old resume files are preserved, but strict resume requires their recorded source
and configuration; this implementation update does not waive those checks.

Equivalent training command (run deliberately, not required to simulate):

```sh
pdeflow training start --problem heat-2d --model resolution-flexible --budget development --device auto
```

The [training guide](training-workflow.md) retains local/Colab/export/resume
commands. The [coordinate guide](coordinate-specialist.md) retains explicit
inference and fixed-grid comparison commands. These are advanced workflows.

## Starting fields and numerical meaning

Cosine uses domain-relative x and y, amplitude/background, integer wave counts,
and optional phases. Its product-mode decay is compared with the existing CPU
float64 FFT solver, independently of that closed formula. Constant/zero-wave
modes are retained. Only periodic boundaries and known constant D are supported.

Hot spots are smooth periodic bumps. Random fields reuse the existing seeded
smooth Fourier generator. Drawing adds/removes heat or erases toward zero in
the initial condition, with wrapped edge brushes and ten undo states; it never
injects a time-dependent source. Presets/formulas provide keyboard alternatives.

Initial expressions use the existing bounded AST evaluator, not Python eval.
Functions: sin, cos, exp, sqrt, tanh, abs; x/y/pi/D/t, with t=0; arithmetic and
bounded literal powers. Invalid/incomplete fields retain the last valid image
and show a local error. A complete u(x,y,t) expression in Advanced is verified
against initial, heat PDE and periodic value/derivative diagnostics. It does not
replace the numerical trajectory or silently become an analytical solution.

Reference evolution is the finite Fourier interpolant of 64×64 supplied nodes,
at 31 saved physical times. Animation consumes one cached response, not a solve
or network request per frame. General grid fields are not exact continuum
solutions. Formula/drawn discontinuities across the periodic seam are interpreted
through this finite Fourier representation and may ring.

Default measurements are exactly N scalar u values at the selected time.
Scattered values are evaluated using the reference's Fourier coefficients and
physical-time exponential evolution, not snapped to nodes. Locations and Gaussian
noise use independent SeedSequence streams. Saved sampling metadata includes
seed, coordinates, times, noisy values and actual absolute sigmas. Clean values
are stored separately in `clean-evaluation-only.npz`. Across-time/fixed-sensor
patterns also mean N **total** scalars. Manual positions and JSON import/export
are in Advanced. Import expects a JSON array of `{field:"u",x,y,time,value,sigma}`.
Import switches to measurements-only; no reference errors are supplied.

Inference goes through the existing validated coordinate request and job queue.
It receives the scalar rows, physical contract, query grid/times and inference
settings only. Initial/full reference/clean values are absent. Reference attachment
occurs only after successful reconstruction. The output grids 16²/32²/64² divide
the 64² reference grid, so displayed comparisons use saved nodes at actual times;
the existing attachment evaluator documents its interpolation convention as well.
The animation does not interpolate model time. Sensor RMSE is each sample's
decoded error against noisy observations, not truth. Field RMSE is the ensemble
mean's physical error across saved field/time queries. Runtime excludes model
loading and export. Ensemble spread is empirical, not calibrated confidence.

## Storage and permanent deletion

Open **Storage**. Select models and **Preview selected deletion**, or **Delete
all model artifacts…**. A concrete preview lists every export/resume/stage weight,
unique logical bytes, selected models affected and retained shared dependencies.
Type the displayed scope confirmation and press **Permanently delete** once.
Cancel does not change scientific files. This is permanent unlink, not trash.

Default scope includes recognized model weights and optimizer/resume checkpoints,
including autoencoder/latent-flow stages, fixed-grid models, retained budget
snapshots, and recognized Gaussian verification model bundles. Optional scopes
include only associated producer-owned generated data and recognized results/logs.
Datasets used by retained models remain. Imported data, references, source,
configs, documentation and uncertain files stay protected. External model links
outside the managed root are not followed or deleted; importing through the
playground creates a managed copy. Unknown `.pt` files are protected.

The normal managed root is project `outputs/`; an external configured workspace
uses only that workspace. Paths are contained, symlinks are not traversed,
hardlinks are counted once, previews fingerprint content and inode, and active
job/trainer locks prevent deletion. Shared stage dependencies cannot be deleted
while retained models need them. A changed artifact invalidates the preview.
Partial failures are reported individually; selections whose artifact was removed
are cleared and the UI rescans. Producer metadata remains for diagnosis; missing
exports are no longer usable. Storage is rescanned after unlink. Reports separate
unique managed bytes removed from last-link allocated blocks released; filesystem
snapshots/compression may affect actual free disk capacity. No retained trash copy
is advertised as reclaimed space.

## Scientific status and limitations

The available coordinate exports are tiny pipeline checks (e.g. AE4/flow4), with
poor representation and reconstruction quality. The screenshot's model is visibly
poor; it is real inference, not a reference copy. Flexible query execution is
implemented; resolution generalization and useful accuracy are unestablished.
The conventional Fourier inverse remains better in earlier evaluations, whose
test performance was non-monotonic across training budgets. No winning budget is
selected using test performance. CUDA and actual Colab execution remain untested.

Reference D supports 0.0001–2 and end times up to 5; neural checkpoints retain
their own exact domain/D/time constraints (typical D .03–.12 and t 0–.15).
No nonperiodic boundary, variable D, forced heat or new PDE is added. Import is
streamed to disk and bounded at 2 GiB (including larger Colab exports); reference grid remains bounded, while display/export frame count is configurable in this
focused UI. Reference/sampling records are saved under workspace `workbench/data`;
jobs and reference-free requests under `workbench/jobs`; training uses its existing
workspace paths. Preferences restore controls, not stale backend IDs.

Interaction inspiration: [VisualPDE heat exploration](https://visualpde.com/basic-pdes/heat-equation/).
No branding, assets or source were copied. PDEFlow retains periodic boundaries.
Verification and screenshots: [heat playground verification](heat-playground-verification.md).
