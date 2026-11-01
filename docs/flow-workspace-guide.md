# Linked mathematical workspace and actual generation

Insert/edit the current equation, review physical parameters/domain/conditions,
and supply measurements manually or through Reference experiment. The complete
reference input and disclosed inverse problem remain distinct. A compatible export
is selected automatically by the server, or explicitly by registered ID; missing
models produce a useful error without discarding the draft.

For a synthetic experiment, Solve reference → Sample measurements → Reconstruct
from measurements → Validate/Run. Results automatically retain and compare the
associated reference. No manual reattachment of that existing artifact is needed.
The association is stored in evaluation-context.json **outside** request.json;
the worker never receives truth. The parent evaluates it only after successful
sampling. Reopening/refreshing a saved result retains comparison. A changed problem
invalidates the current reference link; incompatible associations are rejected and
the old experiment remains saved. Manual reference-free observations still work;
reference errors are then explicitly unavailable. Manual detach suppresses automatic
reattachment. Comparison continues to use the existing coordinate/grid policies.

## Three distinct visual modes

- **Spatial inspection:** linked heatmap and optional scalar height surface, one
  orange selected point, actual native-grid equation terms below. Pointer movement
  or click selects a point; Lock selected point prevents pointer movement. Native
  coordinate index inputs are keyboard accessible and can move the locked point
  deliberately. True3D uses three coordinated orthogonal slices, never a2D surface
  mislabelled as a3D PDE. Reference/reconstruction share scale; error uses its own
  labelled scale. Measured points are orange; unmeasured values are inferred.
- **Physical evolution:** discrete saved physical τ frames with play/pause and
  scrub. There is no interpolation between invented physical states. Static Darcy
  explicitly has no physical-time animation. Fields/species share the existing
  physical-field/time selection. Parent controls also select the weighted mean,
  a saved individual sample or empirical spread. Spread is not a physical PDE
  solution; its local equation residual is intentionally unavailable.
- **Generative evolution:** actual bounded sampler snapshots, initial model noise,
  current model coordinates, decoded current field or clean endpoint estimate.
  Generation s is a different control from physical τ. Captured fields/times,
  particle and3D display slice can be selected. Play/pause/step/scrub never synthesize
  frames by interpolating first/last images. Returned reconstruction remains in
  spatial inspection. Intermediate states are not verified PDE solutions.

FK playback marks ancestry transitions and missing recorded parents. Follow recorded
parent is enabled only when that previous slot was captured. It never presents
unrelated resampled slots as one smooth physical trajectory. Snapshot history is
bounded; no full-resolution history of every particle is retained by default.

Local heat terms use second-order native physical-time derivatives and periodic FD
Laplacian. Darcy terms are actual harmonic face fluxes/divergence and known forcing,
with declared pressure gauge. Chemistry shows species, mass-action reaction and
native FD time/diffusion/reaction contributions. They use the currently selected
mean/sample fields, not evaluation truth. Cached saved arrays and debounced bounded
requests avoid full derivative recomputation on mouse movement. Missing diagnostics
are explained instead of replaced with invented values.

Download the numerical slice/terms and SVG figure. Full experiment ZIP, comparison
JSON, native samples, reference and noisy observations remain exportable. Display
striding is labelled separately from native numerical resolution. Linked views are
≤64² planes; ordinary reference/result preview planes≤128²; native scientific
metrics use the declared full evaluation grid. The optional surface is lightweight
SVG and lazy-loaded with the linked workspace, not a new large3D dependency.
Reduced motion starts in a useful static state; animation is explicit opt-in.

## Clear workspace

Clear workspace confirms before discarding the current editor/observations,
parameters/domain/conditions, reference/model/result selection, validation and
animation/visual state. A running job offers Keep running and clear editor,
Cancel job then clear, or Return without clearing. It never accidentally cancels
on clearing. The component remount invalidates late callbacks; persisted PDEFlow
draft/selected-job/reference keys are removed before the empty editor mounts.
Refresh stays empty even if the old job finishes. Saved jobs, checkpoints,
reference artifacts, training data and history are never deleted. Reset draft
uses the same explicit confirmation. Observation row deletion, equation editing and reference detach retain their narrower scopes.

Design interpretation and inspected animation frames: flow-workspace-design.md.
Actual screenshots, tests and limitations: flow-workspace-verification.md.
