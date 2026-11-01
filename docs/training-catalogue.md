# Training catalogue, coverage, monitoring and storage

## Current status and compatibility

The finite `pdeflow-training-v2` catalogue has 11 targets. CFM is the only trained
generative method. Both architectures in this fixed-grid catalogue operate on direct fields, with fixed saved
physical times packed into channels. This catalogue has no diffusion training or latent
AE/VAE dependency. Heat, conservative periodic Darcy and reversible A+B↔C
chemistry use the common online/hybrid/offline provider.

The user's old local-pilot suite completed all six original targets at 3000/1500
updates. It did not include later capacity/high-resolution targets. Its chemistry
export is valid at 16²/T4. An 8² request is incompatible. Supported equivalent
chemistry equations, including r=k_f*a*b-k_r*c and explicit substitution, now
normalize through bounded polynomial rules. Genuinely different kinetics reject.
Species names remain explicitly a,b,c with the declared D_A,D_B,D_C,k_f,k_r;
arbitrary renaming is not implemented. Structured routing reports expose expected
and actual family, operator, fields, dimension, shape, domain, boundary convention
and physical-time differences, plus remedies. Known conditioned parameter values
are not exact model identities; recorded training ranges still constrain claims.

The original failed chemistry draft was not recoverable from saved jobs/current
browser state. Reproduced same-family layout and kinetic failures, and unrelated
wrong-family aggregate-error clutter, establish the mechanisms without pretending
the exact original draft was recovered. See catalogue-findings.md and the saved
compatibility report for the actual requests and exports examined.

## The finite full catalogue

| Target | Architecture | Native grid | Physical τ | Updates/batch |
|---|---|---|---|---|
| heat-2d |CNN width24,3 blocks|32²|0,.05,.1,.15|3000/2|
| darcy-2d |CNN width24,3 blocks|32²|static|3000/2|
| chemistry-2d |CNN width24,3 blocks|16²|0,.05,.1,.15|3000/2|
| heat-3d |CNN width16,3 blocks|16³|0,.05,.1,.15|1500/2|
| darcy-3d |CNN width16,3 blocks|16³|static|1500/2|
| chemistry-3d |CNN width16,3 blocks|8³|0,.05,.1,.15|1500/2|
| heat-2d-medium |U-Net width16,3 levels|64²|0,.05,.1,.15|3000/1|
| heat-2d-large |U-Net width24,4 levels,2 blocks|128²|0,.05,.1,.15|3000/1|
| heat-2d-512 |U-Net width8,4 levels|512²|0,.05,.1|10000/1|
| heat-2d-1024-cuda |U-Net width8,4 levels|1024²|0,.05,.1|10000/1|
| colab-heat-2d |Distinct CNN accelerator profile|64²|0,.05,.1,.15|10000/16|

All use periodic unit Cartesian domains, known specialist conditioning and an
explicit physical_v1 training distribution. `pdeflow catalogue --json` prints every
resolved config, architecture/parameter count, convention, distribution/ranges,
expected checkpoint capability, budget and resource profile. Original preset
quickstart-heat-2d (8²) and heat-2d-native-cpu are utility checks, not additional
substantive catalogue targets. Their exports remain usable for their exact layouts.

`train-all` now defaults to full/all. `local-pilot` explicitly retains the six
original targets; `quickstart` retains those same layouts with two updates/batch1;
`capacity-local` explicitly selects medium64² then large128². Full is a workload
specification, not evidence that every target has trained or fits every device.
Only the intended backend or an explicit override executes; no hidden coarsening.
Without --device, each target retains its declared resource policy. On one MPS
machine, the explicit MPS override runs targets sequentially, including the
narrow 1024² profile; the colab batch16 profile is a distinct potentially costly
workload. Resource failures are recorded, not silently repaired by changing grids.

```sh
pdeflow catalogue
pdeflow catalogue --json
pdeflow coverage
pdeflow train-all --plan
pdeflow train-all --device mps --plan
pdeflow train-all --device mps
pdeflow train-all --device mps --missing --plan
pdeflow train-all --device mps --missing
pdeflow train-all --suite local-pilot --device mps
pdeflow train-all --include heat-2d-medium --force --device mps
pdeflow suite-status latest
pdeflow resume-suite latest
pdeflow resume-suite latest --retry-failed
```

Always use the same --workspace if selecting a custom persistent root. Planning
never launches training. Execution prints targets before preparation and freezes
its versioned target list/configs. Fail-fast is default; --continue-on-error is
explicit. --missing omits only successful compatible completed budgets. --reuse
records completed matching child references; --force explicitly retrains selected
targets. --missing and --force conflict. A mere inference.pt file, partial run,
failed manifest, corrupt export or shorter budget cannot satisfy a full target.
Seeds/device do not change capability; optimizer batch/learning rate and recorded
distribution matter for completion. Safe CPU-loaded checkpoint tensors are checked against the actual model factory
on the meta device (keys, shapes and architecture metadata) before reuse. Live
inspection does not construct full CPU models or initialize thread pools. Resume still requires recorded implementation/config identity. Do not edit
Python source while a run is interrupted; restore its recorded source or launch
a new run. Source updates alone do not make an old inference model incompatible.

States: pending means a frozen child has not started; interrupted means a child
stopped before budget or its owner exited; partial means a compatible run/export
has insufficient target effort; failed means execution recorded an error; completed
means verified target budget and valid inference export; obsolete target means a
named run's architecture/distribution/layout differs from the current target;
incompatible means the current request/export physical contract differs or export
validation fails. Evaluated means a completed evaluation artifact exists, not that
quality is good. Unsupported layouts/methods have no catalogue cell/claim.
Old completed suite snapshots retain original success and show current targets
not present in that snapshot. Coverage across all runs is reported separately.

## Native 512² heat on the MacBook

```sh
pdeflow train --preset heat-2d-512 --device mps
pdeflow status latest
pdeflow resume latest
pdeflow evaluate --run latest --device mps
pdeflow experiment --problem configs/reference/heat-512.json --model auto --inference configs/inference/heat-512.json --output outputs/my-512-experiment --device mps
pdeflow compare --result outputs/my-512-experiment/reconstruction --reference outputs/my-512-experiment/reference --output outputs/my-512-comparison.json
```

This is actual 512²,T3,unit periodic endpoint-excluded nodes, batch1,float32,
activation-checkpointed width8/4-level U-Net,condition width16 and coordinates:
73,939 parameters. D is sampled in [.03,.12]. Initial fields use the recorded
smooth random-field generator (bandwidth1.2, amplitude1,offset0); all resolved
settings are inspectable. Four training-only calibration samples freeze scaling;
validation/test each have four independent fixed cases. Validation and atomic
checkpointing occur every100 updates and at the final/interrupted boundary.
No full training dataset is saved in online mode. Native fields are never enlarged
small-grid predictions. The finite receptive field is not one global 512² context;
iterated ODE propagation and deterministic coordinate features do not establish
learned global quality. Guidance additionally differentiates the model/physical
decoder, so population and step counts affect memory. No independent patch tiling.

The bounded MPS probe actually executed update, guidance and two-step source-to-
candidate sampling at this layout. Long 10000-update training remains user launched.
Its matching reference JSON is supplied. In the editor, Load 512² heat example
opens a before/after review. Use model-compatible layout similarly reviews a
chosen catalogue grid/domain/times while preserving equation, physical parameters
and observations. Existing observations need revalidation after any layout change.
No equation or saved-time change is silently applied. In Reconstruction, expand
Feedback controller and budgets and explicitly choose Use requested native grid
budget (262144), or disable feedback. The existing default controller budget is
32768 nodes; API validation now rejects an undersized controller before submission.
This reviewed limit is separate from actual device-memory feasibility.

## Monitoring

```sh
pdeflow runs
pdeflow status latest
pdeflow suite-status latest
pdeflow models
pdeflow serve
```

Open Models & Training at http://127.0.0.1:5173 with the frontend running. It reads
CLI and UI-visible filesystem artifacts, polls coverage/progress every5 seconds,
shows active ownership, current/target update, family/native grid, actual model
parameters, device/dtype,data mode, unique consumed problems, measured throughput,
checkpoint update/time, CFM curves, actual validation positions, cumulative data
wait/neural host time, configurations/logs and copyable commands. ETA is unavailable
until20 measured updates and10 seconds. Legacy missing telemetry is unavailable,
not fabricated. Kernel run locks determine active status; heartbeat/progress files
provide freshness. Closing the browser does not own or terminate training.

## Storage and safe retention

```sh
pdeflow storage
pdeflow storage --refresh
pdeflow cleanup --preview
pdeflow pin local/runs/RUN_ID
pdeflow pin local/runs/RUN_ID --unpin
```

Default inventory root is the repository outputs directory. Use --root PATH for
a configured managed root; pin paths are relative to that root. The UI uses the
same output root for repository-contained workspaces, or an external workspace as
its own root. Inventory is cached stat metadata with explicit refresh, per-file
logical sizes/path-inferred types/times, related run/suite, pin/reference/active/protection flags.
It rebuilds from actual files/manifests and does not read every checkpoint payload
on each storage refresh. References are resolved conservatively; scientific files
remain protected even when an indirect dependency cannot be inferred.

Cleanup previews use the cached inventory and save its scan timestamp, exact
paths/sizes and a review digest. Refresh storage explicitly when needed; applying
a plan always rescans and revalidates current protection. Only explicitly
producer- or user-registered, unchanged recognized disposable caches older than24h qualify;
filename alone never assigns ownership to user files. Historical caches and all
scientific results, checkpoints, provenance, unknown files, QA images and archives
are protected by default. This conservative implementation does not automatically
reclaim old accepted QA or scientific trees. It does not relocate old runs.

`pdeflow cleanup --register-cache RELATIVE_CACHE_PATH` explicitly records ownership
of a reviewed, recognized disposable cache; it only writes metadata. Registration
does not bypass age, content, active-run, pin, reference, symlink or Git protection.
Pin/registration metadata and application serialize against concurrent changes.
Inventory schema2 records actual birth time where available (otherwise null),
modification time and suite relationships, including sibling suite child runs.

After a human reviews a concrete preview, an explicit terminal action can use:
`pdeflow cleanup --apply PLAN_ID --confirm REVIEW_DIGEST` (same --root when needed).
It revalidates root containment, symlinks, pins, references, ownership, liveness,
Git tracking and unchanged content, acquires affected run locks and moves only
that plan into .storage/quarantine/PLAN_ID with recovery.json. There is no permanent
delete operation. Quarantine bytes remain on disk until a separately authorized
manual decision. No cleanup apply/quarantine was run on user outputs in this task;
only temporary test fixtures exercised mutation. Current atomic checkpoint policy
retains latest resumable state and latest/validation-selected exports; no historical
checkpoint pruning or automatic source-archive regeneration was introduced.

## Analytical examples and learning diagnostics

In Reference experiment expand Analytical example. Select product (2D or3D),
single-axis, two-mode or labelled higher-frequency challenge; inspect the proposed
u0 and u_ref then replace reference inputs explicitly. Edit their separate fields,
verify, solve, sample and reconstruct. The domain/time layout remains unchanged.
Known D is edited in the main problem; initial/reference fields stay evaluation-only.
Candidates are labelled and inconsistent formulas reject; exact supported templates
receive finite numerical verification, not an unrestricted symbolic proof.

```sh
pdeflow diagnose-learning --checkpoint /path/to/inference.pt --output outputs/my-learning-diagnostic --device cpu --updates 100 --cases 2
```

This bounded command supports physical-provider exports up to16384 native spatial
nodes (3D axes<=16); it never silently downsamples a larger model. It copies the
model for fixed-data/source/time overfit, leaving original weights intact, runs an
independent tractable Gaussian toy, compares generated/reference distribution
means/variances/radial spectra/physical-time RMS, and uses three fixed validation
seeds ×4/8 integration steps ×unguided/observation/observation+PDE against the
conventional baseline where implemented. Counts, actual parameters, optimizer
updates, memory/time, guidance settings and all errors are recorded. Repeated
reference arrays are not independent new physical cases; fixed validation IDs
and case counts define the effective reference sample size. This is a diagnostic,
not an automatic full scientific study or proof of a single cause of poor quality.

Lecture concepts and implemented/background/optional status, with actual source
pages, are mapped in generative-foundations.md. No ELBO/KL was appended to CFM.

The separate optional AE/latent-coordinate heat stages are explicitly selectable
with coordinate-train and are not silently added to train-all. See
[coordinate-specialist.md](coordinate-specialist.md).
