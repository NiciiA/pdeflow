# Workbench visual API

The existing reconstruction validation/job/artifact routes remain authoritative.
Frontend validation is additional feedback; the API independently validates the
current canonical mathematics, specialist, checkpoint layout, conditioning,
boundary, observations and device before starting a worker.

| Route | Input / meaning | Limits |
|---|---|---|
| POST /api/input-preview | `problem`, editable `rows`, `noise_std`; optional `field,time_index,axis,index`. Reads supplied arrays and exact on-grid observations only. Unknown `values:null`; row-indexed `issues` preserve invalid data. | 262144 grid nodes, 128 displayed nodes/axis, 512 markers; explicit downsampling flag |
| GET /api/jobs/{id}/input-preview | Same selectors, immutable saved problem/observations. Requires completed job. | Same bounds; no evaluation reference |
| GET /api/jobs/{id}/slice | `field_index,time_index,axis,index,sample`; -1 weighted mean, -2 weighted empirical spread, nonnegative actual particle. | One plane, 65536 nodes, bounded observation markers, no full ensemble JSON |
| GET /api/jobs/{id}/plots | Actual observed/predicted/residual points from weighted population mean; saved estimated/final diagnostics, recorded generation positions and measured cost. | 256 observations, last 512 process rows; missing positions stay null |
| GET /api/training-runs | Actual managed run metadata/history and checkpoint-event positions; no curve inferred from an external model export. | 50 most recently started runs; extrema-preserving bounded loss rows, no smoothing |
| GET /api/training-presets | Actual packaged training configurations and profile status. Read-only. | Eight current named presets |
| POST /api/compare | `ids`: 2–4 distinct completed jobs. Hashes actual ProblemSpec plus serialized observations/noise; incompatible tasks return 422. Reference accuracy is included only for identical attached reference arrays in every run. | Quality/cost summaries, no automatic ranking or budget selection |

Reference attachment/removal recomputes evaluation accuracy separately and cannot
change saved particles, weights, input previews or observation-prediction plots.
Generation diagnostics without recorded coordinates remain unavailable rather
than receiving invented coordinates. Existing exports retain all numerical data
and exact resolved settings; plots are explanatory views, not new solvers.

## Reference and suite routes — 26 October 2026

| Route | Contract |
|---|---|
| GET /api/reference-capabilities | Declared family/dimension/solver/input support; no neural model |
| POST /api/reference-input-preview | `{request,inputs,solver,accuracy,seed,field,axis,index}`; safe parser, ≤65536-node plane |
| POST /api/references | `{request,inputs,solver,accuracy,seed}`; immutable new ID; current equation compiled independently |
| GET /api/references/{id} | Metadata/provenance, known spatial arrays redacted in public summary |
| GET /api/references/{id}/slice | field_index,time_index,axis,index; optional sampling_id overlays ≤512 exact matching measured positions |
| POST /api/references/{id}/sample | `{sampling:{pattern,count,fields,time_indices,seed,noise_std,positions,reuse_indices,disclosure}}`; noisy inverse mathematical request/rows and sampling ID |
| GET /api/references/{id}/artifacts/{name} | Reference NPZ/full input/manifest; with sampling_id: noisy CSV/JSON, inverse request, sampling manifest. Name and IDs allowlisted |
| POST /api/jobs/{id}/reference | Existing post-run route adds policy=native/restrict_reference, shared_times_only and metric names. Optional sampling_id must match actual saved inverse observations; omitted ID resolves a matching sampling artifact when available |
| GET /api/jobs/{id}/artifacts/comparison.json | Full EvaluationReport after attachment |
| GET /api/jobs/{id}/experiment.zip | Request/results/baselines/reference/matching sampling/clean evaluation-only values plus manifest; no inference changes |
| GET /api/training-suites | Last30 durable suite records, child progress/paths/resume; no execution |

The existing worker ReconstructionRequest strictly forbids reference_id, complete
reference inputs, clean values and evaluation/display controls. API independently
validates current equation/checkpoint/conditioning. Reference arrays/solver work
are bounded to2 million field-time nodes; previews never send complete3D arrays.
Legacy synthetic references remain attachable read-only, with missing detailed
solver provenance labelled as a historical limitation. New artifacts validate
specification, physical metadata and array checksums. JSON schemas and disclosure/
transfer limits are documented in reference-experiments.md; suite state transitions
and locks in training-suites.md.

## Explicit selection and linked workspace continuation

POST reconstruction/validate and jobs accept
`model_selection:{mode:"auto",id:null}` or `{mode:"explicit",id:"32hex"}`.
Legacy valid model_id is accepted separately; empty IDs and combining contracts
reject. Server validates/resolves current physical request before queue creation;
resolved model_id and selection rationale persist in request/job records. Human
error.message is separate from request_id/details with field path/code/message.
The reference context is excluded from this worker request.

| Route | Meaning |
|---|---|
| POST /api/jobs/{id}/evaluation-context | Reference ID/sampling/evaluation settings, independently stored and compatibility checked. Parent attaches after successful inference; reopen retains it. |
| GET /api/jobs/{id}/inspection | field_index,time_index,native comma-separated point,sample_index(-1 weighted mean,-2 spread,nonnegative particle). Cached actual arrays,≤64² orthogonal planes, exact observations and native local terms; spread has no PDE residual. |
| GET /api/jobs/{id}/generation | No frame: bounded capture metadata. With frame,particle,kind(current_model/current_physical/endpoint_physical),captured field/time and optional display slice: actual saved state, not interpolated frames. |
| GET /api/models | Adds actual/derived architecture configuration, parameter count and exact resolution policy. Legacy original weight architectures remain intact. |

Reference/result display planes now stride explicitly to≤128² rather than refusing
native heat1024². Native numerical arrays/coordinate metadata and scientific metrics
are retained; generation capture is bounded to16MiB. Full ensembles are not sent as
JSON. No physical-time control is supplied for static Darcy.

## Catalogue, monitoring, analytical examples and storage

| Route | Contract |
|---|---|
| GET /api/training-coverage | Versioned configured targets, actual partial/completed effort, managed inverse-evaluation availability, discoverable exports and obsolete target records |
| GET /api/training-plan?suite=full | Pure plan; resolved frozen target list and budgets, no jobs |
| GET /api/training-runs | Existing history plus kernel ownership, heartbeat/live progress, actual architecture, checkpoint time, data mode/counters, timing, configuration and bounded log tail |
| POST /api/checkpoint-compatibility | `{problem}` → every registered candidate's expected/actual mismatches and remedies |
| POST /api/analytical-examples | `{problem}` → domain-scaled editable heat initial/reference templates; no layout changes |
| POST /api/analytical-verify | ReferenceSolveRequest → finite PDE/initial/periodic checks; inconsistent candidate report does not create a reference |
| GET /api/storage?refresh=false | Cached stat inventory; explicit refresh rescans. No file content/weight scan on routine UI refresh |
| POST /api/storage/preview | Saves exact reviewed-plan candidate paths/sizes; no mutation/deletion. Uses the cached inventory; apply revalidates |
| POST /api/storage/pin | `{path,pinned}` relative to configured inventory root; containment/symlinks checked |

Mathematics parse routing now includes structured candidate compatibility reports
and matching current target statuses/commands. Automatic selection prioritizes
actual optimizer updates, completed-budget provenance, then export role among
compatible candidates; no reference error, training-loss cross-model ranking or
quality guarantee. Inference independently revalidates all contracts.

Storage inventory's own .storage metadata/quarantine is outside public artifact
totals; quarantine does not free disk bytes. CLI apply requires the exact reviewed
plan digest and only permits registered, unchanged stale disposable caches after
fresh protection/ownership checks and affected-run locking. There is no API training
launch or automatic cleanup apply. See training-catalogue.md.


## Coordinate specialist and full reference imports — 27 October2026

POST /api/references/import accepts {artifact: full-field JSON}; GET
/api/references/ID/artifacts/reference-field.json exports the matching format.
Strict explicit coordinates/times/fields/units prevent silent axis inference or
extrapolation. Existing sampling accepts noise_mode absolute/relative_rms and
returns actual_sigma/row sigma; observations.json carries measurement_noise_std.
Reference slices report visible and off-slice counts.

GET /api/coordinate/runs reads real stage manifests/checkpoints and kernel liveness.
POST /api/coordinate/validate and /api/coordinate/reconstruct accept a registered
model_id, physical problem, scalar rows (field=u,x/y/z,time,value,sigma), optional
ordered queries, steps/particles/seed/chunk_size/device and optional current
mathematical_request. That source is re-parsed with the coordinate_point_set
capability; family/operator/fields/physical parameters/domain must match. Only
query discretization/time counts may change within supported ranges. Numeric JSON
0 and0.0 compare equivalently. Unsupported equations reject independently of UI.
Reference/initial arrays are forbidden extra input keys. Canonical ProblemSpec
clients still receive full physical/checkpoint/coordinate validation.

The existing single JobManager dispatches a coordinate worker. request.json has
only reference-free inference inputs; mathematical-source.json is separate parent
provenance. Generic job status/cancel remain shared. Coordinate jobs reopen in the
coordinate panel; the original fixed-grid RunComparison selector excludes them.
GET /api/coordinate/jobs/ID/result returns metrics and the first256 ordered query
rows; /slice provides real regular-grid planes/mean/sample/spread. POST /compare
accepts only reference_id after success, storing evaluation arrays separately.
Artifacts result.npz/metrics.json/comparison.json/problem.json/observations.json
are available through the corresponding coordinate artifacts route. Scattered
queries are exported as ordered points rather than invented heatmaps.


## Integrated training routes

GET /api/training/options reports supported choices and actual backend availability.
POST /api/training/plan validates/resolves settings without training; POST
/api/training/start adds a submission_token and launches a persistent worker.
GET /api/training/runs and /api/training/runs/{id} report saved progress/logs.
POST /api/training/runs/{id}/stop requests a safe update-boundary stop; resume
accepts an explicit allow_poor_representation flag. POST .../{id}/extend accepts
additional_updates for a completed fixed-grid run and preserves a budget snapshot.
GET .../{id}/preview returns validation field/reference/error slices.
POST /api/training/models accepts optional current problem for compatibility;
POST /api/training/use selects an artifact and accepts explicit
adopt_training_problem. GET /api/training/selection exposes the explicit choice.
No import/page-load endpoint trains. Both UI and CLI call training_workflow.py;
see training-workflow.md for persistence, stages, gates and scientific limits.
