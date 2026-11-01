# Process diagram audit — 25 October 2026

Scope: the user supplied a diagram and an explicit implementation request. The
image is a target process, not evidence that its arrows execute. Existing source
and saved outputs were inspected before changes. This audit distinguishes the
initial state from the resulting implementation. No accepted output was reused
as a new execution claim. New evidence lives in outputs/process-20261025/.

## Boxes

| Diagram box | Intended / actual entry point | Config, inputs → outputs | Tests and execution evidence | Status / remaining limit |
|---|---|---|---|---|
| Full problem specification | symbolic.analyze; workbench.api.resolve; core.ProblemSpec | Full 𝒞 source, declarations, known λ/forcing, domain, boundary/initial roles, measured y/M/noise → validated physical ProblemSpec/ObservationSet | Existing symbolic/routing/API tests; new process pipeline and actual browser requests | Verified for allowlisted scalar periodic heat/Darcy/chemistry. Unknown initial state is not a known zero. Partial initial data must be point observations; unsupported walls/geometry/vector execution rejected. |
| Equation and geometry encoding | specialist.request_problem, Conditioning.from_problem, model encoder | Current AST/roles/domain → heat scalar D, Darcy known f map, chemical known D/rates, fixed coordinates/layout | Existing current-source/wrong-family regressions; actual three-family process runs | Verified narrow adapters, not learned generic equation/geometry understanding. |
| Select compatible specialist | api.parse_mathematics/discover/resolve; inference.validate_compatibility; sampling._sample_once | Current request + checkpoint content ID, exact grid/times/representation/formulation → compatible bundle or descriptive error | Existing routing tests/browser; new refinement compatibility tests and API worker content checks | Verified. API uses server-owned paths, never client-supplied checkpoint paths. Lexical compatible content-ID selection is not scientific model selection. |
| Conditional generation | sampling.sample → process.run → _sample_once; generative.LinearCFM; Euler/Heun | Seed, known conditions, trained export, actual observations through optional steering → independent candidates | New heat/Darcy/chemistry process runs, true3D; previous smoke artifacts inspected | Verified conditional flow and heuristic steering; later FK adds the explicitly documented numerical SDE tilt. No diffusion/CFG claim. |
| Physics and observation evaluation | process.diagnostics.report; specialist.physics; ObservationSet/PointOperator; evaluation.metrics | Decoded physical fields + request/measurements → component reports and optional reference-only errors | Existing numerical manufactured/convergence/gradient tests; new report/mask/noise tests and process traces | Verified on supported discretizations. Steering/final component reports share FD/flux operators; final metrics additionally use spectral space for transient families, not an independent continuum certificate. |
| Next action | process.ProcessConfig/ProcessState/run | Phase, component failures, capabilities and remaining budgets → legal action and reason | Deterministic constructed policy tests plus real traces; browser process workflow | Initially absent; now bounded and verified. Thresholds are explicit per component; absent thresholds cannot yield criteria met. |
| Resolution refinement | refinement.refine_request; process.run; CLI --fine-checkpoint; server-owned fine_model_id | Nested 2× nodal heat grid, same times/domain/λ/y → checked fresh Gaussian fine reconstruction | Refinement transfer/compatibility tests; real heat-fine-smoke/heat-refinement/CLI outputs and browser workflow | Initially absent; now one global spatial level for heat. No temporal refinement, Darcy/chemistry transfer, learned bridge or local AMR. Interpolation utilities are diagnostic only. |
| Reconstructions and diagnostics | evaluation.export_result; API result/slice; frontend ProcessPanel | Actual returned fields/grid, budgets/history/stop reason → NPZ, JSON, figures and durable UI | Serialization/reopen/reference-isolation tests; actual CLI/API/browser artifacts | Verified. Finished execution and criteria-met are separate. Budget-limited fields may be incomplete generation states and are labelled accordingly. |

## Every directed connection

The following shares the same entries/configuration and test evidence as the
referenced boxes; it explicitly records what crosses each arrow.

| Arrow | Actual connection, inputs → outputs | Initial → final status; specific limit |
|---|---|---|
| Full specification → encoding | API reparses mathematical_request and compares resolved ProblemSpec and measurement rows/noise with submission | Verified → verified. Draft/preset labels cannot override edited equations. |
| Encoding → compatible specialist | analyze calls registered recognition/request adapters; API validates checkpoint exact layout/conditions | Verified → verified; unsupported ASTs block execution while drafts remain editable. |
| Compatible specialist → conditional generation | validated bundle/layout and known condition enter _sample_once; worker checks export content ID | Verified → verified. Training quality remains a separate capability. |
| Encoding → physical evaluation | same canonical ProblemSpec/operator/known λ builds specialist.physics and measurement operator | Verified → verified; no generic residual fallback returning zero. |
| Generation → evaluation | decode endpoint estimate at configured checks; decode actual state at final verification | Partial final/gradient evaluations → verified explicit labels, component reports and call/budget accounting. |
| Evaluation → next action | report mandatory component statuses/finite flags/history passed into deterministic ProcessState | Absent → verified; no reference argument or accuracy metric in policy. |
| Next action → steer/continue | intermediate decision toggles configured correction for subsequent intervals; patience before switching it off | Absent → verified. Never early-finishes an estimated endpoint or invents a new time/noise position. |
| Steer/continue → generation | ordinary Euler/Heun proposal then optional negative endpoint-gradient split correction | Fixed guidance only → verified feedback-controlled configured guidance. Schedule still ends at s=1. |
| Next action → resample → generation | fk.pde.run → fk.simulate → fixed eligible ESS check → systematic joint gather/weight reset; process RESAMPLE records coordinate/parents/ESS/budget | Absent → implemented/verified in the FK continuation. Core analytic targets, real PDE/API/browser/MPS traces in fk-verification.md. Gradient independent retries remain distinct. |
| Next action → increase detail | REFINE only at completed candidate boundary with mismatch, level/attempt budget and compatible fine export | Absent → verified heat global-grid branch; explicit unavailable reason otherwise. |
| Increase detail → resolution refinement | fine physical shape must be exactly twice coarse shape; remap node indices ×2 preserving y/masks/noise/coordinates | Absent → verified. Same physical times only. |
| Resolution refinement → reselect specialist | refine_request calls validate_compatibility again; worker loads/checks server-owned fine content ID | Absent → verified. Wrong family/grid/times/domain rejected before fine sampling. |
| Reselected specialist → fine generation | new source RNG seed+1 and fine model; coarse candidate is neither condition, measured pseudo-data nor Gaussian source | Absent → verified. This is a new compatible inference, not conditional recovery of coarse missing frequencies. |
| Next action → finish → reconstructions/diagnostics | actual final report gates FINISH_CRITERIA_MET; otherwise exhaustion/failure/cancellation recorded | Absent → verified. Returned grid/problem/observations and stop reason are persisted. |

## Current truth of the diagram

The gradient-feedback loop, one global heat refinement route and the later
mathematically specified FK branch execute. FK uses its own fixed legal
resampling policy; FK plus refinement/retries remain explicitly unsupported.
The entire architecture is not claimed universally complete. Arbitrary geometry, off-grid/time interpolation,
vector-valued execution, correlated observation noise and temporal/other-family
refinement remain absent. Numerical consistency does not certify reconstruction
accuracy, identifiability or posterior calibration. See process-verification.md
for formulas, actual commands, action traces and final verification outcomes.

## FK arrow qualification

Compatible specialist → generation now also validates independent-Gaussian linear
CFM stochastic capability and an explicit FK config. Generation → evaluation
uses named physical endpoint energy at intermediate nodes and actual fields at1.
Evaluation → RESAMPLE uses fixed eligible steps plus ESS, never hidden truth or
arbitrary policy-reset weights. RESAMPLE → generation jointly gathers fields,
parameters, previous potential and root ancestry; future noise is independent.
Finish → results carries terminal weights/completion separately from physical
criteria status. Increase detail → refinement still belongs to the legacy
gradient path; combining it with FK rejects before execution. This new route was
verified on CPU and actual MPS, including true3D; CUDA/Colab remain untested.
