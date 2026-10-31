# Capability matrix

## Active operator-aware physical-state v2 —30 October2026

| Capability | Implemented / available / smoke-trained | Scientific status |
|---|---|---|
| One shared periodic heat/wave2D model | Real shared A1/A2/B, typed u/v/D/c masks and heads; final two-update export available | Poor accuracy, no joint-training advantage demonstrated |
| Directed typed PDE graph | Ordered argument edges, type/unit/order checks, safe registered parser, permutation/forward-gradient tests | Structured conditioning; not symbolic solving or novel-PDE generalization |
| Training equation selection | Visible independent Heat / Wave / joint selector; actual API/data/stage metadata and gradient audit | Execution verified on tiny CPU budgets; no accuracy claim |
| Learned physical-time evolution | Complete wave state, identity/static scalar, decoded/latent alignment and composition | No exact semigroup/conservation guarantee |
| Wave numerical reference | Analytic/zero/Nyquist/energy/non-square/arbitrary-time checks | Finite spectral reference, not learned prediction |
| Known/unknown D and c | Real API/browser through same checkpoint; explicit-noise CPU/MPS inference | Potentially ambiguous; scalar accuracy unestablished |
| UI displacement/velocity | Independent drawing/presets, selected-quantity observations/profiles/errors, coherent time queries | Predictions remain smoke quality |
| Region-balanced objective | Actual A1/A2 decoded gradients, separate signed u/v regions; α=.5 new-run default | Experimental; no improvement demonstrated |
| Portable inference | Training-only state encoder/data omitted; graph/evolution/flow/decoders embedded | CPU/MPS executed; CUDA/actual Colab untested |
| Capacity ladder | Smoke through XLarge, exact total252.7M Large /644.3M XLarge counts | Large hardware fit/throughput/quality unmeasured |
| Static/weak/forced/new boundaries or neural1D/3D | Explicitly unsupported in this specialist | Future modules; historical numerics remain |

Detailed contracts/counts: operator-wave-model.md. Actual saved evidence and limitations:
operator-wave-verification.md. The tables below retain historical research records;
they do not declare old generative weights compatible with the active workflow.



## Historical joint-v1 generator —30 October2026

| Capability | Implementation / evidence | Scientific status |
|---|---|---|
| Structured joint u/u0/D latents | Shared global/local groups, typed conditioning, frozen AE + masked equal-group CFM; CPU/MPS tiny execution | No accuracy/identifiability validation |
| Known/unknown D in same heat2D model | Actual browser reconstruction in both modes; no full hidden initial/D conditioning | Ambiguous sparse inverse problem; bounded prior |
| Continuous physical time and coordinate queries | Preserved [0,1], fixed latent across frames; periodic/chunk/derivative tests | Resolution flexibility, not demonstrated fine-detail recovery |
| Region-balanced field objective | Real encoder/decoder gradients, alpha=.5 new UI/Colab default, scalar/latent losses separate | Experimental; no demonstrated improvement |
| Portable decoder/flow export | No training encoder/data dependency, CPU-first reload and explicit-noise MPS agreement | CUDA/actual Colab untested |
| Large capacities / Custom | Exact meta counts through XLarge465,820,675 total stage parameters; no capacity cap | Hardware fit and useful training unmeasured |
| Optional decoded auxiliary / correction | Explicit real-gradient tests; frozen decoder; known groups fixed; uncorrected outputs retained | No posterior guarantee or scientific benefit study |

Detailed source and execution limits: joint-latent-model.md and
joint-latent-verification.md. The tables below retain historical implementation
and scientific evidence. Their old generative checkpoints are incompatible with
the current normal workflow; numerical reference functionality remains available.

## Historical research capabilities

Evidence refers to the source in this repository and docs/verification.md.
"Trained" below means actual bounded optimizer execution, not scientific
accuracy. "Numerically verified" means analytic/convergence/gradient checks,
not a learned benchmark. A bounded two-case, one-seed heat pilot is now evaluated empirically; robust scientific performance and calibration remain unestablished.

| Capability | Implemented | Numerically verified | CPU smoke / trained | Scientific evaluation | Evidence or next work |
|---|---|---|---|---|---|
| Serializable contracts, known/unknown roles, physical/model layout | Yes | Roundtrip and invalid-case checks | 1D–3D pipelines | No | tests/unit/test_contracts_observations.py |
| Periodic Cartesian FD gradient/divergence/Laplacian | 1D–3D | Second-order analytic convergence | Used in guidance | No | tests/numerics/test_periodic.py |
| Spectral Laplacian and exact Fourier heat reference | 1D–3D | Analytic modes and mass | 2D and true 3D datasets | No | Independent NumPy reference vs tensor FD |
| Periodic Poisson, strict compatibility, zero-mean gauge | 1D–3D | Manufactured solution | Numerical only | No | Neural Poisson inference planned |
| Data IDs/seeds, complete-trajectory splits, training-only transform | Yes | Integrity, disjointness, roundtrip | 2D/3D smoke | No | Integration tests; manifest SHA256 |
| On-grid point observations, masks, Gaussian noise | Yes | Index/coordinate/gradient checks | Spatial sensors at all saved times | No | Off-grid interpolation planned |
| Arbitrarily indexed saved space-time points | Operator supports them | Manual sparse index tests | Default generator uses full-time sensors | No | No separate sparse-time benchmark |
| Scalar expression residual evaluation | Limited safe grammar | Heat/Poisson expressions | Heat residual steering | No | Vector grammar and expression neural encoder planned |
| Conv1d/2d/3d residual CNN, fixed time channels | Yes | Endpoint Jacobian checks | 1D/3D small integration tests; 2D/3D smoke training | No | No time/resolution generalization claim |
| Independent Gaussian linear CFM | Yes | Path endpoint identities | 30 updates 2D; 20 updates 3D | No | Path-specific score/corrected FK SDE now verified; no diffusion training |
| Checkpoint/export/CPU reload/resume | Yes | Bitwise same-backend resume; dataset removed before inference | Both CPU profiles | No | Accelerator-tag portability is metadata simulation only |
| Euler/Heun ODE and uniform/quadratic schedules | Yes | First/second-order and invocation counts | Heun uniform in smoke | No | Euler/quadratic separately verified in tests |
| Unguided, observation, observation+PDE sampling | Yes | Full endpoint gradients vs finite differences; controlled loss decrease | All three in 2D/3D | No | Heuristic corrections, no posterior guarantee |
| Independent particle/candidate axis | Yes | Candidate invariance to population size | Two candidates per smoke problem | No | FK core now verifies independent B and interacting P; PDE adapter B1 |
| Fourier-basis conventional inverse | Yes | Known basis recovery 1D–3D | 2D/3D held-out cases | No | Truncated basis mismatch recorded |
| Physical metrics and figures | Yes | Used in integration tests; figures visually inspected | Four comparisons per CPU profile | No | Single-case coverage is descriptive only |
| CPU runtime | Yes | float64 numerics, float32 training | Executed macOS arm64 | No | Source/run manifests record actual environment |
| CUDA / MPS runtime selection | Yes, guarded | Availability/precision guards | CUDA untested; MPS float32 executed | Bounded MPS regression | Six updates, endpoint/PDE gradients, export and CPU reload with host Metal access |
| Colab package launcher | Yes | Cells compile; actual archive extraction/cache/path checks | Colab/CUDA untested | No | Disposable data cache, persistent data/checkpoints, portable export |
| Variable-coefficient Darcy coefficient/pressure inference | Implemented periodic 2D/3D | Independent manufactured second-order convergence, harmonic flux/gauge/compatibility, positive transform and gradients | 20 updates 8²; 12 updates true 8³; all three guidance modes | No; restricted inverse benchmark implemented | tests/numerics/test_darcy.py; tests/integration/test_darcy_specialist.py; multispecialist.md |
| A+B reversible C chemical reaction–diffusion | Periodic2D/3D implemented | Independent ODE/diffusion/refinement/positivity/invariant/gradient checks | 8 updates2D;4 updates true3D; actual MPS preflights | Smoke only, no scientific accuracy claim | chemistry.md; common provider and real API/browser |
| Waves / Burgers / incompressible flow / Fokker–Planck | Planned | No | No | No | Dedicated family formulations and verification |
| Diffusion training/sampling and CFG | Planned | No | No | No | Explicit compatible noise/prediction process and trained dropout required |
| FK weights/resampling/stochastic dynamics | Implemented EM/systematic | Gaussian and mixture analytic targets, four seeds/budget grid | CPU and actual MPS PDE integration | One negative heat smoke comparison; no robust accuracy/calibration | fk-mathematics.md; fk-verification.md |
| Global refinement / local AMR / new geometry or boundaries | Planned | No | No | No | Conservative transfers, compatible checkpoints and measured convergence |

| Continuation capability | Status | Evidence / limitations |
|---|---|---|
| Absent/correct/unrelated reference isolation | Verified bitwise on CPU | Three-reference regression across all three sampling modes and inverse baseline |
| Physical-coordinate CSV/JSON import/export | Implemented and tested | On-grid saved-time rows, finite/noise/duplicate/field validation; interpolation rejected |
| Local API and persistent job queue | Implemented and integration-tested | Real subprocess, idempotency, cancellation, failure and interrupted restart states; configured roots/opaque IDs |
| React scientific workbench | Implemented and browser-tested | Real shadcn/Tailwind, synchronized spec, real results/history/export; no training on open |
| 1D curves, 2D heatmaps, orthogonal 3D previews | Implemented | Dimension-specific controls; bounded slice payload; mean/sample/empirical spread |
| Training-budget runner and local pilot | Executed 0/100/500 | Fixed two validation/test cases; validation-only tuning; negative/nonmonotonic results retained |
| Darcy extension | Executable static joint a/u specialist | docs/darcy.md; actual CPU/MPS evidence in verification.md; restricted inverse baseline implemented |

| Symbolic/data continuation | Status | Evidence / limits |
|---|---|---|
| MathLive visual mathfield and nine editable system presets | Implemented, real Chromium keyboard-tested | Local npm/fonts; fractions, Greek toolbar, multiple rows, synchronized LaTeX; no unrestricted LaTeX semantics |
| Complete versioned 𝒞 state and request import/export | Implemented, unit/API/browser tested | Explicit roles, units, boundary/initial/measurement/operator state; static drafts have no initial time |
| Explicit PDE parser with scalar/vector ranks and rate definitions | Implemented for documented subset | tests/unit/test_symbolic.py; bounded depth/size, aliases, declarations and canonical-source roundtrips |
| Symbolic source to executable checkpoint matching | Heat, Darcy and chemistry, automatic routing | Current full request/operator trees, compatible checkpoint IDs/exclusions, explicit override validation; other templates remain exportable |
| Larger named heat data profiles | Configured; small-instance generation tested | Full 2400/12000/1200 instance sets not generated in this task; no new-family data |
| Restartable shards/lazy batches/train-only streaming normalization | Implemented and integration-tested | Seeded independent instances, immutable held-out manifest, checksums, interruption, exact CPU resume |
| Intended-layout MPS preflight | Actually executed for 32² and true 16³ | Two updates each, float32, configured model/batch, validation/resume/guidance and CPU reload; no quality claim |
| CUDA / actual Colab for new profiles | Untested | Package/notebook remains compatible by shared CLI; requires real hardware/runtime execution |

## Multi-specialist status distinctions

| Level | Heat | Darcy |
|---|---|---|
| Implemented specialist | Periodic transient scalar, 1D–3D | Periodic static positive scalar permeability + pressure, 2D/3D |
| Available checkpoints | Preserved CPU smoke and heat pilot outputs | outputs/multispecialist-20261021/cpu-{2d,3d}/training/inference.pt |
| Smoke-trained | Existing 1D/2D/3D and MPS runs | Actual 20/12 CPU updates; 32² and true16³ MPS two-update preflights |
| Scientifically evaluated model | Small negative 0/100/500 pilot only; robust accuracy/calibration unestablished | No; conventional restricted inverse benchmark implemented |
| Full larger training | Not run for larger profiles | Configured only, not run |
| CUDA / actual Colab | Untested | Untested |

Partial observations of either/both Darcy physical fields, static CSV/JSON,
separate field previews, exact cell coordinates and gauge diagnostics are
implemented. Neither arbitrary equation understanding nor resolution/time
generalization follows. See multispecialist.md and the dated verification
record for commands and actual browser evidence.

## Online / inverse / chemistry continuation

| Capability | Implemented / actual evidence | Limitation |
|---|---|---|
| Offline/online/hybrid reference providers | Shared trainer; ID regeneration, checksummed fixed evaluation, bounded replay, frozen calibration | Single producer/rank; workers=0, no asynchronous prefetch |
| Deterministic resume and interruption | Bitwise CPU model/ID sequence tests for fresh/replay; SIGINT safe-boundary checkpoint | No cross-device bitwise guarantee; source/config must match |
| Mode cost comparison | Matched priors/model/budgets; actual preparation/solve/load/train times and reuse counts | Tiny CPU single trials; not a general speed claim |
| Darcy conventional inverse | Positive bounded Fourier prior, converged forward/adjoint CG, verified gradient, BFGS | Single-forcing identifiability; six/eight study test optimizations capped |
| Darcy online learning study | One0/20/80 path; validation-only tuning, fixed held-outs and all methods | Inverse better; generative field errors non-monotonic; two independent test problems |
| Chemistry reference/representation | Positive periodic Strang, exact reaction, square-root channels, physical residual | No no-flux walls, advection, unknown kinetics or invariant projection |
| Chemistry training/reconstruction | Actual2D/true3D CPU and MPS resume/guidance/export/CPU reload | Integration-only checkpoints, not calibrated ensembles |
| All-three-family routing and workbench | Current parsed system/roles/layout; species and baseline error tables; ten real browser workflows | Fixed names/operator patterns and exact grids |
| Online Colab notebook | Local real CLI launch/resume test without saved training directory; portable export | Actual Colab/CUDA untested |

Earlier rows are historical evidence; this continuation supersedes any prior
planned/unfinished chemistry or Darcy inverse labels. Current details and
commands are in online-training.md, chemistry.md, darcy-study.md and the latest
verification section. Full larger pilot training was not launched.

## Process diagram continuation — 25 October 2026

| Capability | Current actual implementation | Limit |
|---|---|---|
| Component physical/observation reports | Named units/reductions/counts/thresholds, finite flags, raw/masked/noise residuals, estimated/final labels | Exact nodal/saved-time observations; no correlated noise/interpolation |
| Diagnostic next-action controller | Continue/configured guidance, patience, final verification, bounded fresh retry, stop/failure/cancel, durable trace | Explicit component bounds required for criteria met; low residual is not accuracy |
| Global physical refinement | Actual8²→16² heat fresh compatible checkpoint; y/coordinates preserved; CLI/API/browser | One spatial level, same times; no Darcy/chemistry/temporal/local AMR |
| FK resampling | Later FK continuation implements actual scheduled systematic resampling | Fixed supported SDE; FK plus refinement/retries rejected |

Current process status supersedes earlier blanket refinement/controller planned
labels. See process-audit.md and process-verification.md. The whole diagram is
not claimed universally complete; FK now executes, while coupled multilevel FK,
adaptive SDE, rejuvenation and inference resume remain absent.

## FK continuation status

| Level | Evidence | Limit |
|---|---|---|
| Analytic target correctness | Gaussian/mixture four-seed P512/4096 × N16/128; drift/potential/resampler tests | Finite-step and finite-particle error |
| Learned-prior approximation | Explicit Q_h terminal tilt and current CFM decoder | Learned score/ODE density consistency not guaranteed |
| PDE integration | Heat1D–3D browser/CPU, heat2D/3D + Darcy2D + chemistry3D MPS probes | Existing smoke checkpoints, not new scientifically evaluated models |
| Scientific reconstruction | One fixed heat test / three sampling seeds, validation-selected scales | FK unobserved error worse than unguided; no superiority or calibration |
| Backends | CPU and actual MPSfloat32; explicit CPUfloat64 weights/RNG | CUDA/actual Colab untested |
| Refinement | Legacy MPSheat8²→16² executes separately | FK plus refinement is rejected |

See fk-verification.md for current tests/actual browser evidence and exact commands.
Earlier planned FK entries are superseded only for the narrow method above.

## Everyday usability/orchestration

| Operation | Implemented behavior | Evidence / limit |
|---|---|---|
| Setup/doctor | Safe local directories, npm ci, matching Node engines, headless independence, JSON diagnostics | Actual setup and doctor; no global software install |
| Named train | Automatic held-outs/calibration/online generation, progress/manifests/atomic exports | Real CPU/MPS updates; full pilot budgets unexecuted |
| Resume/extend | Safe-boundary SIGINT, exact logical IDs/optimizer/RNG/transform restore, explicit total extension/device mapping | CPU deterministic and actual signal/device checks; source identity strict |
| Discovery | Managed manifests, latest/best roles, external links, safe incomplete-export rejection | API tests and real locally trained workbench reconstruction |
| Evaluation | One validation case by default; recorded explicit test/full; conventional baseline | Actual CPU evaluation; no scientific improvement claim |
| Serve/frontend | Independent startup, useful disconnected state, port conflicts, reload queue lock/cleanup | Real Chromium/service probes; localhost only |

Current commands/evidence are everyday-usage.md and everyday-verification.md.
The mathematical capabilities above are preserved; ordinary training completion
does not promote a pilot to a scientifically validated model.

## Scientific workbench redesign

| Capability | Implemented view | Evidence / restriction |
|---|---|---|
| Problem definition | MathLive, specialist interpretation, declaration glossary, C map, explicit default review, persisted invalid drafts | Current equation drives API routing; unsupported boundaries remain draft-only |
| Input views | Real 1D/2D/3D coordinates, known fields, hatched unknowns, selected observation rows/markers | Exact saved nodes/planes; no full-volume interpolation |
| Reconstruction inspection | Coordinated weighted field/spread/input/reference/error slices, observation prediction/residual graphs | Truth is optional and evaluation-only; spread uncalibrated |
| Live/process graphs | Received endpoint/PDE diagnostics and FK ESS/events, separate final verification | No invented future points; polling may miss rapid intermediate updates |
| Models/training | Managed discovery, actual CFM/validation history, checkpoint positions, actual preset metadata | External exports without histories have no fabricated curve; pilot quality unknown |
| Run comparison | Same-problem quality versus measured cost, common-truth accuracy only | 2–4 runs; exact request identity required, no automatic ranking |

See workbench-verification.md and workbench-api.md. This changes interaction and
visibility, not the mathematical capability or scientific validation level.

## Reference/suite continuation — 26 October 2026

| Capability | Implementation / evidence level |
|---|---|
| Reference solving without model weights | Current-request capability selection; heat finite Fourier analytic, heat grid interpolant numerical, harmonic-FV Darcy and positive chemistry numerical; independent 1D–3D tests |
| Reference-only input editor and previews | Editable explicit JSON scalar/array/safe expression/Fourier/file, solver/accuracy, immutable artifacts and bounded slice endpoints; actual browser journeys |
| Explicit inverse disclosure / sampling | Fixed/random/regular/space-time/exact positions/reuse; field/time/noise/seed, shared exact on-grid operator, hidden full fields excluded |
| Post-run coordinate-aware evaluation | Per-field quadrature MSE/RMSE/MAE/relL2, observed/unobserved/per-time/per-sample, weighted summaries, null near-zero norms, Darcy gauge/log-a; native or explicit nested restriction/shared times |
| Complete experiment export | Existing numerical result/diagnostics plus reference/sampling provenance and manifest ZIP; numerical archive integrity tests |
| Managed six-layout training suite | Existing sequential trainer, plan/filter/budget/source/config/reuse/lock/failure/SIGINT/resume; real CPU six-child and bounded MPS two-child execution |
| Suite model/history discovery | Real managed exports; browser suite summary and child history; no training on opening UI |
| Scientific quality | Available exports and smoke-trained checkpoints are distinct from scientifically evaluated models. Two-update suite completion is execution evidence only. Existing negative heat/Fourier and Darcy/FK studies remain authoritative. |
| Unsupported | Arbitrary symbolic PDE solver, arbitrary boundary/mesh/interpolation, fully known target conditioning and unknown parameter/source specialists; generic temporal/local refinement. CUDA/actual Colab suite execution untested. |

## Flow workspace/capacity continuation

| Capability | Actual implementation/evidence |
|---|---|
| Automatic/explicit model contract | Captured the original empty-ID UI request; new explicit selector resolves compatible registered content server-side. Legacy valid IDs remain strict; stale/wrong-layout/unavailable IDs reject clearly. |
| Flow matching | Production Gaussian-source linear CFM objective, fixed transforms/conditions, real changed weights, analytic ODE order/endpoints and learned Gaussian moments; flow-matching-audit.md. |
| Backbones | Original residual CNN retained; one new periodic U-Net, explicit width/blocks/levels/condition/coordinate/fixed-time metadata, strict load. Actual parameter counts in capacity-resolution.md. |
| Native1024² heat | Actual MPS reference, forward/backward/update, physical guidance and2-step production Heun with shape[1,1,1,3,1024,1024]. One-update operation probe only; no scientific high-resolution reconstruction claim. |
| Genuine generation visualization | Bounded actual model/current/endpoint snapshots, separate s/τ, FK parent composition/gaps; passive observability regressions. No fabricated interpolation. |
| Linked spatial explanation | Actual weighted mean/individual field or empirical spread, heatmaps/lightweight surface/shared point, native local heat/Darcy/chemistry terms, true3D orthogonal slices and bounded displays. Spread has no invented PDE residual. |
| Reference association / clear | Automatically retained evaluation-only context outside worker; reopen preserves comparison. Confirmed local clear handles keep/cancel/return and late responses, without deleting saved artifacts. |
| Capacity pilots | New medium64²/large128² and explicit capacity-local suite. Bounded matched20-update comparison and MPS probes/resume executed; full3000/10000 budgets not launched. |
| Limits | Fixed-grid specialists, no validated cross-resolution or general symbolic solver; high-resolution FK and long-training quality unverified; CUDA/actual Colab untested. Larger did not win the bounded comparison; Fourier inversion remains better. |

## Training catalogue / native 512 continuation

| Capability | Implemented and evidence level | Remaining limit |
|---|---|---|
| Full finite catalogue | 11 inspectable targets, full/all default, named quickstart/local-pilot/capacity-local; frozen manifests and verified budget-aware missing/reuse | No full expanded suite launched by this task |
| User's original suite | Six original 3000/1500-update targets completed; current coverage distinguishes later missing targets | Completion does not establish scientific accuracy |
| Chemistry equivalents | Bounded polynomial rearrangement/commutation/explicit mass-action substitution; exact law/layout reports | Fixed species/parameter names; other kinetics reject |
| Native 512² heat | Real MPS forward/backward/update, gradient and Heun probe; bounded managed training/resume and real reference/measurement/inverse UI slice | No long-training quality validation; no high-resolution FK claim |
| Training overview | CLI manifests, kernel ownership/heartbeat/live progress, actual losses/validation positions/counters/config/logs, full plan and coverage | Legacy missing telemetry remains unavailable; no speculative ETA |
| Storage | Rebuildable cached stat inventory, pinning, preview-first exact digest plans; fixture-tested recoverable quarantine | Historical/scientific/QA/unknown files protected; no scientific output cleanup performed |
| Analytical examples | Editable product/single-axis/multi-mode/challenge, non-unit domain scaling, bounded candidate PDE/initial/periodic checks | Numerical consistency is not a general symbolic proof |
| Learning diagnostics | Actual fixed-pair overfit, tractable CFM toy, prior spectra/decay/statistics, fixed validation seeds/steps and conventional baseline | Small diagnostics do not establish a unique explanation or generalization |
| Extra model families | No new diffusion, VAE, autoregressive or likelihood-trained flow | Seven lecture PDFs mapped as background/optional/implemented, with page citations |

Current verification and commands: catalogue-verification.md and training-catalogue.md.


## Optional coordinate heat/source continuation — 27 October2026

| Capability | Implementation and actual evidence | Limit |
|---|---|---|
| Reference source modes | Analytical/manufactured, numerical and measurements-only; actual browser generation/import/invalid-input journeys | Manufactured functions are labelled unverified; forced heat is not fabricated |
| Observations | Explicit scalar totals, fields/times/seed, fixed/random/regular/space-time/positions, independent absolute or relative-RMS Gaussian noise, actual per-row sigma, clean/noisy separation, row export/import | Existing fixed model uses exact nodes/times; no correlated noise |
| Latent coordinate heat | Trainable set AE, frozen latent affine transform and observation-conditioned transformer CFM; real two-stage save/resume in CPU1D/2D/3D tests and MPS2D probe | Only scalar unforced periodic heat; exact dimension/domain/schema/D/time-range contract |
| Set/query behavior | Padding/permutation invariance, order/chunk consistency, zero-observation training path, continuous/scattered/regular queries and physical derivatives tested | One reference discretization per checkpoint; no arbitrary geometry/coefficient fields |
| Checkpoints/status | Explicit AE/flow identity, embedded AE portable CPU inference without original stage/data, fingerprinted registration, current-source API checks, actual locks/manifests/history and Runs reopening | Full-budget completion is not scientific accuracy; strict optimizer resume source/config/data |
| Comparison | Same held-out problems/noisy observations; actual8²/12²/32² queries, counts2/5, sigma0/.03, seeds91/92, ensembles2, per-sample/mean/physics/cost reports | Four-update latent smoke versus3000-update fixed baseline is not matched scientific training; interpolation labelled |
| Scientific status | Existing user fixed-grid checkpoints preserved; CPU AE4/flow4 and MPS AE2/flow2 exports available | No scientifically validated coordinate model; AE held-out distortion is large. No long jobs, new family, CUDA or actual Colab |

Commands: [coordinate-specialist.md](coordinate-specialist.md). Checks, failures and
source stages: [coordinate-verification.md](coordinate-verification.md).

## Integrated training continuation

| Capability | Evidence | Limit |
|---|---|---|
| One-action UI/CLI training | Real CPU pipeline checks for both model types; both coordinate stages, fixed validation, registration and explicit selection | Development/extended budgets not executed by this task |
| Persistent control | Real detached worker survives API restart; both trainers stop/resume; stale-owner, duplicate-token and failed-stage regressions; actual rendered stop/reload/resume | Strict source/config/dependency resume; no latent budget extension |
| Representation sanity gate | Actual validation errors and training-mean constant comparator; paused development experiment and explicit override regression | Minimal sanity check, not scientific adequacy; pipeline exemption labelled |
| Unified discovery | Existing fixed/coordinate exports and historical project workspaces; compatibility enforced independently by API | External workspaces explicit; absent historical validation stays unknown |
| Train further | Real bounded fixed-grid continuation preserves previous checkpoint/export/history/validation snapshot and model selection | User-initiated only; same distribution; no accuracy guarantee |
| Scientific status | New runs clearly marked pipeline checks; fixed validation case/error previews and metrics | No new accuracy improvement, CUDA or actual Colab verification |

Commands and evidence: [training-workflow.md](training-workflow.md),
[training-workflow-verification.md](training-workflow-verification.md).

## Focused heat playground — 27 October 2026

| Capability | Implemented/evidence | Scientific limit |
|---|---|---|
| Default heat2D preset/draw/formula, cached FFT playback, exact cosine comparison | Focused numerical tests and actual rendered browser journeys | Finite Fourier grid for general initial data; periodic constant D only |
| Exactly N scattered measurements, seeded absolute noise, time retention, points-only import | Fourier/cosine point agreement, API and browser checks | Sparse measurements do not uniquely determine an initial field |
| Coordinate inverse and aligned comparison | Actual preserved AE4/flow4 inference, no truth in worker request; points-only omits reference metrics | Poor pipeline-trained model; flexible queries are not accuracy |
| Separate heat2D coordinate training view | Reuses verified orchestration, focused preset/device checks; no new training launched | Development budget not demonstrated adequate |
| Reviewed model/resume/stage deletion | Disposable fixture + real browser preview/cancel/apply, shared dependencies, active locks, traversal and protected-file checks | Unknown/external files protected; filesystem free-space effects qualified |

Exact evidence, failures and backend status: heat-playground-verification.md.

## Substantial Colab coordinate heat training — 27 October 2026

| Capability | Implemented / evidence | Limit |
|---|---|---|
| Six-cell notebook / source ZIP/private checkout | Real ordered CPU archive/setup/profile/two-stage/export execution; source/dependency checks | Colab UI, credentialed checkout and mounted Drive untested |
| Large / Extra large presets | Exact requested Large; meta parameter counts and configuration validation; configurable XL | No substantial job, GPU fit, duration or accuracy claim |
| Diverse procedural periodic data | Six families, split identity/train-only calibration; 64² vs128² reference gallery | Single-grid training; no arbitrary-drawing coverage or learned resolution validation |
| Shared local/CLI/Colab training | One stage driver and optimizer loop; detached local worker regression passes | Legacy optimizer resumes need original source |
| Resume / extend / retention | Exact CPU resume, weighted accumulation, immutable AE, partial export, content-verified recovery and retention tests | CUDA RNG/AMP and real Drive disconnect recovery unverified |
| GPU optimizations | Precision policy, scaler, SDPA request, activation checkpointing, microbatching implemented; CPU gradient/accumulation checks | Actual CUDA test skipped; no speed or memory improvement claim |
| Portable import / inference | Isolated CPU reload without data, real API and rendered playground numerical reconstruction | Pipeline checkpoints have poor quality, not scientifically evaluated models |

See docs/colab-training.md and docs/colab-verification.md. No new PDE family,
scientific accuracy improvement or validated cross-resolution learning is claimed.

## Continuous heat2D and measurement placement — 28 October2026

| Capability | Implemented / actual evidence | Scientific boundary |
|---|---|---|
| Continuous physical time | New main presets [0,1], fixed reference contexts + fresh AE/sensor times; analytic constants/modes/endpoints/bounds tests | Default D unchanged; not four-frame interpolation |
| Temporal checkpoint metadata | Mode/interval/distributions, raw legacy capability, independent continuous-request rejection | Old snapshots are not continuously trained; temporal accuracy flag remains false |
| Arbitrary-time inference/display | Ordered chunk checks, actual0.237 sensors, saved latent decoded at repeated times; real browser | Syntactic/execution flexibility, not demonstrated learned accuracy |
| Random/even/heat/manual placement | Exact-count/rectangular/prime/edge/signed/constant/coverage/relaxation tests; real overlays | Heat focus is reference-assisted and limited to retained regions |
| Multiple times / import/export | Fixed/adaptive semantics, actual times/noise/policy provenance; no reference payload to model | N is total scalars; no optimal-design/calibrated-posterior claim |
| CPU | Focused and affected training/validation/resume/API tests; actual notebook cells | Only tiny optimization budgets |
| MPS | Actual AE2/flow2 + physical validation/export, CPU reload after explicit host-FP64 statistics correction | Execution only; no long training or cross-device bitwise claim |
| CUDA / actual Colab | Not executed; existing AMP/scaler/retention/source workflow retained | GPU precision/performance/real Drive behavior unverified |

No new model is scientifically evaluated. Uniform/initial/early/late metrics and
a bounded equal-budget policy comparison are saved; no test-based winner or
accuracy improvement is selected. See continuous-heat-verification.md.

### Spatial heat profiles —28 October 2026

Implemented and numerically/browser verified: periodic physical integrals beside
current heat2D fields, signed/magnitude display, reference/model common scales,
coordinate hover, live edit/time updates and responsive aspect-preserving layout.
13 focused tests, real CPU reconstruction and real field browser journeys; no new
training or scientific accuracy claim. Evidence/limits: spatial-profiles.md.

### Configurable heat representation losses —29 October2026

Implemented with actual encoder/decoder optimizer updates: scalar periodic heat2D
MSE, MAE, contrast-weighted MSE and experimental blended equal-region MSE. New UI/
updated notebook default α.5; old runs/configs retain recorded MSE. Continuous time
groups, physical-area periodic segmentation, explicit supervision budgets, CPU/MPS
smokes, exact CPU resume, shared notebook execution and rendered metric/overlay
comparison are verified. CUDA AMP/actual Colab and scientific accuracy gains are
unverified. Other PDEs/3D region losses and decoded auxiliary flow losses are absent.
Details/evidence: field-reconstruction-losses.md, field-loss-verification.md.

### Training performance recovery — 31 October 2026

| Capability | Status | Implementation | Evidence | Limits |
|---|---|---|---|---|
| Performance recovery | Implemented and bounded CPU/MPS execution verified | Native precision selection policy, bounded batched spectral references, exact-key frozen caches, actual default-B microbatches, reconciled step/nested timing | docs/training-performance.md; outputs/performance-20261031/ | Actual T4/CUDA/Colab speed and FP16 execution still require the supplied same-config benchmark; no accuracy claim |
