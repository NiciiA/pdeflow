# Integrated training verification — 27 October 2026

Evidence root: outputs/integrated-training-20261027/. These are actual prior
implementation runs. The final handoff inspected saved evidence, completed docs,
reran lint/whitespace and checked preservation; no training or completed suites
were repeated after the handoff-only instruction.

| Check | Actual saved result |
|---|---|
| Affected Python regressions | **57 passed**, 48.54s, python-final.log: real tiny coordinate/fixed training, validation/previews, stop/resume, stage/preparation failures, gate/override, detached worker/API restart, duplicates, discovery, continuation snapshots and retained local/stage/selection/storage behaviour. |
| Frontend | **46 passed**, frontend-verified.log: passive opening, settings, duplicate clicks, explicit use, stop and quality override. These UI unit tests are distinct from numerical browser integration. |
| TypeScript / production build | Passed, build-verified.log. Existing large MathLive-containing bundle warning remains. |
| Lint | Zero errors, three existing component-export warnings, lint-handoff.log. |
| Actual browser | **3 passed**, 1.8m, browser-readiness.log: two-stage training→validation/error preview→reload→use→numerical coordinate inference; fixed-grid training→validation→use→reference/observations→numerical inverse; safe stop→reload→resume. Numerical services were not mocked. |
| High-level CLI | cli-coordinate.log: AE4/flow4; cli-fixed.log: fixed2, CPU, automatic validation/registration. cli-resume.log restores the real stopped80-update representation probe without checkpoint paths or a special workspace. |
| Devices | CPU high-level execution, actual availability/Auto resolution and unavailable-device rejection. Prior coordinate MPS evidence remains in coordinate-verification.md. No new high-level MPS training, CUDA or actual Colab execution is claimed. |
| Preservation | preservation-before.json and completion.json retain original checkpoint/index fingerprints. Discovery never moves weights. |

The final browser run preceded metadata-only refinements to canonical UI/CLI
request defaults, offline/hybrid count descriptions, training-example counters
and exact Darcy field naming. Final Python/frontend/build checks cover those
refinements. Repeated successful checks are not added into larger test counts.

## Actual training and scientific limits

Only execution budgets were launched: fixed-grid2, coordinate AE4/flow4, and two
AE80/flow4 cancellation/resume probes. All task-owned workflows are completed at
handoff. The first80-update probe resumed through CLI, the final one through UI.
Neither is a useful trained flow model merely because its AE received80 updates.
Development500 and extended budgets were not executed.

A recorded four-update AE has validation MSE approximately0.7617 versus constant
baseline0.7524 (ratio1.012). Pipeline exemption continues execution while displaying
this poor result. A regression uses real tiny development stages with a deliberately
strict configurable gate, confirms no flow is created, then verifies explicit
override. This is a control test, not a scientific training-budget study.

Test arrays can be prepared in their separate immutable split; test reconstruction
metrics were not used to choose settings or promote models. Prior Fourier-inverse
superiority and non-monotonic heat learning findings remain unchanged. No new
accuracy improvement, fine-scale generalization or calibrated uncertainty is claimed.

## Corrected failures and remaining limits

- The initial adapter called a nonexistent specialist method; it now uses
  default_problem. A 1D sensor budget was bounded to available nodes. Initial
  failures remain in python-workflow.log.
- Hidden advanced decimal HTML constraints initially blocked submission. The
  resolved server validation is authoritative; normal browser Train clicks work.
- Stale polls could replace a newly accepted run. Poll epochs now guard mutations;
  polling never overlaps and loading differs from an empty registry.
- Initial browser selectors assumed exact text without appended storage details.
  Cold discovery of85 exports can exceed5 seconds. Final tests wait for readiness
  up to20 seconds, without weakening numerical assertions or using fixed sleeps.
- Preview assertions await asynchronous results; persisted coordinate selection has
  an explicit string type. Final frontend/TypeScript/build pass; initial logs remain.
- Resume needs recorded source/configuration/dependencies. Latent budget extension
  is unsupported. Fixed-grid continuation preserves the previous budget snapshot.
- Full development/extended training quality and larger-profile resource fit remain
  unverified. No scientific advantage follows from a completed pipeline check.

## Preservation and shutdown

The original90-entry Git index is unchanged; no staging/reset/commit was performed.
The repository still has no HEAD commit. Existing working changes remain available.
No source archive regeneration or scientific cleanup was performed.

Only task-owned API/Vite verification services and the temporary browser tab were
closed. Ports8000/5173 were checked free. completion.json records final source,
checkpoint preservation and process/port checks. See [training-workflow.md](training-workflow.md)
for normal operation and exact simple commands.
