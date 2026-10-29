# Instructions for Codex working on PDEFlow

## Read and preserve the goal

Read docs/project.md, docs/architecture.md and docs/experiments.md before implementation. If these files disagree, identify and resolve the specific conflict in docs/decisions.md. Preserve the modular 1D–3D architecture and the chemistry/steering research purpose. A small verification problem is a first implementation slice, not a scope reduction.

The files in references/ are scientific sources. Preserve attribution and licenses when reusing code. Do not claim a paper proves our proposed adaptation. If a reference is absent or inaccessible, record that limitation and proceed on components that do not depend on it. Derive and cite delicate sampling conversions before implementation.

## Deliver complete working slices

- Implement real numerical behaviour; no success-returning placeholders, random arrays presented as solved data, fake training, hard-coded reference predictions, or empty notebook demonstrations.
- Use small concrete modules and typed dataclasses/protocols. Avoid speculative plugin infrastructure and deep inheritance trees.
- Keep training, numerical reference solving, differentiable residuals, generation, steering and evaluation separate.
- Every claimed capability must be implemented and tested at the claimed level. Mark planned features as planned.
- Advance autonomously through the requested scope. Make routine reversible choices and record them; ask only when a missing decision blocks meaningful work.
- Do not launch long/expensive training or external services merely to make a smoke test look impressive. Run bounded real training in the acceptance budget and provide larger configurations.

## Mathematical correctness

- Separate spatial dimension, physical time, generation time/noise level, training iteration, problem batch and particle population.
- Record field/channel meanings, coordinates, boundary treatment, units/scaling and known/unknown roles.
- No target leakage through masks, conditioning, normalization, crops or train/test splitting.
- Evaluate physical residuals on correctly decoded physical fields. No detach/NumPy conversion in required guidance gradients.
- A low residual, attractive image or successful smoke run is not scientific validation.
- Preserve relevant reaction stoichiometry, fluxes, gauges and compatibility conditions. Do not call u=0 a Neumann condition or assume all chemical concentration sums are conserved.
- Never invent a general score/velocity conversion, exact posterior guarantee, adaptive-SDE rule or FK weighting procedure. State assumptions and endpoint handling.
- Do not add uncorrected noise to an ODE and claim preserved marginals. FK particle weighting/resampling is distinct from classifier-free or gradient guidance.
- Projections/corrections can change sample distributions; document them and attribute their cost and benefit.

## Modularity and compatibility

- Exchange backbones, formulations, paths/schedules, compatible steppers, steering strategies and PDE modules through explicit interfaces/configuration.
- Prefer factories and capability declarations over PDE-name branching inside the trainer/sampler.
- Validate checkpoint/layout/conditioning/grid/boundary/device/formulation compatibility before expensive execution.
- Unsupported combinations must raise descriptive errors rather than silently change physics or algorithms.
- Generic equation evaluation does not establish generic neural equation understanding.
- Physical refinement and generation step-size adaptation are independent. New grid execution is not automatically resolution generalization.

## Colab and MacBook execution

- One installable package; notebooks only install, configure, launch, resume, inspect and export.
- Support CPU smoke execution. CUDA and MPS are optional backends selected centrally; no unconditional .cuda(), CUDA-only mandatory core dependencies, or concealed CPU fallback.
- Precision and device choices must be validated. Never claim CUDA or MPS tested without running on it.
- All paths are configurable. No hard-coded user home, repository URL, credentials, Drive account, mount requirement or private local path.
- Colab cache/runtime is disposable. Persist checkpoints/results to a configured location. Handle interrupted runs and atomic writes.
- Checkpoints retain transforms, model/formulation/condition metadata and capability declarations. Resume includes optimizer, scheduler, scaler if used, progress and random state.
- Inference must load an exported checkpoint without training data and support CPU device remapping. Cross-device bitwise equality is not promised.
- Pin a tested dependency configuration without assuming Colab and the developer's Mac have identical preinstalled packages.

## Tests and verification

Use analytic/manufactured solutions, convergence, conservation, observation/gradient checks, sampler-order checks, and bounded end-to-end training/save/reload/reconstruction runs. Include true 3D coverage. Avoid tests that simply repeat an implementation or only instantiate classes.

Separate numeric references and residual checks enough to expose shared mistakes. Record actual test commands and failures. Fix relevant failures, then stop repeating successful checks without cause. Do not mask failures by loosening tolerances without a numerical reason.

Keep the repository clean: no committed caches, generated datasets, model weights or large output trees by default. User-provided references are not to be deleted or modified. Do not initialize remote services, publish, push or overwrite user work without task authorization.

## Documentation and final handoff

Update README.md with exact local and Colab commands; docs/mathematics.md with implemented formulas/conventions; docs/decisions.md with meaningful design choices; and the capability matrix with evidence. Keep prose plain and distinguish tested behaviour from intended features.

At the end report what works, how it was verified, runnable commands, backend test status, output locations, known limitations and the next complete implementation slice. Do not end with merely a proposed plan when implementation was requested.
