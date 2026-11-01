# Coordinate specialist: inspected starting point

27 October2026. The previous catalogue work is retained, not repeated.

- Existing installed package: PyTorch direct-field linear CFM, spatial CNN/U-Net,
  fixed physical-time channels, heat1D–3D and Darcy/chemistry2D–3D. No trained
  latent encoder/decoder, observation-conditioned set backbone or diffusion model.
- Eleven fixed-grid catalogue entries, explicit large/high-resolution profiles;
  old six-child3000/1500-update pilot completed its original budgets. Expanding a
  model's output tensor did not establish learned resolution flexibility.
- Existing immutable reference solvers, safe scalar expression parser, editable
  heat templates, independent formula diagnostics, strict on-grid row import,
  reference/measurement/inverse/comparison chain and per-time/slice plotting were
  reusable. Complete reference-field import and explicit source modes were missing.
- Sampling already supported fixed sensors, uniform discrete random nodes, regular
  subsets, scattered saved space-time nodes and explicit positions; noise was one
  absolute scalar. Count semantics existed in prose but needed a visible total.
- Existing model discovery/checkpoint contracts correctly enforce fixed grids.
  They must remain strict; portable latent-stage checkpoints need their own
  explicit capability identity instead of masquerading as old CNN exports.
- Existing single workbench queue, runtime selection, physical sampling/reference
  providers, atomic writes, run locks, storage protection and Models & Training
  views are retained. New stage training reuses these primitives; no second queue
  or browser-owned trainer is added.

Implemented extension and its limits: coordinate-specialist.md. Actual execution:
coordinate-verification.md. Scientific accepted outputs and the Git index are
preserved. No long training/full suite or automatic scientific-output cleanup is
part of this task.
