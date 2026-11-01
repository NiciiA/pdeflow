# Heat/wave training-equation wiring audit

The defect was both routing and presentation. In the focused playground training
page, the selector was hidden and its initial request inherited the displayed PDE.
Opening from Heat therefore selected real heat-only training, while the generic
parameter-range listing showed D and c. Explicit joint requests already reached the
shared trainer. In addition, advanced family_weights could disagree with the chosen
problem, and the old summary was based on the problem label rather than the final
configuration.

The visible **Training equations** selector now defaults to **Heat + Wave** for new
runs and remains independent of the playground. Choices route `problem=heat-2d`,
`wave-2d`, or `heat-wave-2d`. API/CLI defaults match. New request payloads explicitly
include the selected reconstruction loss and alpha. Backend validation resolves
family weights and rejects a conflicting advanced family set. Proportions within
the selected set may vary; the summary shows normalized actual probabilities.
Only relevant physical parameter ranges appear for single-family training. Heat
state is u; wave state is displacement u and velocity v. Initial-field shape mixtures
are explicitly identified as shapes within PDE families.

`coordinate.physics.training_spec` supplies the summary and persisted stage metadata.
The trace is UI selector → `training/plan` or `training/start` → validated config →
`Dataset` realization family → common A1/A2/B family sampler → stage loss → config,
workflow/stage manifests and checkpoint. Enabled state schemas and normalized family
weights persist alongside raw config. Per-update family example counts and per-family
losses are recorded. Changed mixture or field-loss settings reject strict resume.
Existing checkpoints/data/results are preserved, and existing recorded loss choices
are not rewritten. New metadata changes the source fingerprint; prior source ZIPs
remain the strict-resume source for earlier runs. Inference architecture is unchanged.

## Actual loss wiring

- A1: the selected differentiable decoded field loss updates the state encoder and
  spatial decoder. Heat u and wave u/v use separate detached signed reference regions
  and quantity-specific training normalization, then average active quantities.
- A2: the same selected field objective acts through learned evolution, with A1
  representation frozen. Existing latent alignment and latent/decoded composition
  terms retain their defaults 0.1 and 0.01.
- B: masked latent flow-matching velocity loss remains unchanged. No spatial region
  weights are assigned to latent tokens. Default optional decoded auxiliaries stay off.

Region-balanced MSE with alpha=.5 remains experimental and selected by default.
The audit changes no scientific loss formula and claims no accuracy improvement.

## Evidence

All evidence is under `outputs/training-wiring-20261030/`.

| Check | Evidence |
|---|---|
| Three real API resolutions / datasets / saved configurations | `python-focused.txt`:5 passes; `heat-2d-request-resolved.json`, `wave-2d-request-resolved.json`, `heat-wave-2d-request-resolved.json` include submitted/resolved/persisted plans and actual generated family sets |
| Actual three-stage joint execution | `joint/`:2 fresh optimizer updates each for A1/A2/B on CPU. Every update contains1 heat +1 wave realization. No existing model was retrained |
| Actual decoded field gradients | `joint-stage-loss-evidence.json`: heat u and wave u/v have nonzero prediction gradients in A1 AND A2; B makes no spatial field-loss calls with its default auxiliary policy |
| Alpha control | Same architecture, seed, data and query budgets; alpha0 A1 first loss0.33576923 vs alpha.5 loss0.33859456. Encoder u/v and decoder u/v optimizer updates differ. `joint/alpha0/` is a separate2-update control |
| Resume | Changed family set and alpha both reject before checkpoint alteration, verified by hash |
| Existing training controls/API | `backend-regression.txt`:13 passes,1 intentionally deselected joint execution already covered above; includes managed stop/resume and quality-gate behavior |
| Frontend | `frontend-final.txt`:63 passes across13 files, including the invalid-summary guard. `build-final-current.txt`:TypeScript/production build passes. `lint-final.txt`:zero errors, three existing Fast Refresh warnings |
| Actual rendered page | `browser-verified.txt`:real Chromium pass against isolated API8001/frontend5174. `browser-training-routing.json` contains real requests/responses for all three choices, alpha preservation and zero training submissions |
| Screenshot | `joint-training-summary-detail.png` and full-page `joint-training-summary.png`; actual rendered UI, visually inspected |

Control-plane persistence probes save configurations without spawning optimizers;
their process launch is mocked explicitly and is not numerical execution evidence.
Their test manifests are retained separately from discoverable user workflows.
The actual A1/A2/B and alpha-control runs above use real optimizers and numerical
references. Browser verification resolves real API plans and does not launch training.
No substantial training, accuracy evaluation or CUDA/MPS training was run for this audit.
The existing poor smoke models and failed representation quality result are unchanged.

Only this audit's services on 8001/5174 received stop signals. No stop signal was
sent to the pre-existing services on 8000/5173; those processes were also absent at
the final read-only port/process check. Start or restart the backend to load changed
backend validation/summary code. All 3348 pre-existing checkpoint/array artifacts
in the preservation baseline remain byte-identical. Hashes and shutdown verification
are recorded in `preservation-after.json` and `completion.json`.
