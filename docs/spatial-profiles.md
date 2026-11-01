# Spatial heat profiles

The heat playground shows **Heat along y** on the left and **Heat along x**
below the current field. **Show profiles** defaults on; **Signed field** is the
default. **Magnitude** integrates the absolute field without changing the heatmap,
measurements, reference, reconstruction or checkpoint. Hover a profile to inspect
its coordinate/value and a linked heatmap line. Keyboard focus and arrow keys
also inspect profile coordinates. Heatmap hover links both directions when no
editing tool is active; painting and sensor placement retain priority.

For the playground's periodic endpoint-excluded arrays `u[i][j]`:

```text
m_x[i] = (Ly / Ny) sum_j u[i][j]
m_y[j] = (Lx / Nx) sum_i u[i][j]
x[i] = x0 + i Lx/Nx; y[j] = y0 + j Ly/Ny
```

The nodal periodic rectangle sum includes the wrap interval once. No endpoint
is duplicated in integration. Curves repeat the first point at the domain end
only to draw that interval. Signed/background values remain intact; a constant
c gives c Ly and c Lx. Positive and negative cosine contributions can cancel.
These curves are integrals, not fitted Gaussians, histograms, cross-sections or
probability densities. Magnitude uses `abs(u)` in the same sums.

The reusable SVG profiles and canvas share CSS grid edges and physical aspect
ratio, with y increasing upwards. The periodic image is offset/wrapped by half
a display cell so nodal peaks align with physical coordinates and sensor markers.
Colour bars and captions are outside that alignment region. Profiles occupy about
20% of the heatmap width/height. Narrow layouts preserve the arrangement, with
fewer annotations and no horizontal page overflow in the tested 390px viewport.

Profiles use the current field before rasterization: all 64² reference nodes,
including in the coarser model comparison; the model's available query nodes;
and the live painting draft. Paused scrubbing uses the already requested
Fourier/latent field. Playback integrates the existing interpolated display field.
No solver/model calls, new inference samples or scale-estimation calls are added.
Reference/model value scales are shared per direction and include all cached
playback fields plus current/drawn values. Scales can expand when newly evaluated
values exceed the cache; edits/mode changes can establish a new range. A display
scale floor of 1e-12 times field amplitude times the largest domain length prevents
floating-point cancellation noise filling the plot; it does not alter integrals.
Zero lines and numeric ranges distinguish signs without relying on colour.
Error and measurements-only panels omit profiles.

Limitations: this component explicitly supports the playground's uniform periodic
nodal grid, not nonperiodic node quadrature. Model profiles approximate integrals
on its available query grid; finer display does not establish scientific accuracy.
Playback remains cached temporal interpolation. No scientific/model changes or
training were performed.

## Verification on 28 October 2026

Evidence: `outputs/spatial-profiles-20261028/`.

```bash
cd frontend
npm test -- src/heatProfiles.test.ts src/heat.test.ts src/HeatPlayground.test.tsx
npm run build
npm run lint
# With the existing API and Vite running on 8000/5173:
npx playwright test e2e/spatial-profiles.spec.ts --workers=1
```

- 13 focused tests passed: signed constants/physical widths on 7×11 nodes and a 2×5
 domain, finite-domain Gaussian integral/shape, two peaks/shared coordinate,
 cosine cancellation/wrap, explicit magnitude/immutability, common interval
 scales/decay, periodic drawing and initial component render.
- TypeScript/Vite build passed. Existing large legacy bundle warning remains.
 Lint has zero errors and three existing fast-refresh warnings in shadcn UI files.
- Actual Chromium field journey checks numerical SVG coordinates against real
 API arrays, pixel hot-spot alignment, exact-time/physical D updates, playback
 curve changes with stable scales, presets/random fields, live drawing/erasure,
 sensor placement/measurements, mode/toggle preservation, signed constant/cosine,
 non-square shifted domain, invalid draft handling and 390px layout.
- Actual Chromium model journey imports the existing 2-update pipeline-check
 checkpoint, runs one real CPU reconstruction and checks full-reference/model
 profiles, shared scales, arbitrary-time decoding and visual changes without
 resampling. This is numerical integration, not mocked inference or accuracy
 validation. Source checkpoint:
 `outputs/continuous-time-20261028/local/pipeline/export/coordinate.pt`.
- Screenshots: `hot-spot.png`, `two-hot-spots.png`, `mobile-signed.png`,
 `comparison.png`. Actual field samples/pixel alignment: `browser-fields.json`.
 Browser logs: `browser-second.log` (model journey passed),
 `browser-fields-final.log` (final field journey). Earlier failed test attempts
 are retained: cancellation comparisons initially magnified roundoff; a wait
 expected a request for an unchanged input; the non-square expected sum first
 used unit lengths. Product numerical tolerances were not loosened.
- No Python, accelerator, training or unrelated full suites rerun for this
 frontend-only change. Earlier CPU/MPS/CUDA/Colab status remains unchanged.
- `preservation-before.json` / `preservation-after.json` record existing checkpoint
 hashes and logical staged entries. Existing outputs were retained; no Git
 staging, reset, commit or push was performed. Only task-owned verification
 services are stopped after checks.

The successful browser reconstruction job is copied into `browser-job/` (including
its original request, numerical arrays and metrics). Temporary service workspace:
`/private/tmp/pdeflow-profiles-20261028`; it is not needed to use the feature.
`process-check.txt` confirms both verification ports are free. All 48 pre-existing
checkpoint files and the 90 logical staged Git entries are unchanged. The next
scientific implementation/experiment slice remains the bounded representation
quality pilot described in continuous-heat-verification.md; this display change
adds no model-quality evidence.
