# Workbench redesign — 25 October 2026

## Inspection before changes

Inspected the real running workbench via the in-app browser and actual Chromium
screenshots from the local-workflow tests. The equation page is about4200px tall
at1440px width. It simultaneously exposes every declaration's kind/rank/role,
geometry/layout, boundary/initial notes, measurement metadata, long model select,
training output paths/provider internals and full raw request JSON. Inputs have
no coordinated domain/measurement plot. Five sidebar steps split editing and
results into an obligatory next-click sequence. No persisted draft or visible
relationship between changed inputs and an older result exists. The scientific
plots work, but interpretation, missing-reference explanations and cost/status
are buried among ungrouped raw diagnostics. Longer model labels caused narrow
layout overflow; that regression is being closed before redesign.

## Information architecture

Top navigation: Workspace, Runs, Models & Training. Workspace keeps three
freely navigable working stages: Define problem, Add information, Reconstruct &
inspect. A compact persistent request map shows equation, parameters/forcing,
domain/resolution, boundaries, initial information, measurements and operator,
with direct edit links and honest missing/default/incompatible statuses. Unknown
targets are named separately. Results remain available when inputs change and
are explicitly marked as belonging to the earlier request.

## Primary journey

Choose an editable supported equation or reopen the saved draft. Read its
specialist-derived interpretation and symbol roles; review domain/conditions
and the actual known-input preview. Add/import observations and inspect their
locations, values and physical-time coverage alongside editable rows. Review
the automatically compatible model. Select method/sample count/device, then
validate and reconstruct. Inspect fields first, coordinated spread/reference/
error where available, followed by grouped physical/observation/process/cost
diagnostics. Return to any section without losing edits. Models & Training
shows real local history and relevant CLI commands when training is needed.

## Components and disclosure

Keep MathLive and the explicit PDE parser. Add request summary, specialist
interpretation/symbol glossary, sparse input preview, model summary, bounded
training-history charts and compatible-run comparison. Expert declaration
editing, complete JSON, conditioning/provider details, model override, numerical
stepper/schedule, guidance weights, controller/FK/refinement and raw logs move
behind named disclosures. Physical parameter values remain conveniently editable.
No preset locks recognition; unsupported source stays editable and cannot run.
Use existing shadcn Button/Card/Dialog/Tabs/Tooltip primitives and Tailwind
foundation with one restrained visual system, consistent sizing and spacing.

## Actual visualization data

New input endpoint reads only submitted ProblemSpec/known arrays/observations,
never hidden reference. Missing field cells are null/hatched, not measured zero.
1D curves/points,2D maps and true3D slices use actual coordinates and saved times.
Darcy forcing is a known field; a/u are separate unknown targets. Chemistry
species roles are distinct from permeability. Table/plot selection shares row IDs.
Result slices continue to use server-side bounded weighted summaries. Charts
read real process/FK traces, measured residuals and managed training history;
comparison requires matching problem and observation identities. No full3D
ensemble is sent as chart JSON. Every plot has axes/units/legend/empty state.

## Accessibility, state and responsiveness

Persist only the editable request/settings locally with a versioned envelope;
keep selected immutable results separate. Ignore stale asynchronous parse/preview
responses; revalidate on every scientific input change. Keyboard-accessible
controls, visible focus, associated labels and semantic text status accompany
colour. Desktop uses an editor/preview arrangement; narrow screens stack at
natural reading width with bounded tables/math and no second competing sidebar.
All input, output and reference types use labels/marker styles in addition to
colour. Verify actual desktop/narrow screenshots and real numerical journeys;
passing a build alone is insufficient.

## Implemented decisions and tradeoffs

The implementation retains the existing shadcn/Tailwind foundation and uses
small labelled SVG views rather than another large chart dependency. Unknown
regions use hatching; observations use orange markers; known inputs use blue;
reconstruction/reference lines differ by colour and dash. Two-dimensional
numerical colour maps retain their explicit ranges. Reconstruction/reference
share a range, while spread/error deliberately have independent labelled ranges.

The full editable request and exact resolved execution settings remain available
behind disclosures. Preset values, model layout suggestions and explicit review
are distinguished. Invalid JSON has its own persisted draft representation and
cannot validate the last valid equation. Reopening a saved result does not replace
the current editable problem. Reset and one-level problem undo are deliberate
actions; this is not a general multi-document undo history.

Input previews are exact node/plane views, not observation interpolation.
Only current supported specialist semantics receive a plain-language
interpretation. Static Darcy does not acquire a fictional physical time.
Training history reads actual optimizer rows and checkpoint events, with
extrema-preserving downsampling and no smoothing. FK event positions use their
actual generation coordinates, including nonuniform schedules. Live charts only
accumulate events actually received while viewing a job; the saved final
diagnostic history is separate.

Responsive QA found clipped narrow stage labels; labels now wrap. Request-map
components scroll horizontally at narrow widths and remain keyboard accessible.
Editable tables have bounded scrolling areas. Large mathematical expressions,
3D slice payloads and charts remain bounded. MathLive still contributes to the
large initial bundle; code splitting is a remaining performance task, not a
claim that every dimension/device combination was benchmarked.

See [actual verification](workbench-verification.md) for the executed journeys,
retained failure evidence, hardware scope and screenshots.

Browser verification also exposed pending-validation cancellation on navigation,
manual validation racing an older debounce, and saved-result restoration
competing with Reset draft. Canonical validation now survives navigation, stale
debounce callbacks exit before updating state, and workspace stage persistence
is separate from selected immutable results. Dedicated frontend and real browser
checks cover these transitions. Baseline and reconstruction error tables have
distinct accessible names.
