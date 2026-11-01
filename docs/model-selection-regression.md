# Empty model selection: reproduced failure and contract fix

The preceding saved frontend build was actually served before rebuilding current
source. In Chromium with an empty registry: insert heat → import manual CSV →
configure → Validate complete request. The network body contained model_id="".
Evidence: outputs/flow-workspace-20261026/legacy-empty-id-browser.json,
legacy-browser.log and legacy-empty-registry.png. The strict legacy pattern correctly
rejected it. This was not repaired by inventing IDs or accepting an empty string.

The cause crossed two states: SymbolicEditor could compile a valid physical problem
with no compatible checkpoint and call onRoute(""); App's mathReady represented
only executable mathematics. The serializer then sent that empty derived modelId.
A second real-browser race after reference sampling allowed validation during its
pending mathematical parse; it now waits and shows loading. Preserved invalid
advanced JSON is explicitly invalid, not an endless loading state.

## Contract

- New request: model_selection={mode:"auto",id:null}, or
  {mode:"explicit",id:"<registered32hex>"}. It cannot be combined with legacy
  model_id. Automatic mode has no explicit ID.
- Backend resolves Automatic using the **current problem and full inference
  configuration**, registry content IDs and exact compatibility checks. The initial repair selected
  compatible IDs lexically. The subsequent catalogue continuation prioritizes
  optimizer updates, completed-budget provenance and export role, with no
  reference/error-based quality ranking.
- Explicit selection must exist and be compatible; stale/incompatible IDs never
  fall back to another model. Legacy valid32hex model_id remains accepted.
  Legacy empty model_id is still pattern-invalid.
- Job request/manifest records the resolved model_id and selection rationale.
  Worker independently verifies the file's content ID at loading. Discovery retires
  replaced IDs rather than executing new weights under an old selection.
- Submission is gated by current parsing/validation; the backend still revalidates
  independently. A ref guard prevents double submission and the durable token
  remains idempotent. Empty/unsupported/stale/wrong-layout requests create no job.

Human-readable error.message is separate from expandable request_id, field paths,
codes and backend details. Missing model, stale ID, unsupported mathematics, pending
loading and backend disconnection retain draft/observations. Reference solving
requires no checkpoint; the old explicit synthetic shortcut now also accepts a
complete supplied physical problem without an empty checkpoint ID.

Browser proof includes the actual empty registry, delayed real registry/parse
responses, stale persisted explicit ID, auto submission, no premature job, and
clear/refresh after late responses. Python covers the legacy schema error, absent
registry, auto/explicit/stale selection and wrong layout. Full regression and
real numerical journeys are recorded in flow-workspace-verification.md.

Training suggestions in the equation editor now match declared family, exact
unit periodic domain, grid and saved times against actual packaged profiles.
An unmatched layout is preserved with a configuration explanation; the UI no longer
suggests a32² preset as though it would satisfy a1024² request. Coefficient ranges
and measured resource limits still require explicit review.
