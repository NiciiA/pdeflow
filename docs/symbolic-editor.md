# Symbolic mathematical requests

The primary input is a real MathLive 0.111.1 mathfield, installed through npm.
It supports visual fractions, derivatives, powers, Greek letters, keyboard
input and LaTeX paste. The toolbar inserts editable notation and returns focus
to the mathfield. Optional LaTeX source views remain synchronized. Multiple
rows describe systems and rate definitions. Fonts are bundled by Vite from
mathlive/fonts.css; sounds and the optional compute engine are disabled.
No runtime CDN is needed for the tested editing workflow. Integration follows
[official MathLive instructions](https://mathlive.io/mathlive/guides/integration/)
and its [API](https://mathlive.io/mathfield/api/). This is an editor, not a
claim that generic MathJSON understands PDEs.

The UI starts with empty equations independently of checkpoint discovery.
Optional presets insert editable equations and declarations. Changing equation
text makes the draft custom. Reapplying a preset retains existing values,
units, roles and meanings of declarations with unchanged name/kind/rank;
measurements and boundary information are preserved. Any insertion into a
nonempty draft or checkpoint-layout replacement requires a visible before/after
review. Removed declarations and changed static/time/initial status appear in
that review. No preset or page load launches training.

## Full request schema v1

𝒞 = (𝓔, λ, Ω, ℬ, ℐ, y, M) is the versioned editing contract:

| Component | JSON / meaning |
|---|---|
| 𝓔 | equations: ordered LaTeX/PDE-source rows |
| λ and unknown declarations | declarations: name, kind (field/parameter/coefficient/definition), scalar/vector rank, value, units, meaning, information status |
| Ω | domain: spatial shape, origins, lengths, Cartesian/other geometry, boundary convention, periodic-node convention, saved physical times |
| ℬ | boundary: type, information status, descriptive details |
| ℐ | initial: information status, descriptive details |
| y | measurements: physical-coordinate rows, noise_model, noise_std, information status |
| M | observation_operator: kind, information status, descriptive details; locations are in measurement rows |

Root keys also include schema_version=1, units, optional preset origin and
optional known_fields (Darcy requires the explicit mean-zero cell array f).
Information statuses are known, partially_known, unknown and not_applicable.
Unknown fields and coefficients/parameters are explicit declarations; vector
components follow spatial x/y/z order. Units are required per declaration;
current checkpoint execution requires the documented nondimensional scale.
Boundary/initial/operator details are descriptive notes, not numeric constraint
arrays. Known/partial initial constraints and other boundary implementations
are rejected for current inference rather than silently ignored.

Static drafts have domain.times=[] and initial.status=not_applicable. They do
not require artificial initial conditions or a time column in manually edited
static observation rows. Executable heat uses the checkpoint's exact saved
physical times and on-grid space/time measurements. Physical parameter λ is
separate from numerical guidance weights in Reconstruction. Spatial coordinate
y is a location; the observation vector y is the measurements section. Meanings
such as pressure versus probability density come from declarations, not names.

The complete-request JSON view and visual forms update one request state.
Malformed JSON is retained visibly and blocks validation. Export/import retains
unsupported or incomplete equations and all information sections, without
claiming execution support. Observation CSV/JSON continues to use the existing
strict heat, static Darcy and chemistry adapters. Darcy rows omit time and use physical
a/u names at cell centres. Other unsupported system drafts can edit/export rows
but do not gain an execution adapter.

## Parser subset and limits

Three separate representations exist: editable source; parsed, rank-checked
PDE nodes; executable legacy ProblemSpec. Python's explicit tokenizer and Pratt
parser are authoritative. There is no eval, Python code compilation, generic
symbolic execution, or automatic neural equation understanding.

Supported source includes:

- Declared scalar fields/coefficient fields/parameters; declared vectors for
  vector calculus and dot/convection notation. Names include ASCII identifiers
  and subscripts such as D_A, c_w, k_f; Greek ν, λ, ρ and LaTeX aliases are
  supported. ν is never silently changed to velocity v.
- Finite numeric constants; +, -, multiplication (*, scalar cdot, or juxtaposed
  symbols), division/fractions, parentheses and integer scalar powers 0–3 (vector powers are rejected).
- Physical derivatives u_τ, u_{τ}, ∂τ u, ∂u/∂τ,
  `\partial_\tau u`, `\frac{\partial u}{\partial \tau}`. ASCII t aliases physical
  τ only when a physical-time axis exists. u_x/u_xx and explicit partial
  fractions work for declared spatial coordinates. Second physical-time
  derivatives use u_{ττ}, `\partial_{\tau\tau}u`, or
  `\frac{\partial^2 u}{\partial \tau^2}`.
- `\nabla u` / grad(u), `\nabla\cdot(v)` / div(v), `\Delta u` / lap(u), explicit
  vector dot products, and `(v·∇)v` / adv(v,v). The convective form currently
  requires two declared vectors. grad accepts scalars; div accepts vectors;
  dot products and equation/addition ranks are checked. Vector-vector ordinary
  multiplication is rejected.
- Multiple equations and explicitly declared definition symbols such as
  r=k_f a b-k_r c. Each definition has one row; cyclic definitions are rejected.
  Definitions remain explicit AST nodes, not unchecked textual substitution.
- Canonical source output uses parenthesized arithmetic and explicit operator
  functions (dt/dtdt or dτ/dτdτ, dx/dxdx, etc.); supported ASTs roundtrip.

Declared subscripted symbols take precedence over derivative shorthand. For
example, if D_x is declared, D_x means that coefficient; use an explicit partial
fraction to differentiate D instead. Operator and coordinate names are reserved.
A derivative along z in a 2D domain is rejected. Functions such as sin, mixed
partial derivatives, integrals, arbitrary LaTeX macros, matrix/tensor operators
and unrestricted symbolic equivalence are unavailable. Errors preserve source
and name the failing construct. Limits: eight rows, 4096 characters and 512
tokens per row, recursive expression depth 32, and 32 declarations. The API
also applies its 2 MB request limit.

Execution recognition is deliberately narrow: scalar u_τ=D Δu or
u_τ-D Δu=0 (including the supported derivative notation variants), with only
unknown scalar field u and known positive scalar D. D maps to the existing
known diffusivity condition. It also requires nondimensional units, known
periodic Cartesian boundaries, unknown initial state, the saved grid/time
layout, and the implemented Gaussian/on-grid measurement model. Rearranging
factors or adding unsupported advection/forcing never invokes the
original heat preset. Darcy explicitly recognizes -div(a grad(u))=f and
-div(a grad(u))-f=0 with unknown positive spatial a/u, known compatible f,
static cell-centred 2D/3D grids, no initial state and known periodic boundaries.
The current full request, not the preset label, drives routing. See
multispecialist.md for compatible checkpoint selection/exclusion semantics. A syntactically valid custom AST may lack a physical
residual, reference generator or checkpoint adapter.

## Support and execution boundary

The UI distinguishes editable source, parsed source, physical residual,
reference/data generator, checkpoint compatibility and validated reconstruction.
Heat, Darcy and A+B reversible C chemistry have implementations; other presets are editable/parsed
systems without execution adapters. Poisson's standalone numerical
solver remains verified separately. See equation-specialists.md for extension
requirements. Unknown checkpoint training ranges stay labelled unknown.

GET /api/mathematics/presets returns template metadata. POST
/api/mathematics/parse takes {request, model_id?}, returning parsed nodes,
canonical source, recognition, optional ProblemSpec, status and reasons.
Browser reconstruction requests include mathematical_request. Validation and
job submission parse it again, require equality with the executable ProblemSpec,
and check measurement/noise agreement. Typed structural equality tolerates
JavaScript's 0.0→0 numeric spelling without weakening the equation comparison.
Legacy CLI/API clients can still supply their original ProblemSpec directly.
Workers store the full mathematical request for provenance but receive no
reference field. No stale successful parse is substituted for invalid current
source. Duplicate MathLive input events do not invalidate a parse indefinitely.

## Implemented chemical system recognition

The four explicit species/rate equations now route to chemistry, independent
of preset labels and row order. Species residual forms lhs-(rhs)=0 are also
recognized; the rate definition remains r=k_f a b-k_r c. Fixed symbol names are
required; consistent renaming or arbitrary kinetic algebra is not supported.
Edited unsupported kinetics preserve the draft and invalidate execution.
Known positive D_A/B/C and nonnegative k_f/k_r are validated. Hidden initial
species/unknown kinetic parameters are not accepted as known conditions.
Checkpoints declare three physical channels, exact times and the square-root
transform. Generator/residual support remains separate from available
checkpoint support; data-mode UI controls export CLI configuration only.
