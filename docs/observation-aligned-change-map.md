# Design change map - 1 November 2026

Authority: the user's concrete implementation request. The attached 16-page
model-design PDF is guidance, including hypotheses/alternatives, not instructions
to implement every mechanism. Its cited review does not prove this adaptation.

| Recommendation | Current evidence | Disposition |
|---|---|---|
| Factorized initial u/v/coefficient, typed graph, observation-conditioned CFM, spatial decoder | Implemented physical-state v2; retained | Already implemented; no new capacity or architecture |
| Exact initial identity and static scalar authority | Residual dt multiplier, typed masks and known-value override | Retain and explicitly test values/derivatives/nontrivial evolution |
| Decoded clean-endpoint observation alignment | Legacy optional penalty exists but is serial, uses implicit sigma flooring, no explicit likelihood policy | In scope: separately configured batched Gaussian/tolerance objective, zero-beta control |
| PDE enforcement through actual T/D | Strong residual evaluation exists, not optional batched A2 training | In scope: optional smooth heat/wave normalized A2 residuals; supervised dynamics remains essential |
| Domain/RB/phase-sensitive spectral objectives | Domain, RB alpha.5, WMSE and MAE exist | Retain; add explicit structured-grid complex-DFT option, disabled unless selected |
| Richer flow-to-observation attention | Conditioning responds weakly in some cases; precise bottleneck unestablished | Deferred architecture ablation; no untrained attention graft |
| Direct observation-conditioned decoder | Could bypass coherent initial-state/evolution semantics | Deferred; no scientific evidence justifies it |
| Stage C joint refinement / inference geometry redesign | Exact Medium E/T missing; tiny baseline is not accuracy evidence | Deferred until representation/evolution errors measured and alignment/recalibration designed |
| Correction as normal posterior sampling | Single-case observation-only optimization improves errors but changes samples | Remains explicit diagnostic; no calibration/exact posterior claims |
| Shocks, weak solutions, forcing, new geometry/families | Unsupported in current smooth periodic2D specialist | Deferred; preserve broader1D-3D chemistry research roadmap |
| Complete diagnostic and resumable artifacts | Inference export deliberately omits E; content-addressed persistence exists | In scope: verified manifest views sharing canonical files, complete state/encoder/dependencies, optional download archives |
| Persistent saves during training and fresh-runtime restore | Save callback tied to coarse checkpoints; local resume can ignore newer authoritative remote state | In scope: successful-update policy, first checkpoint/stage barriers, failure status, verified restore, unique IDs, final/restore notebook cells |

Objective changes, optional physics/spectral terms and persistence policy are
recorded separately. Defaults preserve legacy objectives. Experimental likelihood
settings are explicit choices and are not asserted optimal. No substantial
training, automatic gate override, source-hash edits or accuracy claim is authorized.
