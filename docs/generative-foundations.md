# Generative foundations and the implemented model

The seven supplied KTH lecture PDFs were read from the user's Downloads folder;
all pages were extracted for inspection and representative mathematical pages
were rendered. Originals were not changed or copied into distributable source.
References below use **physical PDF page numbers**, starting at one. Some printed
slide numbers differ. These are teaching sources for background, not validation
of our PDE adaptation or a direct derivation of conditional flow matching (CFM).

| Supplied PDF | Relevant physical pages | Concept | PDEFlow status/mapping |
|---|---|---|---|
| 1_Intro_to_Generative_Models.pdf | 5–9 | Sampling, density estimation and conditional generation are distinct goals | Implemented: sampling from Gaussian model coordinates through sampling.sample; core.Conditioning contains disclosed coefficients/forcing. Exact density estimation is absent. |
| 2_1_Intro_to_Variational_Auto_Encoders.pdf | 4–7, 10–15 | Latent representation, implicit generation; deterministic bottleneck AE and reconstruction objective | The fixed-grid baseline remains direct-field; the27 October optional coordinate specialist now trains a deterministic reference AE with reconstruction supervision, then freezes it for latent CFM. It is not a VAE. U-Net pooled features alone remain distinct from a trained AE. |
| 2_2_VAE_Steps.pdf | 3–7, 10–18 | Latent prior, approximate posterior, decoder likelihood, variational objective, amortization and reparameterization | Background. No VAE, posterior encoder, ELBO or KL regularizer exists in PDEFlow. Sampling a training source/time is not a VAE reparameterized latent posterior. |
| 2_3_VAE_How_to_Train_and_Use.pdf | 1–6, 12–15 | Gaussian VAE training, prior→decoder sampling, decoder variance/mean choice, lower-bound and importance estimates | Background. Decoder mean output and a sample from its likelihood are different distributions. PDEFlow has no such decoder likelihood and does not report a likelihood from its velocity MSE. |
| 2_4_VAE_An_Extension.pdf | 1–3 | Beta-weighted variational objective and representation motivation | Background only. Neither diagonal Gaussian priors nor beta weighting establish disentanglement; no beta/ELBO term was added to CFM. |
| 3_AutoRegressive_Generative_Models.pdf | 2–3, 6–11, 21, 26–30 | Chain-rule factorization, conditional factors and sequential sampling | Background / alternative family, not implemented. No autoregressive pixel/time ordering is hidden in fixed physical-time channel packing. |
| 4_Normalizing_Flows.pdf | 3–8, 15–16 | Invertible transformations, change of variables, Jacobian determinants and sampling | Background comparison. Our learned ODE supplies samples; discrete likelihood-trained invertible layers and exact density evaluation are not implemented. |

For a deterministic AE, a reconstruction loss trains compression/reconstruction;
for a VAE the objective includes an expected decoder log likelihood and a KL
penalty between the approximate posterior and prior. The latter is not the loss
of our direct-field CFM. A future latent-flow proposal must separately train and
verify its encoder/decoder, persist their dependency, and measure field, spectrum
and physical-derivative distortion before attributing errors to the flow model.

For a differentiable bijection x=f(z), change of variables gives
p_X(x)=p_Z(f^{-1}(x)) |det Df^{-1}(x)|. A regular continuous velocity can define
an ODE transport under additional existence/uniqueness conditions. CFM instead
learns a conditional velocity by regression along a chosen probability path;
our production loss is mean squared error against U_model−Z, not a discrete
normalizing-flow maximum-likelihood/Jacobian objective. The package does not
implement divergence integration for continuous densities, and physical decoding
can fold/project coordinates (chemistry/Darcy). No exact density or posterior
claim follows from successful ODE sampling.

The independent Gaussian path and endpoint/guidance conventions are derived in
[flow-matching-audit.md](flow-matching-audit.md); use the original Flow Matching
paper attribution in [mathematics.md](mathematics.md) rather than attributing
that algorithm to the normalizing-flow lecture. FK's numerical target, stochastic
extension and weighting/resampling assumptions remain separately attributed in
[fk-mathematics.md](fk-mathematics.md). No new diffusion, AE/VAE, autoregressive,
normalizing-flow likelihood objective or ELBO was implemented for this task.

Why a sample can look unlike an entered formula: a physically conditioned prior
models a distribution over unknown trajectories, not a deterministic forward
prediction from a hidden initial condition. Paired inverse error requires disclosed
measurements; unconditional/physical-coefficient conditioning should be assessed
with distribution statistics. Observation geometry/noise, identifiability,
training distribution, optimization, integration and heuristic guidance all matter.
The new diagnose-learning command separates fixed-pair memorization, a tractable
Gaussian toy, prior statistics/spectra/decay and fixed-validation inverse seeds.
None is a substitute for untouched held-out scientific evaluation.

## 27 October continuation

The optional deterministic AE/latent-CFM branch now implements the representation
concept that was previously optional. Its reference encoder is used for supervised
training and AE evaluation; inverse inference uses measurement-set conditioning.
No VAE posterior, ELBO/KL, autoregressive factorization or density estimator was
added. New math, attribution and limitations: coordinate-specialist.md and the
optional coordinate section in mathematics.md. Earlier future-tense statements
above describe the original direct-field slice.
