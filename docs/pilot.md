# Bounded local pilot: actual results

One seed, two fixed test trajectories, four independent samples. These measurements do not establish convergence or calibrated uncertainty.

| Updates | Fixed training objective | Fixed validation objective |
|---:|---:|---:|
| 0 | 2.262254 | 2.332563 |
| 100 | 0.775055 | 0.879483 |
| 500 | 0.453315 | 0.574266 |

| Updates | Method | Mean unobserved relative L2 | Observation RMSE | FD residual RMSE | Batched model calls |
|---:|---|---:|---:|---:|---:|
| 0 | unguided | 1.120581 | 1.248955 | 80.888254 | 48 |
| 0 | observation | 1.125430 | 0.595033 | 79.175511 | 71 |
| 0 | observation_pde | 1.109725 | 0.588626 | 71.010315 | 71 |
| 0 | fourier_inverse | 0.236383 | 0.134789 | 0.037614 | 0 |
| 100 | unguided | 0.949002 | 1.031933 | 45.298202 | 48 |
| 100 | observation | 0.756363 | 0.506007 | 44.552044 | 71 |
| 100 | observation_pde | 0.751510 | 0.501658 | 41.647243 | 71 |
| 100 | fourier_inverse | 0.236383 | 0.134789 | 0.037614 | 0 |
| 500 | unguided | 1.632448 | 1.449810 | 13.504936 | 48 |
| 500 | observation | 1.190294 | 0.844532 | 13.580540 | 71 |
| 500 | observation_pde | 1.189886 | 0.844549 | 13.259680 | 71 |
| 500 | fourier_inverse | 0.236383 | 0.134789 | 0.037614 | 0 |

Settings were selected on two validation cases at 500 updates from four guidance settings and two ridge values: observation weight 1, PDE weight .01, Fourier ridge .1. They were frozen across all test budgets. The model uses 16x16 nodes and six physical times. Detailed per-case/individual-sample errors, physical diagnostics, spread, memory and measured runtime are in metrics.json.

The validation objective decreases, but the 500-update model has worse test reconstruction than the 100-update model. The inverse baseline remains much better. Undertraining, representation and guidance limitations are hypotheses; this pilot does not isolate their causes. No best checkpoint was selected on test error.

The full study took 31.61 seconds including case reconstructions and figure exports. Increment timing includes checkpoint/data load and fixed-noise objectives; it is not pure optimizer throughput. A separate 10-update probe took .799 seconds including initialization/loading, projecting an upper estimate of about 40 seconds for 500 updates; this justified the bounded ladder.

Outputs: outputs/workbench-20261021/budget-study/ with fixed_inputs/, selection.json, study.json, objectives.json, metrics.json, per-budget checkpoints/results, learning_curves.png, quality_budget.png, quality_cost.png and ensemble_spread.png. The quality-cost plot counts batched forward invocations; it excludes offline training/data costs, which are reported separately.

Further evaluation should repeat training/sampling seeds, enlarge validation/test sets and compare observation operators before drawing scientific conclusions. The next implementation module is the conservative 2D/3D Darcy slice specified in darcy.md, followed by substantive chemistry.
