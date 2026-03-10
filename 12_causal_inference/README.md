# 12 — Causal Inference

Correlation is not causation — but causal inference provides a rigorous framework for when and how observational data can support causal claims. This folder covers the foundational frameworks, the major quasi-experimental designs, and the sensitivity analyses that determine how much trust to place in a causal estimate.

---

## Notebooks

### `potential_outcomes_dags.ipynb`
Potential outcomes framework: Y(0), Y(1), ATE, fundamental problem of causal inference. DAG construction and backdoor path identification. Confounding bias quantified by comparing naive mean difference to true ATE. Overlap assumption via propensity score plot. Backdoor criterion: adjustment via regression recovers the true effect. Collider bias: conditioning on a downstream variable opens a spurious path. Pitfalls: causal interpretation without a causal model, controlling for colliders, ignoring overlap, untestable ignorability, naive mean difference in observational data.

### `matching_methods.ipynb`
Propensity score estimation via logistic regression. Standardised mean differences (SMD) before and after matching. 1:1 nearest-neighbour matching with caliper (0.2 × SD of PS). Love plot for balance visualisation. ATT estimation from matched pairs with paired t-test CI. Pitfalls: ATE vs ATT estimand confusion, no post-match balance check, caliper too large or absent, matching with replacement without adjusted SEs, treating matching as equivalent to randomisation.

### `difference_in_differences.ipynb`
Panel data setup: 80 sites × 8 periods, intervention at period 5. Parallel pre-trends visualisation and formal interaction test. Basic DiD regression and Two-Way Fixed Effects (TWFE) model. Event study plot: pre-trend coefficients flat, post-treatment effect emerges. Clustered standard errors at the site level. Pitfalls: no parallel trends check, staggered timing with standard TWFE, unclustered SEs, concurrent events as confounders, DiD with a single treated unit.

### `regression_discontinuity.ipynb`
Sharp RD with a nitrate score running variable and cutoff at 5.0. RD plot with bin means and separate linear fits on each side. Linear same-slope, linear different-slopes, and quadratic estimators. Bandwidth sensitivity analysis: LATE estimate and N vs bandwidth h. McCrary-style density test for manipulation at the cutoff. Pitfalls: global polynomial vs local linear, no manipulation test, arbitrary bandwidth choice, no pre-specified covariate placebo checks, extrapolating beyond the cutoff.

### `instrumental_variables.ipynb`
IV framework: relevance, independence, exclusion restriction. First-stage regression and F-statistic test for weak instruments. 2SLS via `statsmodels IV2SLS`. Comparison of naive OLS, manual 2SLS, and correct 2SLS. Weak instrument demonstration: F < 10 → bias toward OLS. Hausman endogeneity test. Pitfalls: weak instrument ignored, exclusion restriction assumed without justification, LATE ≠ ATE, no overidentification test with multiple instruments, manual 2SLS with incorrect SEs.

### `sensitivity_analysis.ipynb`
E-value computation for continuous outcome effect size. Rosenbaum gamma bounds on matched data: p-value at each level of hidden bias. Placebo outcome test: treatment should not predict baseline covariates. Permutation (placebo treatment) test: observed ATE vs null distribution of randomised assignments. Sensitivity dashboard combining all checks. Pitfalls: no sensitivity analysis reported, high E-value as proof of causality, post-hoc placebo test selection, permutation test omitted, sensitivity analysis treated as optional.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels, sklearn
```

## Data
Simulated stream ecology panel: elevation, nitrate, upstream land use, species richness, treatment assignment by threshold rules and propensity scores — with known true treatment effects for validation.
