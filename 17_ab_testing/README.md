# 17 — A/B Testing

Rigorous experiment analysis from study design through reporting. Covers frequentist and Bayesian inference, multiplicity, design optimisation, and effect size communication — applied to ecological restoration trials comparing restored vs control riparian sites.

---

## Notebooks

### `power_analysis.ipynb`
`TTestIndPower` and `NormalIndPower` from statsmodels. Sample size curves vs effect size for two power levels. Proportions test power with Cohen's h. Achieved power for fixed n=30. ANOVA power with `FtestAnovaPower` across k groups. Bootstrap power simulation for any complex design. Pitfalls: underpowered study effect sizes, 80% as default, post-hoc power, attrition/clustering inflation, MDE vs practical relevance.

### `frequentist_tests.ipynb`
Levene variance test before t-test selection. Welch t-test with Cohen's d. Mann-Whitney U with rank-biserial r. Proportions z-test with Wilson CIs. Paired t-test (before/after). One-way ANOVA with Tukey HSD post-hoc. Confidence interval forest plot. Pitfalls: p-value interpretation, paired vs independent, equal-variance assumption, post-hoc without correction, p-value without effect size.

### `bayesian_ab.ipynb`
Beta-Binomial conjugate model for binary outcomes. Prior/posterior visualisation. Monte Carlo P(treatment > control). Sequential updating of posterior probability. Normal-Normal model for continuous outcomes. ROPE (Region of Practical Equivalence) analysis with 3-region decision. Pitfalls: flat priors, stopping rule inflation, credible vs confidence interval, MC sample size, ROPE chosen post-hoc.

### `multiple_comparisons.ipynb`
Simulated multi-site experiment with known true/null effects. Bonferroni, Holm, Benjamini-Hochberg compared: rejections, TP, FP, observed FDR. p-value histogram, Q-Q uniform diagnostic, volcano plot. Adjusted p-values table. FWER simulation vs number of tests across methods. Pitfalls: Bonferroni on exploratory tests, BH for individual confirmation, unspecified family, cross-study correction, adjusted p > 0.05 as null result.

### `experiment_design.ipynb`
CRD vs RBD power comparison (catchment as blocking factor). 2×2 factorial ANOVA with interaction plot. Stratified randomisation — balance table before/after. O'Brien-Fleming sequential boundaries with interim analysis simulation. Pitfalls: no stratification with strong prognostic factors, ignoring blocks in analysis, testing before examining interaction, unspecified interim rules, unit of randomisation vs observation.

### `effect_sizes_reporting.ipynb`
Cohen's d, Hedges' g, rank-biserial r for continuous outcomes. Distribution overlap visualisation. Binary effect sizes: ARR, RR, OR, NNT with log-normal OR CI. Bootstrap CI for Cohen's d. Effect size forest plot. Complete results reporting template. Pitfalls: p-values without effect sizes, Cohen's benchmarks as universal, OR vs RR at high baseline, asymptotic CI at small n, effect size without CI.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy
statsmodels (pip install statsmodels)
```
