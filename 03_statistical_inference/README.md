# 03 — Statistical Inference

Statistical inference draws conclusions about populations from samples. This folder covers the core frequentist hypothesis testing toolkit, with consistent emphasis on effect sizes, confidence intervals, and the distinction between statistical and practical significance.

The guiding principle throughout: a p-value answers whether an effect is detectable. An effect size and confidence interval answer whether it matters.

---

## Notebooks

### `t_tests.ipynb`
One-sample, two-sample (Welch), and paired t-tests with `scipy.stats`. Confidence intervals via `stats.t.interval()`. Cohen's d and Hedges' g effect sizes. Q-Q plots and Levene's test for assumption checking. Quantifying the power loss from ignoring pairing. Pitfalls: Student's t-test by default, p-value-only reporting, two-sample test on paired data, Levene-then-choose procedure, non-significant = no difference.

### `anova.ipynb`
One-way ANOVA with `scipy.stats.f_oneway()`. Omega-squared effect size (bias-corrected). Tukey HSD post-hoc with `statsmodels`. Two-way ANOVA with interaction using `statsmodels OLS` and Type III sums of squares. Residual diagnostics: residuals vs fitted, Q-Q plot. Pitfalls: no post-hoc tests after significant F, interpreting main effects when interaction is significant, standard ANOVA for unequal variances, Type I SS for unbalanced designs.

### `chi_square_tests.ipynb`
Goodness-of-fit test and test of independence with `scipy.stats`. Cramer's V effect size. Fisher's exact test for small expected cell counts. Stacked bar visualisation for contingency tables. Pitfalls: chi-square with expected counts below 5, no effect size reported, proportions instead of counts, Yates' correction by default, confusing GOF with independence test.

### `confidence_intervals_bootstrap.ipynb`
Parametric CIs (t-distribution). Wilson interval for proportions. Bootstrap percentile CIs for any statistic (mean, median, correlation). BCa (bias-corrected accelerated) bootstrap via `scipy.stats.bootstrap`. CI width vs sample size. Pitfalls: misinterpreting CIs as probability statements, normal-approximation CI for proportions, B=100 resamples, percentile bootstrap for skewed statistics, not stating the CI method.

### `nonparametric_tests.ipynb`
Mann-Whitney U (two independent samples), Wilcoxon signed-rank (paired/one-sample), and Kruskal-Wallis (three or more groups) with rank-biserial correlation and eta-squared effect sizes. Comparison of paired vs independent test power. Post-hoc guidance for Kruskal-Wallis. Pitfalls: assumption-free misconception, Mann-Whitney as median test, low power for small n, no post-hoc after Kruskal-Wallis, heavily tied data.

### `effect_sizes.ipynb`
Cohen's d, Hedges' g, omega-squared, Cramer's V, rank-biserial r. Visualising distribution overlap by effect size. Bootstrap CIs for effect size estimates. Cohen's benchmark thresholds and their domain-specific limitations. Pitfalls: no effect sizes reported, misapplying Cohen's benchmarks, eta-squared without bias correction, Cohen's d for skewed data, no CIs for effect sizes.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels
```

## Data
Simulated ecological dataset: site-level species richness, water chemistry, and treatment indicators across control and restored catchments.
