# 10 — Statistical Modelling

This folder covers confirmatory statistical modelling in Python using `statsmodels`, `pyGAM`, and `lifelines`. It is the Python counterpart to the R library's regression, mixed effects, GAM, and survival folders — honest about where Python tools are fully capable and where R remains the stronger environment.

---

## Notebooks

### `ols_glm.ipynb`
OLS and GLM families via `statsmodels` formula interface. Linear regression with categorical factors using `C()`. Logistic regression with odds ratios. Poisson GLM with overdispersion check (Pearson chi2/df ratio). Negative Binomial for overdispersed counts with AIC comparison. Gamma GLM (log link) for right-skewed positive responses. Pitfalls: OLS for binary/count outcomes, ignoring overdispersion, link-scale coefficient interpretation, deviance/df check, C() reference level.

### `mixed_effects.ipynb`
`statsmodels.mixedlm` for Gaussian mixed effects models. Random intercept model with ICC computation. Random intercept + random slope via `re_formula`. Two-level nesting (sites within catchments) with BLUP visualisation. Model comparison: fixed-only vs random intercept via ML likelihood ratio test (boundary-corrected). Pitfalls: REML vs ML for LRT, fixed vs random effects choice, ICC before committing to mixed model, GLMMs need pymer4/glmmTMB, over-specified random structures.

### `gams_pygam.ipynb`
Generalised Additive Models with `pyGAM`. `LinearGAM` with `s()` smooth and `l()` linear terms. Partial effect plots with 95% CI for each smooth. Smoothing parameter selection via GCV grid search. Comparison of oversmoothed/optimal/undersmoothed fits. `LogisticGAM` and `PoissonGAM` for non-Gaussian responses. Honest note on pyGAM vs mgcv capabilities. Pitfalls: GAM over GLM when linear suffices, no smooth visualisation, extrapolation, pyGAM vs mgcv limitations, concurvity.

### `survival_analysis.ipynb`
Time-to-event analysis with `lifelines`. Kaplan-Meier estimator with group comparison and log-rank test. Cox Proportional Hazards model with hazard ratios. Proportional hazards assumption check via Schoenfeld residuals and `check_assumptions()`. Weibull AFT model as parametric alternative. Pitfalls: ignoring censoring, no PH assumption test, log-rank with crossing hazards, HR vs RR vs OR confusion, informative censoring.

### `model_selection.ipynb`
AIC and BIC comparison across 8 candidate models. Likelihood ratio test for nested models. Best subset selection across all predictor combinations with AIC/BIC curves. AIC weights for model averaging and uncertainty quantification. Pitfalls: selection as hypothesis test, stepwise without validation, AIC across different response transformations, multicollinearity during selection, no diagnostics after selection.

### `multicollinearity_assumptions.ipynb`
VIF computation and interpretation (threshold 5/10). Coefficient instability demonstration under multicollinearity. Breusch-Pagan heteroscedasticity test with residual plots. HC3 robust standard errors vs OLS SEs. Q-Q plot and Shapiro-Wilk for residual normality. Durbin-Watson autocorrelation check. Pitfalls: removing correlated predictor without domain justification, heteroscedasticity tests at large n, robust SEs without diagnosis, Durbin-Watson for spatial data, uncentred terms before VIF.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels
pygam (optional, for GAMs: pip install pygam)
lifelines (optional, for survival: pip install lifelines)
```

## Data
Simulated riparian monitoring data: species richness, invertebrate counts, and biomass as functions of elevation, nitrate, phosphorus, temperature, and restoration treatment — with known coefficients for validating model estimates.
