# 04 — Regression

Regression models the relationship between a continuous (or count) outcome and one or more predictors. This folder progresses from simple OLS through generalised linear models and regularised regression, with consistent attention to model diagnostics and evaluation.

All inference is done with `statsmodels` (for CIs, p-values, and diagnostic tools); `sklearn` is used for prediction pipelines and regularisation.

---

## Notebooks

### `linear_regression.ipynb`
OLS regression with `statsmodels.OLS`. Interpreting coefficients, R-squared, and residual SE. Fitting a regression line with confidence band. Distinguishing prediction intervals (for a new observation) from confidence intervals (for the mean response). Residuals vs fitted and scale-location plots. Breusch-Pagan heteroscedasticity test. Pitfalls: R-squared as correctness, CI vs PI confusion, extrapolation, skipping residual plots, coefficients without CIs.

### `multiple_regression.ipynb`
Multiple OLS regression: partial effects, VIF for multicollinearity detection, standardised coefficients for cross-predictor comparison. Cook's distance for influential observations. Practical guidance on when to drop or combine correlated predictors. Pitfalls: marginal vs partial effect confusion, collinearity without VIF check, raw coefficient comparison across scales, ignoring Cook's distance, adjusted R-squared vs cross-validation.

### `logistic_regression.ipynb`
Binary logistic regression with `statsmodels.Logit`. Log-odds coefficients, odds ratios, and predicted probabilities. Marginal effect curves across predictor ranges by group. ROC curve and AUC-ROC. Calibration plot. Absolute risk differences and relative risk alongside odds ratios. Pitfalls: ORs without predicted probabilities, 0.5 threshold assumption, AUC without calibration, complete separation, direct interpretation of log-odds.

### `poisson_regression.ipynb`
GLM Poisson regression with `statsmodels.GLM`. Log-link, rate ratio interpretation, and `log(effort)` offset for variable sampling effort. Overdispersion diagnosis (Pearson chi-squared / df). Quasi-Poisson with corrected SEs. Pearson residual plots and observed vs predicted counts. Pitfalls: no overdispersion check, omitting the effort offset, interpreting log coefficients directly, ignoring excess zeros, Poisson for bounded counts.

### `regularisation_ridge_lasso.ipynb`
Ridge (L2), LASSO (L1), and Elastic Net regularisation with sklearn. `RidgeCV` and `LassoCV` for cross-validated alpha selection. Coefficient path plots. Comparing true vs estimated coefficients with 20 predictors (5 true signals, 15 noise). Pitfalls: no feature standardisation before regularisation, scaling on full data before split, LASSO for correlated predictors, LASSO zero coefficients as confirmed irrelevant, alpha tuned on test set.

### `model_evaluation_regression.ipynb`
Training vs test metrics: RMSE, MAE, R-squared. 5-fold cross-validation with `cross_val_score`. Bias-variance tradeoff visualised through polynomial degree vs train/test RMSE. Residuals vs predicted, actual vs predicted, and Q-Q plots on the test set. Pitfalls: training-only evaluation, R-squared for predictor count comparison, test-set-guided model selection then reporting that metric, no test-set residual plots, RMSE without scale context.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels, sklearn
```

## Data
Simulated ecological dataset: species richness and count outcomes modelled as functions of elevation, water chemistry, and treatment, with variable survey effort for Poisson models.
