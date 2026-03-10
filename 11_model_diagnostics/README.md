# 11 — Model Diagnostics

Fitting a model is not the end of the analysis — it is the beginning of checking whether the model is trustworthy. This folder covers the diagnostics that sit between fitting and reporting: residual checks, validation strategy, calibration, interpretability, and hyperparameter selection.

---

## Notebooks

### `residual_diagnostics.ipynb`
The four core diagnostic plots: Residuals vs Fitted, Q-Q, Scale-Location, and Residuals vs Leverage. Studentised residuals, leverage, and Cook's distance via `OLSInfluence`. Formal tests: Breusch-Pagan heteroscedasticity, Shapiro-Wilk normality, Durbin-Watson autocorrelation. Side-by-side comparison of well-specified and heteroscedastic models. Cook's distance stem plot with 4/n threshold. Pitfalls: skipping diagnostics at high R-squared, normality tests at large n, auto-removing influential points, Scale-Location ignored, no autocorrelation check for ordered data.

### `cross_validation.ipynb`
k-Fold and Stratified k-Fold (always use stratified for classification). Group k-Fold to prevent leakage from clustered data. Pipeline-based preprocessing inside CV vs the leaky alternative — demonstrated empirically. `cross_validate` for simultaneous train/test scoring. Bias quantification for the leaky approach. Pitfalls: random k-fold for time/group data, preprocessing outside pipeline, CV mean without SD, LOO-CV at large n, tuning and reporting the same CV score.

### `bias_variance_tradeoff.ipynb`
Polynomial degree sweep from underfit to overfit. Empirical bias-variance decomposition via bootstrap. Learning curves for underfit and reasonable models: train/val MSE convergence patterns. Ridge regularisation path: CV MSE vs alpha. Pitfalls: diagnosing overfitting from training accuracy only, adding data to fix high bias, test set used for regularisation tuning, ignoring CV variance when comparing models, assuming regularisation always helps.

### `calibration.ipynb`
Calibration curves (reliability diagrams) for Logistic Regression, Random Forest, and Naive Bayes. Platt scaling and isotonic regression post-hoc calibration with held-out calibration set. Expected Calibration Error (ECE) implementation. Brier score and Brier Skill Score relative to climatological baseline. Pitfalls: AUC-only deployment readiness, calibrating on training data, isotonic calibration at small n, flat curve as apparent good calibration, no recalibration after prevalence shift.

### `interpretability_shap.ipynb`
Permutation importance on test set (corrects for impurity bias). Partial dependence plots + ICE curves via `PartialDependenceDisplay`. SHAP TreeExplainer: beeswarm summary plot, mean absolute SHAP global ranking, local waterfall decomposition for individual predictions, dependence plot for interaction detection. Pitfalls: impurity-based vs permutation importance, PDP with strong interactions, SHAP as causal effects, explaining a poor model, global SHAP without local examination.

### `hyperparameter_tuning.ipynb`
Grid search vs random search: score distribution comparison at matched compute budget. Halving random search for larger spaces. Nested cross-validation to quantify optimism bias from non-nested tuning. `HalvingRandomSearchCV` round-by-round resource allocation. Pitfalls: tuned CV score as test performance, best-of-1000 overfitting to CV noise, no random_state for reproducibility, tuning before establishing a baseline, transferring parameters across datasets.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels, sklearn
shap (optional, for SHAP values)
```

## Data
Simulated ecological regression and classification datasets: species richness as a function of elevation, nitrate, phosphorus, pH, and treatment assignment, with known ground-truth coefficients for diagnostic verification.
