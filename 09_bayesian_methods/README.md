# 09 — Bayesian Methods

Bayesian inference treats parameters as random variables with probability distributions, updates prior beliefs with data, and produces full posterior distributions over unknowns rather than point estimates. This folder progresses from first principles through MCMC, hierarchical models, and Gaussian processes — with consistent emphasis on what Bayesian methods offer over frequentist alternatives, and where they require more care.

---

## Notebooks

### `bayesian_fundamentals.ipynb`
Bayes' theorem applied to the Beta-Binomial conjugate model for estimating restoration success rate. Prior, likelihood, and posterior visualised together. Three prior choices compared. Prior sensitivity analysis across five priors. Sequential updating as observations accumulate. Normal-Normal conjugate for mean estimation. Pitfalls: credible vs confidence interval confusion, uniform priors as objective, no sensitivity analysis, small-n prior dominance, conjugacy availability.

### `mcmc_sampling.ipynb`
Metropolis-Hastings from scratch: log-posterior, accept/reject step, acceptance rate tuning. Trace plots, posterior histogram, and chain ACF. PyMC with NUTS for the same model. R-hat and effective sample size computed manually and via ArviZ. Pitfalls: no burn-in discard, single chain, R-hat < 1.1 threshold too lenient, raw sample count as ESS, divergences treated as minor warnings.

### `bayesian_regression.ipynb`
Bayesian linear regression: flat prior equivalence to OLS, weakly informative priors on standardised predictors, PyMC implementation with NUTS. Analytical Normal-inverse-gamma posterior. Posterior predictive intervals for new observations. Shrinkage priors demonstrated through normal prior width vs coefficient magnitude. Pitfalls: uninformative priors everywhere, unstandardised predictors, reporting only posterior means, no posterior predictive checks, coefficient CIs vs prediction intervals confusion.

### `hierarchical_models.ipynb`
Partial pooling for grouped ecological data (8 catchments, variable n). No-pooling vs complete-pooling vs partial-pooling comparison with known true means. `statsmodels.mixedlm` for random intercepts. Shrinkage visualisation: small groups pulled further toward grand mean. Analytical shrinkage factor per group. ICC computation. Pitfalls: complete pooling ignoring groups, no-pooling with small groups, random effects normality assumption, between-group SD not reported, fixed vs random effects misclassification.

### `bayesian_model_comparison.ipynb`
PSIS-LOO-CV and WAIC via PyMC + ArviZ as the recommended comparison method. AIC and BIC compared across four OLS models including an overfit polynomial. Posterior predictive checks: replicated histogram overlay and distribution of replicated means. Savage-Dickey Bayes Factor for a point null hypothesis. Jeffreys evidence scale. Pitfalls: AIC across different likelihoods/subsets, lowest AIC without absolute fit check, Pareto k warnings, prior-sensitive Bayes Factors, skipping PPCs.

### `gaussian_processes.ipynb`
GP regression with `sklearn.GaussianProcessRegressor`: RBF + WhiteKernel, kernel hyperparameter optimisation via log marginal likelihood. Uncertainty bands widening in data-sparse regions. Kernel comparison: RBF, Matérn 3/2, Matérn 5/2, Rational Quadratic. 2D spatial interpolation with anisotropic RBF. Posterior function samples. Pitfalls: no hyperparameter optimisation, RBF for non-smooth processes, GP bands as frequentist CIs, scalability to large n, extrapolation beyond training range.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels, sklearn
pymc, arviz (optional but recommended for full MCMC workflow)
```

## Data
Simulated ecological datasets: binary restoration success (Beta-Binomial), species richness at sites (regression, hierarchical), nitrate along elevation gradients (GP regression), and 2D spatial nitrate fields (GP interpolation).
