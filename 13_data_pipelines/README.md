# 13 — Data Pipelines

Production-ready data handling: from raw ingestion through validation, imputation, feature engineering, and deployment-safe sklearn pipelines. The central theme is preventing leakage — ensuring that no information from the validation or test set contaminates any step of the training process.

---

## Notebooks

### `sklearn_pipelines.ipynb`
`Pipeline` and `ColumnTransformer` for mixed numeric/categorical data. Separate impute→scale and impute→encode branches merged by ColumnTransformer. Leakage demonstration: scaler fitted before vs inside CV. Swapping classifiers without touching preprocessing. `GridSearchCV` with double-underscore nested parameter access. `joblib` serialisation of the complete pipeline. Pitfalls: preprocessing outside pipeline, wrong step names, remainder dropped silently, sparse OHE output, saving model only.

### `feature_engineering.ipynb`
Log and Box-Cox transforms for right-skewed features. Domain ratio features: N:P ratio, nutrient load. Cyclic sine/cosine encoding for month. Target encoding with `TargetEncoder` fitted on training fold only. Full pipeline comparison: baseline vs domain-engineered features. Pitfalls: target encoding leakage, integer month as linear, ratio by zero, log of zero, engineering with test-set statistics.

### `missing_data.ipynb`
MCAR/MAR/MNAR mechanisms with visual diagnosis. Missingness matrix and KS test for MAR pattern. Mean, median, KNN, and iterative (Bayesian Ridge) imputation compared against true values. Missingness indicator features: adding binary flags for informative missingness. Multiple imputation via `IterativeImputer` with Rubin's rules pooling. Pitfalls: ignoring mechanism, imputing the target, fitting imputer on test data, single imputation with CI reporting, no missingness indicators.

### `data_validation.ipynb`
Manual validation: dtype checks, range bounds, allowed categories, duplicates, nulls. Distribution monitoring with KS test between historical and new batch. `pandera` schema validation with `lazy=True` for full error collection. Custom validation report function with PASS/FAIL per rule. Pitfalls: validate once at project start, passing as correctness guarantee, silent dropping, rules from same data, no distribution shift monitoring.

### `custom_transformers.ipynb`
`FunctionTransformer` for stateless operations (log1p, ratio features). `BaseEstimator + TransformerMixin` for stateful transforms: `Winsorizer` class with percentiles fitted in `fit()`. `DomainRatioFeatures` class with tunable parameters. Integration into full `Pipeline` with CV. `GridSearchCV` tuning custom transformer hyperparameters via `__` notation. Pitfalls: statistics in `transform()` not `fit()`, no `check_is_fitted`, in-place modification, missing `return self`, parameter name mismatch.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, sklearn
pandera (optional: pip install pandera)
joblib (usually installed with sklearn)
```

## Data
Simulated riparian monitoring dataset with elevation, nitrate, phosphorus, flow rate, catchment, and species richness — including deliberately injected data quality problems (negatives, duplicates, wrong dtype, distribution shift) for validation demonstrations.
