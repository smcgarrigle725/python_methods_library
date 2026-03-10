# 01 — Data Wrangling

Data wrangling is the process of loading, inspecting, reshaping, and cleaning raw data into a form suitable for analysis. In practice it consumes the majority of a data scientist's time; doing it carefully and reproducibly is as important as any downstream modelling step.

This folder covers the core pandas toolkit for working with tabular data in Python.

---

## Notebooks

### `pandas_basics.ipynb`
Core pandas workflow: Series and DataFrame construction, `.loc` vs `.iloc` indexing, inspecting data with `.info()` and `.describe()`, column selection with `select_dtypes()`, boolean filtering, method-chaining with `.assign()`, `groupby().agg()` with named aggregation, `.transform()` for group-level summaries, and `sort_values()`. Pitfalls: chained indexing, modifying during iteration, forgetting `reset_index()`, misusing `apply()`.

### `data_cleaning.ipynb`
Detecting and correcting structural problems: wrong dtypes, inconsistent categories, duplicate rows and columns, out-of-range values, and non-standard column names. Covers the full cleaning checklist with before/after comparisons. Pitfalls: silently dropping out-of-range rows, unstandardised category strings, sentinel missing values (-999), parsing dates without `errors='coerce'`.

### `reshaping_tidy_data.ipynb`
Tidy data principles and the pandas tools that implement them: `melt()` for wide-to-long, `pivot()` and `pivot_table()` for long-to-wide, `explode()` for list-valued cells, and splitting compound column names with `str.rsplit()`. Pitfalls: duplicate index-column combinations in `pivot()`, losing metadata columns in `melt()`, splitting on the wrong separator.

### `merging_joining.ipynb`
Combining DataFrames: `pd.merge()` for SQL-style column-based joins (inner, left, right, outer), many-to-one and many-to-many cardinality, `validate=` for cardinality checking, `indicator=True` for coverage auditing, and `pd.concat()` for stacking rows or columns. Pitfalls: undetected row count explosions, mismatched key dtypes, forgetting `ignore_index=True`, using `concat(axis=1)` instead of `merge()`.

### `datetime_handling.ipynb`
Temporal data: parsing with `pd.to_datetime()`, the `.dt` accessor for vectorised datetime operations, seasonal grouping with `.map()`, datetime arithmetic and `DateOffset`, `resample()` for time-series aggregation, rolling windows, and time zone localisation/conversion. Pitfalls: leaving datetimes as object dtype, mixing tz-aware and tz-naive, calling `resample()` without a DatetimeIndex, using deprecated frequency aliases.

### `string_operations.ipynb`
The `.str` accessor for vectorised string processing: case normalisation, whitespace stripping, regex-based extraction with named capture groups, `str.contains()` for boolean flags, `str.split().explode()` for tidy free-text, and standardising messy categorical labels. Pitfalls: calling `.str` on non-string dtypes, forgetting `na=False`, greedy regex over-matching, confusing `regex=True` and `regex=False` in `str.replace()`.

### `missing_data.ipynb`
Missing data mechanisms (MCAR, MAR, MNAR) and their implications. Auditing missingness patterns with heatmaps and group comparisons. Imputation strategies: complete-case analysis, mean/median imputation, `IterativeImputer` (MICE-like), and `miceforest`. Creating missingness indicator columns before imputing. Pitfalls: `dropna()` without investigating mechanism, imputing before the train/test split, mean imputation for inference, not preserving the "was missing" signal.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, sklearn
```

## Data
Simulated ecological water-quality dataset: site-level measurements of elevation, nitrate, phosphorus, pH, and species richness across four catchments with control and restored treatments.
