# 08 — Time Series

Time series data carries temporal structure that standard statistical and ML methods ignore at their peril. This folder covers the full workflow from understanding a series to modelling, forecasting, and detecting structural change — with consistent emphasis on the temporal constraints that govern valid evaluation.

---

## Notebooks

### `time_series_fundamentals.ipynb`
Core concepts: trend, seasonality, stationarity, autocorrelation. Rolling mean and SD. Classical additive decomposition. ADF and KPSS stationarity tests used together. ACF and PACF interpretation guide. First and seasonal differencing. Pitfalls: skipping stationarity checks, ADF alone, over-differencing, ignoring seasonal ACF, reading rolling mean as a forecast.

### `arima.ipynb`
SARIMA(p,d,q)(P,D,Q,s) with `statsmodels.SARIMAX`. ACF/PACF on differenced series for order selection. AIC/BIC for candidate comparison. Full residual diagnostics: `plot_diagnostics()` and Ljung-Box test. Forecast with 95% CI. RMSE, MAE, MAPE evaluation. Pitfalls: no residual checking, AIC-only order selection, ARIMA vs SARIMA confusion, MAPE near zero, forecasting beyond the horizon.

### `exponential_smoothing.ipynb`
Simple Exponential Smoothing (level only), Holt's Linear (level + trend), and Holt-Winters (level + trend + seasonality) in additive and multiplicative forms. Bootstrap prediction intervals. Model comparison table. Pitfalls: multiplicative with zeros, additive vs multiplicative choice by default, boundary-hitting smoothing parameters, no naive benchmark, Holt-Winters for changing seasonal patterns.

### `forecasting_evaluation.ipynb`
Walk-forward expanding window CV with `TimeSeriesSplit`. MASE (scale-independent, naive-benchmarked). Seasonal naive as the minimum baseline. Forecast error profile across horizons 1–24 steps. Residual ACF and Ljung-Box after fitting. Pitfalls: standard k-fold CV, RMSE without benchmark, single-horizon evaluation, autocorrelated residuals ignored, in-sample AIC as selection criterion.

### `trend_seasonality_decomposition.ipynb`
Classical additive vs multiplicative decomposition. STL (Seasonal-Trend by LOESS) with `robust=True` for outlier resistance. Variance partitioned across trend, seasonal, and residual components. Seasonal subseries plot for within-month trend. STL + ETS combined forecasting. Pitfalls: additive when amplitude grows, non-robust STL with outliers, residuals not checked, stale seasonal component in forecasts, wrong period specification.

### `changepoint_detection.ipynb`
CUSUM chart for persistent mean shift detection. `ruptures` Pelt algorithm for multiple structural break detection. Penalty sensitivity analysis. Binary segmentation. Mann-Kendall trend test and Sen's slope. Pitfalls: visual-only changepoint identification, penalty without sensitivity analysis, seasonal series without deseasonalising, trend vs changepoint confusion, breakpoints not validated against external events.

### `ml_for_time_series.ipynb`
Supervised forecasting with lag features and rolling statistics. Cyclical month encoding (sine/cosine). Random Forest and Gradient Boosting regressors. Strict temporal train/test split. `TimeSeriesSplit` walk-forward CV. Permutation importance for lag selection. Pitfalls: random k-fold CV, shift-less lag features, integer month encoding, no lag importance analysis, point forecasts without uncertainty quantification.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, statsmodels, sklearn
ruptures (optional, for changepoint detection)
pymannkendall (optional, for Mann-Kendall test)
```

## Data
Simulated monthly nitrate concentration series with trend, annual seasonality, noise, structural breaks, and occasional outlier events.
