# python_methods_library

**Owner:** Samantha McGarrigle  
**Purpose:** Comprehensive Python data science and machine learning methods library for portfolio and interview preparation. Targets data science and data engineering roles in healthcare, finance, insurance, and ecology. Mirrors the r_methods_library where topics overlap, and adds Python-native areas (pipelines, NLP, deep learning, computer vision, graphs, geospatial).

---

## Folder Structure

| # | Folder | Notebooks | Primary Packages | Theme |
|---|---|---|---|---|
| 01 | `01_data_wrangling` | 7 | pandas, numpy | Ecological / Mixed |
| 02 | `02_eda` | 6 | pandas, matplotlib, seaborn, plotly | Ecological / Mixed |
| 03 | `03_statistical_inference` | 6 | scipy, statsmodels, pingouin | Ecological / Mixed |
| 04 | `04_regression` | 6 | scikit-learn, statsmodels | Ecological / Mixed |
| 05 | `05_classification` | 8 | scikit-learn, xgboost, lightgbm | Mixed |
| 06 | `06_clustering` | 7 | scikit-learn, hdbscan | Ecological / Mixed |
| 07 | `07_dimensionality_reduction` | 6 | scikit-learn, umap-learn, openTSNE | Ecological |
| 08 | `08_time_series` | 7 | statsmodels, prophet, sktime | Mixed |
| 09 | `09_bayesian_methods` | 6 | PyMC, ArviZ, scipy | Mixed |
| 10 | `10_statistical_modelling` | 6 | statsmodels, lifelines, pygam | Mixed |
| 11 | `11_data_pipelines` | 5 | scikit-learn, great_expectations | Mixed |
| 12 | `12_nlp_fundamentals` | 6 | spaCy, NLTK, gensim, transformers | Mixed |
| 13 | `13_deep_learning` | 6 | PyTorch, torchvision | Mixed |
| 14 | `14_computer_vision` | 6 | torchvision, OpenCV, timm | Mixed |
| 15 | `15_ab_testing` | 6 | scipy, statsmodels, pymc | Mixed |
| 16 | `16_graphs_networks` | 6 | NetworkX, PyG | Mixed |
| 17 | `17_geospatial` | 6 | geopandas, rasterio, folium | Ecological |
| 10* | `10_model_diagnostics` | 6 | scikit-learn, SHAP, yellowbrick | Mixed |
| 11* | `11_causal_inference` | 6 | DoWhy, econml, statsmodels | Mixed |
| **Total** | | **~118 notebooks** | | |

*Note: `10_model_diagnostics` and `11_causal_inference` were built as companion folders alongside `10_statistical_modelling` and `11_data_pipelines` respectively. All are complete.*

---

## Notebook Standards

- **Format:** Jupyter notebooks (`.ipynb`), Python 3 kernel
- **Structure:** Overview → setup / imports / data simulation → topic sections with code → Common Pitfalls section → footer
- **Footer:** `*python_methods_library - Samantha McGarrigle*`
- **Data:** Simulated inline with numpy/scipy; uses sklearn toy datasets otherwise — no downloads required
- **Pitfalls:** Minimum 5 concrete pitfalls per notebook
- **Style:** scikit-learn API conventions throughout; sklearn Pipeline integration shown wherever applicable

## Folder Standards

- Every folder has a `README.md` covering: purpose, notebook list with topics, key packages, run instructions

### Environment Setup

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn plotly \
            xgboost lightgbm pymc arviz lifelines pygam spacy gensim \
            torch torchvision networkx geopandas rasterio folium umap-learn
```

---

## Notebook Detail by Folder

### 01 data_wrangling
1. `pandas_basics` — DataFrame creation, indexing (.loc/.iloc), dtypes, method chaining
2. `data_cleaning` — Renaming, dropping, duplicates, type coercion, replace/map
3. `reshaping_tidy_data` — melt/stack/unstack, pivot/pivot_table, wide-to-long and back
4. `merging_joining` — merge(), concat(), join types, indicator columns, many-to-many
5. `datetime_handling` — pd.to_datetime(), DatetimeIndex, resampling, time zones, period arithmetic
6. `string_operations` — str accessor, regex with str.extract/contains/replace, text normalisation
7. `missing_data` — isnull(), fillna(), interpolate(), masking strategies, missing mechanism types

### 02 eda
1. `distributions_and_summaries` — describe(), value_counts(), histograms, KDE, skewness/kurtosis
2. `correlation_analysis` — Pearson/Spearman correlation matrices, heatmaps, scatter matrix
3. `matplotlib_fundamentals` — Figure/axes API, subplots, formatting, saving publication-quality figures
4. `seaborn_statistical_plots` — distplot/histplot, boxplot, violinplot, pairplot, lmplot, catplot
5. `plotly_interactive` — px.scatter/line/bar/box, go.Figure, facets, hover data, Dash overview
6. `outlier_detection` — IQR fences, Z-score, Isolation Forest, LOF, visualising outliers

### 03 statistical_inference
1. `t_tests` — scipy.stats t-tests (one-sample, independent, paired), effect sizes, power
2. `anova` — One-way and two-way ANOVA with statsmodels/pingouin, Tukey HSD post-hoc
3. `chi_square_tests` — chi2_contingency, goodness-of-fit, Fisher's exact, Cramér's V
4. `confidence_intervals_bootstrap` — Parametric CIs, bootstrap CIs with scipy, BCa method
5. `nonparametric_tests` — Mann-Whitney, Kruskal-Wallis, Wilcoxon signed-rank, Spearman
6. `effect_sizes` — Cohen's d, eta-squared, odds ratio, partial eta-squared, pingouin

### 04 regression
1. `linear_regression` — OLS with statsmodels and sklearn, coefficient interpretation, diagnostics
2. `multiple_regression` — Multicollinearity, VIF, feature selection, partial regression plots
3. `logistic_regression` — Binary and multiclass logistic, sklearn vs statsmodels, ROC-AUC
4. `poisson_regression` — Count outcomes, exposure offsets, negative binomial comparison
5. `regularisation_ridge_lasso` — Ridge, LASSO, elastic net with sklearn, cross-validated alpha
6. `model_evaluation_regression` — RMSE/MAE/R², residual plots, train/test split, cross-validation

### 05 classification
1. `decision_trees` — DecisionTreeClassifier, max_depth tuning, feature importance, tree plot
2. `random_forest` — RandomForestClassifier, OOB score, feature importance, partial dependence
3. `gradient_boosting` — XGBoost and LightGBM, early stopping, SHAP integration
4. `svm` — SVC with RBF/linear kernels, C/gamma tuning, support vectors, dual problem
5. `naive_bayes` — GaussianNB, BernoulliNB, MultinomialNB, Laplace smoothing
6. `knn` — KNeighborsClassifier, choosing k, distance metrics, curse of dimensionality
7. `model_evaluation_classification` — Confusion matrix, precision/recall/F1, ROC-AUC, calibration
8. `sklearn_pipelines` — Pipeline, ColumnTransformer, GridSearchCV, preventing data leakage

### 06 clustering
1. `kmeans` — KMeans, elbow method, silhouette score, inertia, k-means++ initialisation
2. `hierarchical_clustering` — AgglomerativeClustering, linkage matrices, dendrograms with scipy
3. `dbscan` — DBSCAN, eps/min_samples tuning, core/border/noise points, cluster shapes
4. `gaussian_mixture_models` — GaussianMixture, BIC model selection, covariance types, soft assignments
5. `cluster_validation` — Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI for external validation
6. `dimensionality_reduction_for_clustering` — PCA/UMAP before clustering, distance choice, scaling
7. `cluster_profiling` — Characterising clusters, feature summaries, radar charts, reporting

### 07 dimensionality_reduction
1. `pca` — PCA, explained variance, scree plots, loadings, biplot, whitening
2. `tsne` — t-SNE with openTSNE, perplexity tuning, early exaggeration, reproducibility
3. `umap` — UMAP, n_neighbors/min_dist, supervised UMAP, UMAP vs t-SNE comparison
4. `nmf` — Non-negative Matrix Factorisation, component interpretability, reconstruction error
5. `ordination` — Ecological ordination (PCoA / metric MDS), Bray-Curtis distances
6. `feature_selection` — Filter (ANOVA F, mutual info), wrapper (RFE), embedded (LASSO) methods

### 08 time_series
1. `time_series_fundamentals` — Stationarity, ADF/KPSS tests, differencing, autocorrelation, ACF/PACF
2. `arima` — ARIMA with statsmodels, pmdarima auto_arima, model selection, residual diagnostics
3. `exponential_smoothing` — SES, Holt's linear trend, Holt-Winters with statsmodels
4. `prophet_forecasting` — Meta Prophet in Python, trend changepoints, seasonality, cross-validation
5. `forecasting_evaluation` — Train/test split for time series, MASE/RMSE/MAPE, walk-forward validation
6. `seasonal_decomposition` — STL decomposition, classical decomposition, trend/seasonal extraction
7. `ts_classification_regression` — Time series features (tsfresh), classification, regression on lagged features

### 09 bayesian_methods
1. `bayesian_fundamentals` — Bayes theorem, prior/posterior/likelihood, conjugate distributions
2. `mcmc_sampling` — PyMC NUTS sampler, trace plots, R-hat, ESS, ArviZ diagnostics
3. `bayesian_regression` — Bayesian linear and logistic regression with PyMC, posterior predictive
4. `hierarchical_models` — Partial pooling, multilevel models, prior predictive checks
5. `bayesian_model_comparison` — LOO-CV, WAIC with ArviZ, model stacking
6. `gaussian_processes` — GPy / scikit-learn GP, kernel composition, uncertainty quantification

### 10 statistical_modelling
1. `ols_glm` — OLS and GLMs with statsmodels, formula interface, deviance, link functions
2. `mixed_effects` — Linear and generalised mixed models with statsmodels/pymer4
3. `gams_pygam` — pygam GeneralizedAdditiveModel, spline terms, partial dependence plots
4. `survival_analysis` — Kaplan-Meier, Nelson-Aalen, Cox PH with lifelines
5. `model_selection` — AIC/BIC with statsmodels, forward/backward selection, cross-validation
6. `multicollinearity_assumptions` — VIF, condition number, partial regression plots, remedies

### 10 model_diagnostics
1. `residual_diagnostics` — Residual plots, Q-Q, scale-location, leverage, Cook's distance with sklearn/statsmodels
2. `cross_validation` — KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
3. `bias_variance_tradeoff` — Learning curves, validation curves, bias-variance decomposition
4. `calibration` — Calibration curves, Brier score, Platt scaling, isotonic regression
5. `feature_importance_shap` — SHAP TreeExplainer / LinearExplainer, summary plots, dependence plots
6. `model_comparison_metrics` — Comparing models with cross-val, McNemar test, DeLong test for AUC

### 11 causal_inference
1. `potential_outcomes_dags` — Rubin potential outcomes, DAGs with dagitty/pgmpy, backdoor criterion
2. `matching_methods` — Propensity score matching with DoWhy/causalinference, balance diagnostics
3. `difference_in_differences` — DiD estimator, parallel trends assumption, event study plot
4. `regression_discontinuity` — Sharp RD with statsmodels, bandwidth selection, McCrary density test
5. `instrumental_variables` — IV estimation with linearmodels, relevance and exclusion conditions
6. `sensitivity_analysis` — Rosenbaum bounds, E-values, omitted variable bias sensitivity

### 11 data_pipelines
1. `sklearn_pipelines` — Pipeline, make_pipeline, ColumnTransformer, FunctionTransformer
2. `feature_engineering` — PolynomialFeatures, binning, interaction terms, custom features
3. `missing_data` — SimpleImputer, KNNImputer, IterativeImputer, imputation in pipelines
4. `data_validation` — great_expectations basics, pandera schema validation, type/range checks
5. `custom_transformers` — BaseEstimator/TransformerMixin, fit/transform pattern, stateful transformers

### 12 nlp_fundamentals
1. `text_preprocessing` — Tokenisation, stopword removal, stemming/lemmatisation with spaCy and NLTK
2. `tfidf_classification` — TF-IDF vectorisation, LogisticRegression/NB text classifiers, evaluation
3. `word_embeddings` — Word2Vec/GloVe with gensim, sentence embeddings, similarity queries
4. `topic_modelling` — LDA with gensim, coherence scores, topic visualisation with pyLDAvis
5. `sentiment_ner` — VADER sentiment, spaCy NER, transformer-based NER with HuggingFace
6. `text_similarity` — Cosine similarity, BM25, semantic similarity with sentence-transformers

### 13 deep_learning
1. `pytorch_fundamentals` — Tensors, autograd, custom Dataset/DataLoader, training loop
2. `training_regularisation` — Batch normalisation, dropout, weight decay, learning rate schedulers
3. `cnns` — Convolutional layers, pooling, CNN architecture design, image classification
4. `sequence_models` — RNN, LSTM, GRU for sequence data, text and time series applications
5. `transfer_learning` — Fine-tuning pretrained models (ResNet, BERT), feature extraction
6. `dl_evaluation` — Loss curves, confusion matrix, precision/recall for DL, inference profiling

### 14 computer_vision
1. `image_preprocessing` — OpenCV, torchvision transforms, augmentation (RandomFlip, ColorJitter)
2. `image_classification` — CNN training on CIFAR-10, transfer learning, top-k accuracy
3. `object_detection` — YOLO overview, torchvision Faster R-CNN, bounding box metrics (mAP)
4. `segmentation` — Semantic segmentation, U-Net architecture, IoU / Dice coefficient
5. `pretrained_models` — timm library, EfficientNet/ViT, zero-shot classification with CLIP
6. `cv_evaluation` — mAP, confusion matrix, grad-CAM visualisation, model deployment considerations

### 15 ab_testing
1. `power_analysis` — statsmodels power, t-test/proportion/chi-square sample size, power curves
2. `frequentist_tests` — Two-proportion z-test, t-test for means, one/two-sided tests, p-value interpretation
3. `bayesian_ab` — PyMC Bayesian A/B, posterior probability of superiority, rope decision rule
4. `multiple_comparisons` — Bonferroni, Holm, BH-FDR with statsmodels/scipy, familywise error control
5. `experiment_design` — Randomisation, blocking, factorial designs, CONSORT checklist
6. `effect_sizes_reporting` — Cohen's d, relative uplift, NNT, communicating results to stakeholders

### 16 graphs_networks
1. `networkx_basics` — Graph/DiGraph creation, node/edge attributes, basic properties
2. `centrality_community` — Degree/betweenness/PageRank centrality, Louvain/Leiden community detection
3. `shortest_paths_flow` — BFS/Dijkstra/A*, max flow with NetworkX, minimum spanning tree
4. `network_statistics` — Clustering coefficient, degree distribution, small-world and scale-free properties
5. `bipartite_temporal` — Bipartite graph projections, temporal networks, dynamic community detection
6. `graph_ml` — Node2Vec embeddings, GNN overview with PyTorch Geometric, link prediction

### 17 geospatial
1. `vector_data` — GeoDataFrame, shapely geometries, CRS, spatial joins, geopandas operations
2. `raster_data` — rasterio, numpy raster operations, zonal statistics, raster/vector interaction
3. `spatial_analysis` — Spatial autocorrelation (Moran's I), spatial weights with libpysal
4. `spatial_interpolation` — IDW, kriging with pykrige, point process models
5. `mapping_visualisation` — Static maps with geopandas/matplotlib, interactive maps with folium/kepler.gl
6. `geospatial_ml` — Spatial feature engineering, geographically weighted regression, spatial CV

---

## Build Status

| Folder | Status |
|---|---|
| 01_data_wrangling | ✅ Complete |
| 02_eda | ✅ Complete |
| 03_statistical_inference | ✅ Complete |
| 04_regression | ✅ Complete |
| 05_classification | ✅ Complete |
| 06_clustering | ✅ Complete |
| 07_dimensionality_reduction | ✅ Complete |
| 08_time_series | ✅ Complete |
| 09_bayesian_methods | ✅ Complete |
| 10_statistical_modelling | ✅ Complete |
| 10_model_diagnostics | ✅ Complete |
| 11_causal_inference | ✅ Complete |
| 11_data_pipelines | ✅ Complete |
| 12_nlp_fundamentals | ✅ Complete |
| 13_deep_learning | ✅ Complete |
| 14_computer_vision | ✅ Complete |
| 15_ab_testing | ✅ Complete |
| 16_graphs_networks | ✅ Complete |
| 17_geospatial | ✅ Complete |

---
*python_methods_library - Samantha McGarrigle*
