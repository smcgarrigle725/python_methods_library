# 06 — Clustering

Clustering finds structure in unlabelled data by grouping observations that are similar to each other. Unlike classification, there are no ground truth labels — the challenge is both finding meaningful groups and deciding whether those groups are real.

This folder covers the major clustering algorithms, evaluation approaches, and the interpretive step that gives clusters scientific meaning.

---

## Notebooks

### `kmeans.ipynb`
K-means partitioning via WCSS minimisation. Standardisation requirement. Elbow plot and silhouette score for k selection. Silhouette plot for per-cluster quality assessment. PCA visualisation of cluster structure. Multiple random initialisations with `n_init`. Pitfalls: no standardisation, single initialisation, elbow-only k selection, label instability across runs, non-spherical clusters.

### `hierarchical_clustering.ipynb`
Agglomerative clustering with `scipy.cluster.hierarchy`. Ward, complete, average, and single linkage methods. Dendrogram construction and cutting with `fcluster`. Linkage method comparison in PCA space. `sklearn.AgglomerativeClustering` with connectivity constraints. Pitfalls: single linkage chaining, no standardisation, visual dendrogram reading without silhouette confirmation, scalability for large n, fcluster 1-based vs sklearn 0-based labels.

### `dbscan.ipynb`
Density-based clustering with explicit noise detection. k-distance plot for eps selection. eps sensitivity analysis. HDBSCAN as the practical upgrade (no eps, handles varying density). Pitfalls: eps without k-distance plot, silhouette on noise-inclusive labels, DBSCAN for density-varying clusters, noise points as automatic anomalies.

### `gaussian_mixture_models.ipynb`
Soft cluster assignments via EM-fitted Gaussian components. BIC and AIC for component count selection. Covariance type comparison: `full`, `tied`, `diag`, `spherical`. Visualising soft assignments by opacity. Pitfalls: full covariance for small n, convergence not checked, BIC/AIC without domain validation, GMM as density estimator vs clustering tool, no standardisation.

### `cluster_evaluation.ipynb`
Internal metrics: silhouette score, Davies-Bouldin index, Calinski-Harabasz score across k. Algorithm comparison table with all three internal metrics plus ARI against true labels. External metrics: ARI, NMI, homogeneity, completeness, V-measure. Bootstrap cluster stability assessment. Pitfalls: silhouette-only k selection, ARI from mismatched reference labels, cross-algorithm silhouette comparisons, no stability assessment, best-of-many-k reporting.

### `cluster_validation.ipynb`
Testing whether cluster structure genuinely exists before fitting: Hopkins statistic for spatial randomness, gap statistic comparing inertia to a null reference distribution. Distinguishing real cluster structure from imposed partitioning of uniform data. Pitfalls: clustering always-uniform data and reporting results, Hopkins statistic dependence on sample size, gap statistic Monte Carlo variance.

### `cluster_profiling_interpretation.ipynb`
Translating cluster labels into scientific meaning: per-cluster summaries, Kruskal-Wallis tests for discriminating features, box plots and standardised centroid heatmaps. Naming clusters from data rather than prior expectations. Validating against external variables not used in clustering. Pitfalls: means without spread, naming before examining all features, circular validation, imbalanced cluster sizes, label instability across re-runs.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, sklearn, statsmodels
hdbscan (optional, for HDBSCAN)
```

## Data
Simulated ecological site data: elevation, pH, nitrate, phosphorus, and species richness across four catchment types, with known ground-truth groupings for external metric benchmarking.
