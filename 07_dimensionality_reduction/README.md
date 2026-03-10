# 07 — Dimensionality Reduction

High-dimensional data is hard to visualise, often contains redundant features, and can cause models to overfit. Dimensionality reduction addresses all three problems — either by transforming features into a lower-dimensional representation, or by selecting a subset of the original features.

This folder covers linear, non-linear, and ecology-specific methods, with consistent attention to what each method preserves, what it distorts, and when to use each.

---

## Notebooks

### `pca.ipynb`
Principal Component Analysis: scree plot and cumulative variance, Kaiser criterion, loadings heatmap for feature interpretation, biplot of scores and loadings, reconstruction error vs components retained. Choosing n_components by combining the 80% threshold, Kaiser criterion, and elbow. Pitfalls: no standardisation, interpreting scores without loadings, 80% threshold as absolute rule, fitting PCA on full data before split, expecting PCA to reveal non-linear structure.

### `tsne.ipynb`
t-SNE for 2D/3D visualisation of non-linear cluster structure. Perplexity sensitivity analysis across four values. PCA pre-reduction before t-SNE for high-dimensional data. Colouring embeddings by continuous features to understand what drives separation. Stability check across multiple random states. Pitfalls: interpreting inter-cluster distances, single-run instability, using t-SNE coordinates as model features, `init="random"` default, applying to raw high-d data without pre-reduction.

### `umap.ipynb`
UMAP embedding with `n_neighbors` and `min_dist` sensitivity analysis. `transform()` for applying a fitted UMAP to new data — the key advantage over t-SNE. UMAP as a feature extractor for downstream classification. Pitfalls: treating UMAP distances as magnitude-meaningful, fitting on full data before split, extreme n_neighbors values, no random_state, substituting UMAP for interpretable feature selection.

### `nmf.ipynb`
Non-Negative Matrix Factorisation for parts-based decomposition of species abundance matrices. Component interpretation as ecological assemblage types. Reconstruction error elbow for n_components selection. Comparison with PCA on count data. Row-normalisation for relative composition. Pitfalls: negative-valued input after scaling, random initialisation, components as independent/exclusive categories, no reconstruction error evaluation, not normalising for total abundance.

### `feature_selection.ipynb`
Three families of feature selection: filter methods (ANOVA F, mutual information), wrapper methods (RFECV with cross-validated feature count curve), and embedded methods (LASSO, permutation importance). Pipeline-based selection inside cross-validation vs the leaky alternative. Pitfalls: selection outside CV pipeline, filter methods for correlated features, LASSO zeros as confirmed irrelevance, re-selection after data changes, comparing methods without a held-out test.

### `ordination.ipynb`
Ordination methods for ecological community data. Bray-Curtis dissimilarity as the appropriate distance for abundance matrices. PCoA (metric MDS) and NMDS (non-metric MDS) on distance matrices. Environmental vector overlays on NMDS plots. Shepard diagram for stress validation. Pitfalls: Euclidean distance on community data, unreported NMDS stress, single random start, interpreting NMDS axis direction, raw counts without Hellinger transformation.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, sklearn
umap-learn (optional, for UMAP)
hdbscan (optional)
```

## Data
Simulated ecological datasets: correlated site-level water chemistry (PCA, t-SNE, UMAP, feature selection), species-by-site abundance matrices with known assemblage structure (NMF, ordination).
