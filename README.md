# python_methods_library

A documented reference library of data science methods implemented in Python. Each notebook contains working code with explanations of when and why to use each method, written to be readable as both a reference and a learning resource.

Notebooks are organized by analysis type and are self-contained — dependencies listed at the top, real or synthetic example data included, outputs shown inline.

---

## Structure

```
python_methods_library/
│
├── 01_data_wrangling/
│   ├── README.md
│   ├── pandas_essentials.ipynb         # reshape, merge, groupby, clean
│   └── data_cleaning_patterns.ipynb    # nulls, dtypes, outliers, encoding
│
├── 02_eda/
│   ├── README.md
│   ├── exploratory_analysis.ipynb      # distributions, correlations, summaries
│   └── visualization_patterns.ipynb    # matplotlib, seaborn
│
├── 03_statistical_inference/
│   ├── README.md
│   ├── hypothesis_testing.ipynb        # t-tests, chi-square, ANOVA
│   ├── nonparametric_tests.ipynb       # Mann-Whitney, Kruskal-Wallis
│   └── ab_testing.ipynb                # experiment design, power analysis, p-values
│
├── 04_regression/
│   ├── README.md
│   ├── linear_regression.ipynb
│   ├── logistic_regression.ipynb
│   └── regularized_regression.ipynb    # ridge, lasso, elastic net
│
├── 05_classification/
│   ├── README.md
│   ├── decision_trees_random_forest.ipynb
│   ├── gradient_boosting.ipynb         # XGBoost, LightGBM
│   ├── model_evaluation.ipynb          # AUC, F1, confusion matrix, calibration
│   └── imbalanced_data.ipynb           # SMOTE, class weights
│
├── 06_clustering/
│   ├── README.md
│   ├── kmeans.ipynb
│   ├── hierarchical_clustering.ipynb
│   └── dbscan.ipynb
│
├── 07_dimensionality_reduction/
│   ├── README.md
│   ├── pca.ipynb
│   ├── umap.ipynb
│   └── tsne.ipynb
│
├── 08_model_interpretation/
│   ├── README.md
│   └── shap_explainability.ipynb
│
└── 09_cross_validation/
    ├── README.md
    └── cv_strategies.ipynb             # k-fold, stratified, time series split
```

---

## Key Libraries

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Statistical inference | `scipy.stats`, `statsmodels` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Imbalanced data | `imbalanced-learn` |
| Explainability | `shap` |
| Dimensionality reduction | `umap-learn`, `scikit-learn` |

---

## Notebook Structure

Every notebook follows this pattern:

```
1. Method overview
   - What it does
   - When to use it
   - Key assumptions
   - Industry applications (healthcare / finance / insurance examples)

2. Setup & imports

3. Example data

4. Implementation

5. Output interpretation

6. Common pitfalls
```

---

*Part of a broader portfolio. See also:
[ecological_data_science](https://github.com/samantha-mcgarrigle/ecological_data_science) ·
[r_methods_library](https://github.com/samantha-mcgarrigle/r_methods_library) ·
[databases](https://github.com/samantha-mcgarrigle/databases)*
