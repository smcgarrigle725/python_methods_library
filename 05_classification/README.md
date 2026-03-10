# 05 — Classification

Classification assigns observations to discrete categories. This folder covers the primary sklearn classification algorithms, imbalanced class handling, multiclass strategies, and a comprehensive treatment of evaluation metrics — because choosing the right metric matters as much as choosing the right model.

---

## Notebooks

### `decision_trees.ipynb`
`DecisionTreeClassifier`: Gini vs entropy splitting, `max_depth` and `min_samples_leaf` regularisation, cost-complexity pruning path, `plot_tree()` and `export_text()` visualisation, Gini feature importance vs permutation importance. Pitfalls: unpruned trees, Gini importance bias, unstratified splits, single-tree feature selection, accuracy-only evaluation for imbalanced classes.

### `random_forest.ipynb`
`RandomForestClassifier`: bootstrap aggregation, `max_features="sqrt"`, OOB score as free cross-validation estimate, OOB error vs n_estimators convergence plot, Gini vs permutation importance comparison, ROC and calibration plots. Pitfalls: Gini over permutation importance, OOB as substitute for held-out test, uncalibrated probabilities, n_estimators too low, no class imbalance handling.

### `gradient_boosting.ipynb`
`HistGradientBoostingClassifier` as the sklearn-native fast implementation. Learning rate vs n_estimators tradeoff. Early stopping with `n_iter_no_change`. Permutation importance on test set. Comparison to XGBoost and LightGBM (conceptual). Pitfalls: large learning rate with few trees, no early stopping, max_depth too large for boosting, skipping a simpler baseline.

### `svm_knn.ipynb`
`SVC` with RBF kernel in a `Pipeline` with `StandardScaler`. Hyperparameter tuning of C and gamma with `GridSearchCV`. `KNeighborsClassifier` with CV-selected k. ROC comparison between SVM and k-NN. Pitfalls: SVM without scaling, k=1 k-NN, `probability=True` cost, k-NN at large n, test-set hyperparameter search.

### `naive_bayes.ipynb`
`GaussianNB` for continuous features, `BernoulliNB` for presence/absence data. Calibration plot showing overconfidence of raw NB probabilities. Isotonic calibration with `CalibratedClassifierCV`. Laplace smoothing for zero-frequency cells. Pitfalls: GaussianNB on skewed count data, trusting raw NB probabilities, assuming failure from correlated features, alpha=0, skipping NB as baseline.

### `imbalanced_classes.ipynb`
The accuracy paradox on imbalanced data. `class_weight="balanced"` as the zero-cost first step. Precision-Recall curve and threshold tuning for F1 maximisation. SMOTE oversampling with `imbalanced-learn`. Average precision (PR-AUC) vs ROC-AUC for imbalanced outcomes. Pitfalls: accuracy as primary metric, SMOTE before the split, ROC-AUC alone, fixed 0.5 threshold, over-relying on SMOTE before trying class_weight.

### `multiclass_classification.ipynb`
Three-class classification with per-class precision, recall, and F1. Macro vs weighted averaging. Raw and row-normalised confusion matrices. One-vs-Rest multiclass ROC curves with per-class AUC. Logistic regression OvR vs multinomial. Pitfalls: macro F1 for very unequal class sizes, reading only the confusion matrix diagonal, raw counts without row-normalisation, OvR vs OvO default confusion, aggregated metrics hiding per-class failures.

### `model_evaluation_classification.ipynb`
Systematic comparison of four classifiers (Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes) on ROC-AUC, PR-AUC, Brier score, and F1. ROC and Precision-Recall curves side by side. Calibration plot comparison. Cross-validated model comparison with `StratifiedKFold`. Pitfalls: test-set-guided model selection, ROC-AUC alone for imbalanced data, no Brier score when probabilities matter, cross-validated F1 variance, single-metric model selection.

---

## Dependencies
```
pandas, numpy, matplotlib, scipy, sklearn, imbalanced-learn (optional, for SMOTE)
```

## Data
Simulated ecological dataset: binary species presence/absence and three-class water quality outcomes modelled from elevation, water chemistry, and treatment predictors.
