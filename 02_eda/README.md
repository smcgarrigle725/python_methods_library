# 02 — Exploratory Data Analysis

Exploratory data analysis is the process of getting to know a dataset before modelling: understanding distributions, detecting anomalies, quantifying relationships, and identifying the most informative visualisations. EDA shapes every modelling decision that follows.

This folder covers the full Python visualisation and exploration toolkit.

---

## Notebooks

### `distributions_and_summaries.ipynb`
Extended summary statistics beyond `.describe()`: skewness, kurtosis, IQR. Histograms with KDE overlay and mean/median reference lines. Notched box plots by group. Q-Q plots for normality assessment. Shapiro-Wilk test with caveats for large n. Pitfalls: relying on `.describe()` alone, reporting means for skewed data, default histogram bin counts, acting on Shapiro-Wilk at large n, ignoring physical bounds.

### `correlation_analysis.ipynb`
Pearson correlation (linear, sensitive to outliers), Spearman rank correlation (monotonic, robust), and Kendall's tau. Correlation matrices with `seaborn.heatmap()`. Scatter plot matrices. Distinguishing correlation from causation. Pitfalls: Pearson on non-linear relationships, correlation on non-monotonic associations, ignoring outlier influence, treating correlation as causation.

### `matplotlib_fundamentals.ipynb`
The Figure/Axes object model. Subplots with `plt.subplots()`. Customising tick labels, axis limits, titles, and colour schemes. Saving publication-quality figures. Combining multiple plot types on one axes. Pitfalls: pyplot state machine vs object API confusion, not setting figure size before plotting, overwriting axes objects in loops.

### `seaborn_statistical_plots.ipynb`
Seaborn's high-level statistical plotting API: `histplot`, `kdeplot`, `boxplot`, `violinplot`, `stripplot`, `pairplot`, `heatmap`, `FacetGrid`. Integrating with pandas DataFrames directly. Choosing between overlapping plot types. Pitfalls: using seaborn's default theme without checking for colourblind accessibility, overplotting with large n, misreading confidence intervals on `lineplot`.

### `plotly_interactive.ipynb`
Interactive browser-based charts with `plotly.express` and `plotly.graph_objects`: scatter, line, bar, histogram, box, choropleth. Hover tooltips, zoom, and filtering. Exporting to HTML for sharing. Pitfalls: interactive charts obscuring overplotting that static plots reveal, exporting to static PNG losing interactivity, large datasets slowing browser rendering.

### `outlier_detection.ipynb`
Univariate methods: IQR fence, Z-score, MAD-based detection. Multivariate methods: Isolation Forest, Local Outlier Factor, Mahalanobis distance. Visualising flagged observations in scatter space. Deciding between investigation, transformation, and removal. Pitfalls: removing outliers without investigation, Z-score on skewed data, treating multivariate outliers as univariate, conflating outliers with errors.

---

## Dependencies
```
pandas, numpy, matplotlib, seaborn, plotly, scipy, sklearn
```

## Data
Same simulated ecological water-quality dataset as `01_data_wrangling`.
