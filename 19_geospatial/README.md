# 19 — Geospatial Analysis

End-to-end geospatial analysis in Python: vector data manipulation, raster processing, spatial statistics, interpolation, cartographic visualisation, and spatially-aware machine learning. All examples use a simulated Scottish riparian monitoring network in British National Grid (EPSG:27700).

---

## Notebooks

### `vector_data.ipynb`
GeoPandas GeoDataFrame construction from Point, LineString, Polygon geometries. CRS setting (EPSG:27700). Multi-layer map with catchment polygons, river lines, and richness-coloured site markers. Buffer (500m), spatial join (`sjoin`, predicate="within"), dissolve, and overlay intersection. Pairwise distance matrix, nearest-neighbour lookup. CRS reprojection to WGS84 (EPSG:4326). Pitfalls: degrees vs metres, missing CRS, sjoin predicate choice, dissolve dropping columns, mixed CRS in operations.

### `raster_data.ipynb`
Simulated 200×200 DTM at 25m resolution using affine transform. Rasterio GeoTIFF write/read with metadata. Slope derivation from `np.gradient` with cell size. Simulated NDVI surface. Multi-panel raster visualisation. Zonal statistics via `geometry_mask`. Block-averaging resample to 50m. Multi-band stack (elevation, slope, NDVI). Pitfalls: NoData handling, CRS mismatch for zonal stats, nearest-neighbour vs bilinear resample, gradient cell size, large raster memory.

### `spatial_analysis.ipynb`
Gaussian process simulation of spatially autocorrelated richness. KDE surface via `scipy.stats.gaussian_kde`. IDW prediction surface. Moran's I with permutation test (999 permutations). Semivariogram: empirical bins + exponential model fit. LISA (Local Moran's I): HH/LL/HL/LH cluster classification and map. Pitfalls: OLS ignoring spatial autocorrelation, variogram lag parameters, global vs local Moran's I, KDE bandwidth choice, variogram range interpretation.

### `spatial_interpolation.ipynb`
IDW (power=2), nearest-neighbour (`NearestNDInterpolator`), thin-plate spline (`RBFInterpolator`). Side-by-side RMSE comparison vs known true field. Empirical semivariogram with exponential variogram fitting. Full ordinary kriging implementation: normalised adjacency kriging matrix, batched prediction. Kriging prediction surface + uncertainty (std) surface. LOO cross-validation: ME, RMSE, MAE. Pitfalls: IDW power tuning, valid variogram models, reporting prediction without uncertainty, no cross-validation, extrapolation beyond sample extent.

### `mapping_visualisation.ipynb`
Publication map: catchment polygons, richness-coloured sites, scale bar, north arrow, legend. Sequential / diverging / qualitative choropleth comparison. Folium interactive HTML map with circle markers and popups (falls back gracefully if not installed). Small-multiples: three-panel catchment maps with shared colourbar and `vmin`/`vmax`. Pitfalls: rainbow colour scales, class boundary selection, CRS for area vs mapping, shared scale for small multiples, missing map furniture.

### `geospatial_ml.ipynb`
Multi-scale neighbourhood feature engineering (1km / 3km windows). Random k-fold vs spatial block CV comparison — optimism gap quantified. Model comparison table: Ridge, RF (base), RF (spatial features), GBM — both CV types. RF richness prediction surface. Feature importance plot. Pitfalls: random CV for spatial data, raw coordinates as features, neighbourhood response leakage, covariate extrapolation, block size vs variogram range.

---

## Dependencies
```
geopandas, shapely, rasterio, numpy, pandas, matplotlib, scipy, sklearn
folium (optional, interactive maps: pip install folium)
contextily (optional, basemap tiles: pip install contextily)
```
