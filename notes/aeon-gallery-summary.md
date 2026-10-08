# aeon Toolkit — Examples Gallery Summary

Source: https://www.aeon-toolkit.org/en/stable/examples.html (fetched 2026-09-22)
Gallery base URL: `https://www.aeon-toolkit.org/en/stable/examples/`

Notebook examples organized by module. Each is a tutorial/overview of a task, algorithm family, or specific estimator. ~49 examples in 12 sections.

## 1. Classification (12 examples)
Task overviews by algorithm family:
- **classification.html** — TSC overview (data formats, fit/predict, tuning)
- **convolution_based.html** — Rocket-family, convolution kernels
- **deep_learning.html** — DL classifiers (InceptionTime, etc.)
- **dictionary_based.html** — BOSS/TDE/WEASEL symbolic approaches
- **distance_based.html** — Proximity/elastic (DTW) classifiers
- **feature_based.html** — catch22/tsfresh features + pipeline
- **hybrid.html** — HIVE-COTE style composable ensembles
- **interval_based.html** — TimeSeriesForest, RSTSF, interval forests
- **shapelet_based.html** — shapelet discovery/transform
- **early_classification.html** — classifying series before full observation
- Specific estimators: **interval_based/drcif.html** (DrCIF), **rotation_forest.html** (RotationForest)

## 2. Regression (1)
- **regression/regression.html** — time series extrinsic regression (TSER) overview

## 3. Clustering (2)
- **clustering/clustering.html** — TSCL overview (k-means/medoids, elastic distances)
- **partitional_clustering.html** — partition-based clustering algorithms

## 4. Transformation (8)
- **transformations/transformations.html** — overview of the module
- **preprocessing.html** — scaling, interpolation, windowing
- **tsfresh.html** — tsfresh feature-extraction wrapper
- **catch22.html** — canonical time-series characteristics (22 features)
- **rocket.html** — ROCKET (random convolutional kernels)
- **minirocket.html** — MiniRocket (faster variant)
- **sast.html** — SAST (scale adaptive shapelet sampling)
- **signature_method.html** — path signatures
- **channel_selection.html** — channel selection in pipelines

## 5. Segmentation (3)
- **segmentation/segmentation.html** — intro/module overview
- **segmentation_with_clasp.html** — ClaSP changepoint detection
- **hidalgo_segmentation.html** — Hidalgo probabilistic segmentation

## 6. Distances (2)
- **distances/distances.html** — distance functions (DTW, erp, edit-ndr...), numba-accelerated, windows/grids
- **sklearn_distances.html** — using aeon distances with scikit-learn estimators

## 7. Similarity Search (4)
- **similarity_search/similarity_search.html** — module intro (pan matrix profile)
- **distance_profiles.html** — deep dive on distance profiles
- **code_speed.html** — benchmarking module speedups
- **simhash_index.html** — SimHash + LSH index

## 8. Forecasting (1)
- **forecasting/forecasting.html** — forecasting with aeon (experimental module)

## 9. Data Formatting & Loading (5)
- **datasets/datasets.html** — data storage/loading overview
- **data_loading.html** — in-memory collection formats (numpy 2D/3D, df-list, nested pd)
- **provided_data.html** — built-in example datasets
- **load_data_from_web.html** — downloading UEA/Zenodo benchmarks
- **data_unequal.html** — unequal-length series / missing values

## 10. Benchmarking (4)
- **benchmarking/benchmarking.html** — comparing estimator performance
- **published_results.html** — retrieving/comparing vs published results
- **reference_results.html** — estimator reference results
- **regression.html** — benchmarking extrinsic regression models

## 11. Base (2)
- **base/base_classes.html** — base class structure (BaseCollectionEstimator etc.)
- **series_estimator.html** — BaseSeriesEstimator (per-series API)

## 12. Visualisation (5)
- **visualisation/plotting_series.html** — plotting time series
- **plotting_results.html** — results plots (critical difference diagrams etc.)
- **plotting_for_learning_tasks.html** — task-specific plots
- **plotting_distances.html** — distance-path plots
- **plotting_estimators.html** — estimator inspection plots

## Takeaways
- Gallery is task/family-oriented overviews first, then a few estimator-specific deep dives (DrCIF, RotationForest, ClaSP, Hidalgo, MiniRocket, SAST, SimHash).
- Uneven coverage: classification rich (12), forecasting/regression thin (1 each).
- Experimental modules flagged in docs: anomaly_detection (no gallery entries), forecasting, segmentation, similarity_search, visualisation, self_supervised, imbalance transformations.
- Heavy emphasis on: numba-fast classic TSC algorithm families, collection data formats, and benchmarking against UEA/published results.
