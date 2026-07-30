//! Integration tests for the `machine_learning` feature.
//!
//! This is the crate root of the `machine_learning` test binary, declared as a `[[test]]`
//! target in `Cargo.toml`. Each per-algorithm file in this directory is a submodule.
//!
//! Shared helpers (a seeded RNG constructor and `assert_allclose`) live in [`common`].

/// Shared helpers: a seeded RNG constructor and `assert_allclose`.
mod common;

/// Tests for `DBSCAN`: constructor validation, fit and predict errors, and cluster correctness.
mod dbscan;

/// Tests for the decision tree classifier: constructor validation and closed-form predictions.
mod decision_tree;

/// Tests for `IsolationForest`: constructor validation, scoring, and outlier detection.
mod isolation_forest;

/// Tests for kernel PCA: constructor validation and per-kernel fit and transform checks.
mod kernel_pca;

/// Tests for `KMeans`: fit and predict errors, inertia, determinism, and save/load round-trips.
mod kmeans;

/// Tests for the KNN classifier: distance metrics, weighting, and save/load round-trips.
mod knn;

/// Tests for `LDA` (linear discriminant analysis): fit and predict, and closed-form checks.
mod linear_discriminant_analysis;

/// Tests for `LinearRegression`: constructor validation, both solvers, and regularization.
mod linear_regression;

/// Tests for `LinearSVC`: constructor validation, label domain, and penalties.
mod linear_svc;

/// Tests for `LogisticRegression` and `generate_polynomial_features`.
mod logistic_regression;

/// Tests for `MeanShift` clustering and `estimate_bandwidth`.
mod mean_shift;

/// Cross-cutting tests for the Fit and Predict traits, save/load, and NotFitted errors.
mod ml_infra;

/// Tests for PCA: construction, validation, and each SVD solver.
mod pca;

/// Tests for `SVC`: label domain and decision function sign.
mod svc;

/// Tests for `TSNE`: fit_transform shape, determinism, and neighborhood preservation.
mod t_sne;
