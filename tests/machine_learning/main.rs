//! Integration tests for the `machine_learning` feature
//!
//! Crate root of the `machine_learning` test binary, declared as a `[[test]]` target
//! in `Cargo.toml`. Each per-algorithm file in this directory is a submodule
//! compiled once
//!
//! Shared helpers (seeded RNG, `assert_allclose`, dataset builders) live in [`common`]

mod common;

mod dbscan;
mod decision_tree;
mod isolation_forest;
mod kmeans;
mod knn;
mod linear_discriminant_analysis;
mod linear_regression;
mod linear_svc;
mod logistic_regression;
mod mean_shift;
mod ml_infra;
mod svc;
