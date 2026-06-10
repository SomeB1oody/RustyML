//! Integration tests for the `machine_learning` feature.
//!
//! This file is the crate root of the `machine_learning` test binary, declared
//! explicitly as a `[[test]]` target in `Cargo.toml` (with `autotests = false`, which
//! avoids the previous double-build). Each per-algorithm file in this directory is a
//! submodule compiled exactly once.
//!
//! Shared helpers (seeded RNG, `assert_allclose`, dataset builders) live in [`common`].
//! As each algorithm's tests are (re)written, add its `mod <name>;` line below.

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
