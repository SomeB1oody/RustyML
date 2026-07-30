//! Integration tests for the `utils` feature
//!
//! Crate root of the `utils` test binary (declared as a `[[test]]` target in
//! `Cargo.toml`, `autotests = false`). Per-topic files in this directory are submodules.
//! Shared helpers live in [`common`]

/// Shared test helpers: the seeded RNG and the array-comparison assertion
mod common;

/// One-hot encoding and argmax-decode tests for `utils::label_encoding`
mod label_encoding;
/// Tests for the stateless `normalize` function across axes and norm orders
mod normalize;
/// Tests for `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, and `Normalizer`
mod scaler;
/// Tests for the stateless `standardize` function across axes
mod standardize;
/// Tests for `train_test_split` and `train_test_split_stratified`
mod train_test_split;
