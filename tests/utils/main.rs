//! Integration tests for the `utils` feature
//!
//! Crate root of the `utils` test binary (declared as a `[[test]]` target in
//! `Cargo.toml`, `autotests = false`). Per-topic files in this directory are submodules
//! Shared helpers live in [`common`]

mod common;

mod label_encoding;
mod normalize;
mod standardize;
mod train_test_split;
