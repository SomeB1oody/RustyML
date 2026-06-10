//! Integration tests for the `metrics` feature.
//!
//! Crate root of the `metrics` test binary (declared as a `[[test]]` target in
//! `Cargo.toml`, `autotests = false`). The previous single 862-line `metric_test.rs`
//! is split here into one submodule per metric domain. Shared helpers live in [`common`].

mod common;

mod classification;
mod clustering;
mod regression;
