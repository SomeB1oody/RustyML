//! Integration tests for the `neural_network` feature
//!
//! Crate root of the `neural_network` test binary (declared as a `[[test]]` target in
//! `Cargo.toml`, `autotests = false`). Per-topic files in this directory are submodules,
//! shared helpers live in [`common`], and `gradient_check` is the finite-difference
//! backprop-correctness harness

mod common;
mod gradient_check;

mod activation;
mod conv_1d_2d;
mod conv_3d_variants;
mod dense;
mod dropout;
mod losses;
mod noise;
mod norm_batch_layer;
mod norm_group_instance;
mod optimizers;
mod pooling_avg;
mod pooling_max;
mod recurrent;
mod reproducibility;
mod sequential;
mod serialize;
