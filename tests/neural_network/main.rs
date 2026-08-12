//! Integration tests for the `neural_network` feature.
//!
//! This is the crate root of the `neural_network` test binary. The binary is a `[[test]]`
//! target in `Cargo.toml` with `autotests = false`.
//!
//! Each file in this directory is a submodule for one topic. Shared helpers live in
//! [`common`]. The `gradient_check` module is the finite-difference harness that checks
//! backprop correctness.

/// Shared test helpers: a seeded RNG, a global-seed guard, and an array-closeness assertion.
mod common;
/// Finite-difference checks that compare analytic backward gradients to numerical derivatives.
mod gradient_check;

/// Tests for the activation layers and the `Activation` enum.
mod activation;
/// Tests for Conv1D and Conv2D forward values, shapes, and error paths.
mod conv_1d_2d;
/// Tests for Conv3D, DepthwiseConv2D, and SeparableConv2D.
mod conv_3d_variants;
/// Tests for Dense and Flatten: forward values, error paths, and parameter counts.
mod dense;
/// Tests for Dropout and the SpatialDropout variants.
mod dropout;
/// Tests for the loss functions: forward values, gradients, and error paths.
mod losses;
/// Tests for GaussianNoise and GaussianDropout.
mod noise;
/// Tests for BatchNormalization and LayerNormalization.
mod norm_batch_layer;
/// Tests for GroupNormalization and InstanceNormalization.
mod norm_group_instance;
/// Tests for the optimizers: SGD, Adam, AdamW, RMSprop, and AdaGrad.
mod optimizers;
/// Tests for AveragePooling and GlobalAveragePooling.
mod pooling_avg;
/// Tests for MaxPooling and GlobalMaxPooling in 1D, 2D, and 3D.
mod pooling_max;
/// Tests for the recurrent layers: SimpleRNN, LSTM, and GRU.
mod recurrent;
/// Tests for the reproducibility and RNG API.
mod reproducibility;
/// Tests for the Reshape layer: target-shape resolution, error paths, and round trips.
mod reshape;
/// Tests for `Sequential`: add, compile, fit, predict, and summary.
mod sequential;
/// Tests for `Sequential` model save and load round trips.
mod serialize;
