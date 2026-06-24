//! Noise-injection regularization layers
//!
//! Groups the two noise-injection layers and re-exports them: multiplicative
//! [`GaussianDropout`](crate::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout),
//! which scales inputs by `N(1, rate/(1 - rate))` during training, and additive
//! [`GaussianNoise`](crate::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise),
//! which adds zero-mean `N(0, stddev^2)` noise during training; both are identity maps at inference
//!
//! This file defines no shared infrastructure of its own. The two layers reuse the
//! training-mode macros (`mode_dependent_layer_set_training` and `mode_dependent_layer_trait`)
//! and the validation helpers from the parent
//! [`regularization`](crate::neural_network::layers::regularization) module

/// Gaussian Dropout layer for neural networks
pub mod gaussian_dropout;
/// Gaussian Noise layer for neural networks
pub mod gaussian_noise;

pub use gaussian_dropout::GaussianDropout;
pub use gaussian_noise::GaussianNoise;
