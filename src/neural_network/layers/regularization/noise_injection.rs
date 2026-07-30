//! Noise-injection regularization layers
//!
//! Groups the 2 noise-injection layers and re-exports them. Multiplicative
//! [`GaussianDropout`](crate::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout)
//! scales inputs by `N(1, rate/(1 - rate))` during training. Additive
//! [`GaussianNoise`](crate::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise)
//! adds zero-mean `N(0, stddev^2)` noise during training. Both are identity maps at inference.
//!
//! This file defines no shared infrastructure of its own. The 2 layers reuse the
//! training-mode macros (`mode_dependent_layer_set_training` and `mode_dependent_layer_trait`)
//! and the validation helpers from the parent
//! [`regularization`](crate::neural_network::layers::regularization) module

/// Gaussian Dropout layer for neural networks
pub mod gaussian_dropout;
/// Gaussian Noise layer for neural networks
pub mod gaussian_noise;

pub use gaussian_dropout::GaussianDropout;
pub use gaussian_noise::GaussianNoise;
