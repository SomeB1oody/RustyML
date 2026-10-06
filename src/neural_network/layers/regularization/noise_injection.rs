//! Noise-injection regularization layers
//!
//! Groups the 2 noise-injection layers and re-exports them. Multiplicative
//! [`GaussianDropout`] scales inputs by `N(1, rate/(1 - rate))` during training. Additive
//! [`GaussianNoise`] adds zero-mean `N(0, stddev^2)` noise during training. Both are identity maps
//! at inference.
//!
//! This file defines no shared infrastructure of its own. The 2 layers read the training flag
//! of the [`Ctx`], and they reuse the validation helpers from the parent [`regularization`] module
//!
//! [`GaussianDropout`]: crate::neural_network::layers::GaussianDropout
//! [`GaussianNoise`]: crate::neural_network::layers::GaussianNoise
//! [`Ctx`]: crate::neural_network::Ctx
//! [`regularization`]: crate::neural_network::layers::regularization

/// Gaussian Dropout layer for neural networks
pub mod gaussian_dropout;
/// Gaussian Noise layer for neural networks
pub mod gaussian_noise;

pub use gaussian_dropout::GaussianDropout;
pub use gaussian_noise::GaussianNoise;
