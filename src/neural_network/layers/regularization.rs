//! Regularization layers
//!
//! Re-exports the 3 families of regularization layers, plus a private `validation` submodule
//! of parameter and input-shape checks shared across the layers.
//!
//! The families are:
//! - dropout: [`Dropout`](crate::neural_network::layers::regularization::dropout::dropout::Dropout)
//!   and the spatial variants [`SpatialDropout1D`](crate::neural_network::layers::regularization::dropout::spatial_dropout_1d::SpatialDropout1D),
//!   [`SpatialDropout2D`](crate::neural_network::layers::regularization::dropout::spatial_dropout_2d::SpatialDropout2D),
//!   and [`SpatialDropout3D`](crate::neural_network::layers::regularization::dropout::spatial_dropout_3d::SpatialDropout3D)
//! - noise injection: [`GaussianNoise`](crate::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise)
//!   and [`GaussianDropout`](crate::neural_network::layers::regularization::noise_injection::gaussian_dropout::GaussianDropout)
//! - normalization: [`BatchNormalization`](crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization),
//!   [`LayerNormalization`](crate::neural_network::layers::regularization::normalization::layer_normalization::LayerNormalization),
//!   [`GroupNormalization`](crate::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization),
//!   [`InstanceNormalization`](crate::neural_network::layers::InstanceNormalization),
//!   and [`UnitNormalization`](crate::neural_network::layers::UnitNormalization)
//!
//! Every layer here behaves differently in training versus inference, except
//! `UnitNormalization`, which reads no statistic of the batch and draws from no RNG. The mode
//! is not a field of a layer. It is the training flag of the
//! [`Ctx`](crate::neural_network::Ctx) that the pass carries, and every layer of this module
//! reads it with [`Ctx::is_training`](crate::neural_network::Ctx::is_training).

/// Dropout layers for neural networks
pub mod dropout;
/// Noise injection layers for neural networks
pub mod noise_injection;
/// Normalization layers for neural networks
pub mod normalization;
/// Input validation functions for regularization layers
mod validation;

pub use dropout::*;
pub use noise_injection::*;
pub use normalization::*;
