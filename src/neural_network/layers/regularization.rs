//! Regularization layers
//!
//! Re-exports the 3 families of regularization layers, plus a private `validation` submodule
//! of parameter and input-shape checks shared across the layers.
//!
//! The families are:
//! - dropout: [`Dropout`] and the spatial variants [`SpatialDropout1D`], [`SpatialDropout2D`], and
//!   [`SpatialDropout3D`]
//! - noise injection: [`GaussianNoise`] and [`GaussianDropout`]
//! - normalization: [`BatchNormalization`], [`LayerNormalization`], [`GroupNormalization`],
//!   [`InstanceNormalization`], and [`UnitNormalization`]
//! Every dropout layer, every noise layer, and `BatchNormalization` change what forward computes
//! between training and inference. `LayerNormalization`, `GroupNormalization`,
//! `InstanceNormalization`, and `UnitNormalization` give the same output either way. Each one
//! takes its statistic from a single sample, never from the batch. The mode is not a field of a
//! layer. It is the training flag of the [`Ctx`] that the pass carries, and every layer of this
//! module reads it with [`Ctx::is_training`]
//!
//! [`Dropout`]: crate::neural_network::layers::Dropout
//! [`SpatialDropout1D`]: crate::neural_network::layers::SpatialDropout1D
//! [`SpatialDropout2D`]: crate::neural_network::layers::SpatialDropout2D
//! [`SpatialDropout3D`]: crate::neural_network::layers::SpatialDropout3D
//! [`GaussianNoise`]: crate::neural_network::layers::GaussianNoise
//! [`GaussianDropout`]: crate::neural_network::layers::GaussianDropout
//! [`BatchNormalization`]: crate::neural_network::layers::BatchNormalization
//! [`LayerNormalization`]: crate::neural_network::layers::LayerNormalization
//! [`GroupNormalization`]: crate::neural_network::layers::GroupNormalization
//! [`InstanceNormalization`]: crate::neural_network::layers::InstanceNormalization
//! [`UnitNormalization`]: crate::neural_network::layers::UnitNormalization
//! [`Ctx`]: crate::neural_network::Ctx
//! [`Ctx::is_training`]: crate::neural_network::Ctx::is_training

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
