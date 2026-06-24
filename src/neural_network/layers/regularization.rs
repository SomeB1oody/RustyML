//! Regularization layers and the shared training-mode infrastructure that backs them
//!
//! Re-exports the three families of regularization layers and defines the macros that generate
//! their common training-mode methods, plus a private `validation` submodule of parameter and
//! input-shape checks shared across the layers.
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
//!   and [`InstanceNormalization`](crate::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization)
//!
//! Because every one of these layers behaves differently in training versus inference, the module
//! defines two macros for toggling the shared `training` field: `mode_dependent_layer_set_training`
//! generates the inherent `set_training` method, and `mode_dependent_layer_trait` generates the
//! `set_training_if_mode_dependent` trait method that delegates to it.

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

/// Defines a layer-specific `set_training` method for toggling training mode
///
/// The generated method sets the `training` field to `true` (training) or `false`
/// (inference). This drives behavior in the forward and backward passes
macro_rules! mode_dependent_layer_set_training {
    () => {
        /// Sets the training mode for the layer, updating its `training` field
        ///
        /// # Parameters
        ///
        /// - `is_training` - whether the layer should be in training mode (`true`) or inference mode (`false`)
        pub fn set_training(&mut self, is_training: bool) {
            self.training = is_training;
        }
    };
}
pub(in crate::neural_network::layers::regularization) use mode_dependent_layer_set_training;

/// Defines the trait method `set_training_if_mode_dependent` for a layer whose behavior
/// depends on training versus inference mode
macro_rules! mode_dependent_layer_trait {
    () => {
        fn set_training_if_mode_dependent(&mut self, is_training: bool) {
            self.set_training(is_training);
        }
    };
}
pub(in crate::neural_network::layers::regularization) use mode_dependent_layer_trait;
