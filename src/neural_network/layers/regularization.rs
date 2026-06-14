//! Regularization layers for neural networks: dropout, noise injection, and normalization,
//! plus macros for the shared training-mode methods

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
