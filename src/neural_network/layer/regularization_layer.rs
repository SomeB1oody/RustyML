/// A macro to define a layer-specific method for setting the training mode.
///
/// This macro generates a `set_training` method within the implementing object
/// to allow toggling the training mode between training (`true`) and inference (`false`).
///
/// The generated method is used to manage the state of `training` within the object,
/// which can be critical for operations like forward and backward passes in machine learning models.
macro_rules! mode_dependent_layer_set_training {
    () => {
        /// Sets the training mode for the object.
        ///
        /// # Arguments
        ///
        /// * `is_training` - A boolean value indicating whether the object should be
        ///   in training mode (`true`) or not (`false`).
        ///
        /// # Effects
        ///
        /// This method modifies the `training` field of the object to reflect the provided value.
        /// It is commonly used to toggle the state of an object between training and inference modes.
        pub fn set_training(&mut self, is_training: bool) {
            self.training = is_training;
        }
    };
}

/// A macro that defines a method `set_training_if_mode_dependent` for a layer that may have
/// behavior dependent on whether it is in training or inference mode.
macro_rules! mode_dependent_layer_trait {
    () => {
        fn set_training_if_mode_dependent(&mut self, is_training: bool) {
            self.set_training(is_training);
        }
    };
}

/// Dropout layers for neural networks
pub mod dropout_layer;
/// Input validation functions for regularization layers
mod input_validation_function;
/// Noise injection layers for neural networks
pub mod noise_injection_layer;
/// Normalization layers for neural networks
pub mod normalization_layer;

pub use dropout_layer::*;
pub use noise_injection_layer::*;
pub use normalization_layer::*;
