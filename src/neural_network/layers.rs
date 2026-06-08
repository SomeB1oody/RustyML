/// Enumeration representing different types of training parameters for neural network layers.
///
/// This enum categorizes layers based on their parameter training capabilities:
/// - Layers with trainable parameters (e.g., Dense, Convolutional layers)
/// - Layers without trainable parameters (e.g., Pooling, Activation layers)
/// - Special layers that have no parameters at all
///
/// # Variants
///
/// - `Trainable(usize)` - The layer has trainable parameters, with the count specified by the `usize` value
/// - `NonTrainable(usize)` - The layer has parameters but they are not trainable (e.g., frozen layers)
/// - `NoTrainable` - The layer has no trainable parameters (e.g., pooling layers, activation functions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingParameters {
    /// The layer contains trainable parameters that will be updated during optimization.
    /// The `usize` value indicates the total number of trainable parameters.
    Trainable(usize),
    /// The layer contains parameters but they are not trainable (frozen).
    /// The `usize` value indicates the total number of non-trainable parameters.
    NonTrainable(usize),
    /// The layer has no trainable parameters whatsoever.
    NoTrainable,
}

/// A module containing activation layer implementations for neural networks
pub mod activation;
/// Convolutional layer for neural networks
pub mod convolution;
/// Dense (Fully Connected) layer implementation for neural networks
pub mod dense;
/// A layer that flattens a 3D, 4D, or 5D tensor into a 2D tensor
pub mod flatten;
/// Convolution-internal helpers (output assembly, gradient accumulation, padding)
mod conv_op_helpers;
/// Output-shape calculators for pooling and convolution layers
mod shape_helpers;
/// Shared input/weight validation for the layer module
mod validation;
/// Container for different types of neural network layer weights
pub mod layer_weight;
/// Pooling layer for neural networks
pub mod pooling;
/// Recurrent layer for neural networks
pub mod recurrent;
/// A module containing regularization layers for neural networks
pub mod regularization;
/// A module containing helper functions and structs for serializing neural network weights
pub mod serialize_weight;

pub use activation::*;
pub use convolution::*;
pub use dense::*;
pub use flatten::*;
pub use pooling::*;
pub use recurrent::*;
pub use regularization::*;

/// A macro that generates standard trait method implementations for neural network layers
/// without trainable parameters.
///
/// Layers with no trainable parameters rely on the default [`Layer::parameters`] (which returns
/// an empty list, so the optimizer skips them); this macro only supplies the remaining required
/// methods.
///
/// Defined *after* the `mod` declarations and path-exported via a `pub(in ...) use` re-export, so callers import
/// it explicitly (`use crate::neural_network::layers::no_trainable_parameters_layer_functions;`)
/// rather than relying on textual macro ordering.
///
/// # Parameters
///
/// - `param_count`: Returns `TrainingParameters::NoTrainable` as the layer has no trainable parameters
/// - `get_weights`: Returns `LayerWeight::Empty` as the layer has no weights
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> TrainingParameters {
            // This layer has no trainable parameters
            TrainingParameters::NoTrainable
        }

        fn get_weights(&self) -> LayerWeight<'_> {
            // This layer has no weights
            LayerWeight::Empty
        }
    };
}
pub(in crate::neural_network::layers) use no_trainable_parameters_layer_functions;
