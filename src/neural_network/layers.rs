//! Neural network layer module: submodule declarations, the shared
//! [`TrainingParameters`](crate::neural_network::layers::TrainingParameters) classification, and helpers for parameter-free layers

/// Classifies a layer by its parameter training capability
///
/// Layers fall into three groups: those with trainable parameters (e.g. Dense,
/// convolutional layers), those whose parameters are frozen, and those with no
/// parameters at all (e.g. pooling, activation layers)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingParameters {
    /// Layer has trainable parameters updated during optimization; the `usize` is the count
    Trainable(usize),
    /// Layer has parameters but they are frozen; the `usize` is the count of non-trainable parameters
    NonTrainable(usize),
    /// Layer has no trainable parameters whatsoever
    NoTrainable,
}

/// A module containing activation layer implementations for neural networks
pub mod activation;
/// Convolution-internal helpers (output assembly, gradient accumulation, padding)
mod conv_op_helpers;
/// Convolutional layer for neural networks
pub mod convolution;
/// Dense (Fully Connected) layer implementation for neural networks
pub mod dense;
/// A layer that flattens a 3D, 4D, or 5D tensor into a 2D tensor
pub mod flatten;
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
/// Output-shape calculators for pooling and convolution layers
mod shape_helpers;
/// Shared input/weight validation for the layer module
mod validation;

pub use activation::*;
pub use convolution::*;
pub use dense::*;
pub use flatten::*;
pub use pooling::*;
pub use recurrent::*;
pub use regularization::*;

/// Generates the trait method stubs for layers without trainable parameters
///
/// Such layers rely on the default [`Layer::parameters`] (an empty list, so the optimizer
/// skips them); this macro supplies the remaining required `param_count` and `get_weights`
///
/// Defined after the `mod` declarations and path-exported via a `pub(in ...) use` re-export,
/// so callers import it explicitly rather than relying on textual macro ordering:
/// `use crate::neural_network::layers::no_trainable_parameters_layer_functions;`
///
/// The generated `param_count` returns `TrainingParameters::NoTrainable` and `get_weights`
/// returns `LayerWeight::Empty`
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> TrainingParameters {
            TrainingParameters::NoTrainable
        }

        fn get_weights(&self) -> LayerWeight<'_> {
            LayerWeight::Empty
        }
    };
}
pub(in crate::neural_network::layers) use no_trainable_parameters_layer_functions;
