//! Neural network layers: the layer subsystem aggregator
//!
//! Declares every layer submodule, glob-re-exports the public layer types, and defines the
//! shared infrastructure used across the subsystem: the
//! [`TrainingParameters`](crate::neural_network::layers::TrainingParameters) classification (a
//! layer is `Trainable`, `NonTrainable`, or `NoTrainable`) and the
//! `no_trainable_parameters_layer_functions` macro that emits the `param_count`/`get_weights`
//! stubs for parameter-free layers.
//!
//! The submodules fall into a few categories:
//!
//! - Core layers (re-exported): [`activation`](crate::neural_network::layers::activation),
//!   [`convolution`](crate::neural_network::layers::convolution),
//!   [`dense`](crate::neural_network::layers::dense),
//!   [`flatten`](crate::neural_network::layers::flatten),
//!   [`pooling`](crate::neural_network::layers::pooling),
//!   [`recurrent`](crate::neural_network::layers::recurrent), and
//!   [`regularization`](crate::neural_network::layers::regularization)
//! - Weight containers: [`layer_weight`](crate::neural_network::layers::layer_weight)
//! - Shared (private) helpers: `conv_op_helpers` (2D/4D convolution zero-padding) and
//!   `shape_helpers` (pooling/convolution output-shape calculators)
//! - Validation: `validation` (shared input/weight checks)
//! - Serialization: [`serialize_model`](crate::neural_network::layers::serialize_model)
//!   (model-level snapshot and load-time weight application)

/// Classifies a layer by its parameter training capability
///
/// Layers fall into 3 groups: those with trainable parameters (e.g. Dense,
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
/// Model-level serialization scaffolding (whole-model snapshot and load-time weight application)
pub mod serialize_model;
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
/// It is path-exported via a `pub(in ...) use` re-export, so callers import it explicitly
/// rather than depending on textual macro ordering:
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
