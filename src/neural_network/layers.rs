//! Neural network layers: the layer subsystem aggregator
//!
//! Declares every layer submodule and glob-re-exports the public layer types. It also defines
//! the shared infrastructure used across the subsystem: the
//! [`ParamCounts`](crate::neural_network::layers::ParamCounts) report (how many parameter
//! elements a layer holds, split into trainable and non-trainable), and the
//! `no_trainable_parameters_layer_functions` macro. That macro emits the `param_count` and
//! `get_weights` stubs for parameter-free layers.
//!
//! The submodules fall into a few categories:
//!
//! - Core layers (re-exported):
//!   - [`activation`](crate::neural_network::layers::activation)
//!   - [`border`](crate::neural_network::layers::border)
//!   - [`convolution`](crate::neural_network::layers::convolution)
//!   - [`dense`](crate::neural_network::layers::dense)
//!   - [`embedding`](crate::neural_network::layers::embedding)
//!   - [`flatten`](crate::neural_network::layers::flatten)
//!   - [`identity`](crate::neural_network::layers::identity)
//!   - [`permute`](crate::neural_network::layers::permute)
//!   - [`pooling`](crate::neural_network::layers::pooling)
//!   - [`recurrent`](crate::neural_network::layers::recurrent)
//!   - [`regularization`](crate::neural_network::layers::regularization)
//!   - [`repeat_vector`](crate::neural_network::layers::repeat_vector)
//!   - [`rescaling`](crate::neural_network::layers::rescaling)
//!   - [`reshape`](crate::neural_network::layers::reshape)
//!   - [`upsampling`](crate::neural_network::layers::upsampling)
//! - Weight containers: [`layer_weight`](crate::neural_network::layers::layer_weight)
//! - Shared (private) helpers: `conv_op_helpers` (2D/4D convolution zero-padding) and
//!   `shape_helpers` (pooling/convolution output-shape calculators)
//! - Validation: `validation` (shared input/weight checks)
//! - Serialization: [`serialize_model`](crate::neural_network::layers::serialize_model)
//!   (model-level snapshot and load-time weight application)

/// How many parameter elements a layer holds, split by whether training updates them
///
/// The 2 counts are independent, and a layer reports both. A Dense layer holds trainable
/// elements only. A pooling or activation layer holds none of either.
/// [`BatchNormalization`](crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization)
/// holds both: `gamma` and `beta` are trainable, and the running mean and the running variance
/// are not. The running statistics move on every training forward pass, but no optimizer ever
/// sees them, so they are non-trainable exactly as Keras 3 reports them
///
/// [`Sequential::summary`](crate::neural_network::sequential::Sequential::summary) adds the 2
/// columns over the model and prints the total of both
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParamCounts {
    /// Number of parameter elements that the optimizer updates
    pub trainable: usize,
    /// Number of parameter elements that the layer keeps but the optimizer never updates
    pub non_trainable: usize,
}

impl ParamCounts {
    /// A layer that holds no parameter at all
    ///
    /// # Returns
    ///
    /// - `ParamCounts` - Both counts set to 0
    #[inline]
    pub const fn none() -> Self {
        Self {
            trainable: 0,
            non_trainable: 0,
        }
    }

    /// A layer whose parameters are all trainable
    ///
    /// # Parameters
    ///
    /// - `count` - Number of trainable parameter elements
    ///
    /// # Returns
    ///
    /// - `ParamCounts` - The given trainable count, and 0 non-trainable
    #[inline]
    pub const fn trainable(count: usize) -> Self {
        Self {
            trainable: count,
            non_trainable: 0,
        }
    }

    /// A layer that holds both kinds of parameter
    ///
    /// # Parameters
    ///
    /// - `trainable` - Number of parameter elements that the optimizer updates
    /// - `non_trainable` - Number of parameter elements that the optimizer never updates
    ///
    /// # Returns
    ///
    /// - `ParamCounts` - The 2 given counts
    #[inline]
    pub const fn new(trainable: usize, non_trainable: usize) -> Self {
        Self {
            trainable,
            non_trainable,
        }
    }

    /// Every parameter element the layer holds, of both kinds
    ///
    /// # Returns
    ///
    /// - `usize` - The sum of the 2 counts
    #[inline]
    pub const fn total(&self) -> usize {
        self.trainable + self.non_trainable
    }
}

/// A module containing activation layer implementations for neural networks
pub mod activation;
/// Zero-padding and cropping layers that resize the spatial axes at their ends
pub mod border;
/// Convolution-internal helpers (output assembly, gradient accumulation, padding)
mod conv_op_helpers;
/// Convolutional layer for neural networks
pub mod convolution;
/// Dense (Fully Connected) layer implementation for neural networks
pub mod dense;
/// A trainable lookup table that turns whole-number indices into dense vectors
pub mod embedding;
/// A layer that flattens a 3D, 4D, or 5D tensor into a 2D tensor
pub mod flatten;
/// A layer that passes its input through unchanged
pub mod identity;
/// Container for different types of neural network layer weights
pub mod layer_weight;
/// A layer that reorders the axes after the batch axis
pub mod permute;
/// Pooling layer for neural networks
pub mod pooling;
/// Recurrent layer for neural networks
pub mod recurrent;
/// A module containing regularization layers for neural networks
pub mod regularization;
/// A layer that repeats a feature vector into a sequence
pub mod repeat_vector;
/// A layer that applies a fixed affine map to every element
pub mod rescaling;
/// A layer that rewrites the axes after the batch axis into a target shape
pub mod reshape;
/// Model-level serialization scaffolding (whole-model snapshot and load-time weight application)
pub mod serialize_model;
/// Output-shape calculators for pooling and convolution layers
mod shape_helpers;
/// Upsampling layers that enlarge the spatial axes by a whole-number factor
pub mod upsampling;
/// Shared input/weight validation for the layer module
mod validation;

pub use activation::*;
pub use border::*;
pub use convolution::*;
pub use dense::*;
pub use embedding::*;
pub use flatten::*;
pub use identity::*;
pub use permute::*;
pub use pooling::*;
pub use recurrent::*;
pub use regularization::*;
pub use repeat_vector::*;
pub use rescaling::*;
pub use reshape::*;
pub use upsampling::*;

/// Generates the trait method stubs for layers without trainable parameters
///
/// Such layers rely on the default [`Layer::parameters`] (an empty list, so the optimizer
/// skips them). This macro supplies the remaining required `param_count` and `get_weights`
/// methods
///
/// It is path-exported via a `pub(in ...) use` re-export, so callers import it explicitly
/// rather than depending on textual macro ordering:
/// `use crate::neural_network::layers::no_trainable_parameters_layer_functions;`
///
/// The generated `param_count` returns `ParamCounts::none()`, and `get_weights`
/// returns `LayerWeight::Empty`
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> ParamCounts {
            ParamCounts::none()
        }

        fn get_weights(&self) -> LayerWeight<'_> {
            LayerWeight::Empty
        }
    };
}
pub(in crate::neural_network::layers) use no_trainable_parameters_layer_functions;
