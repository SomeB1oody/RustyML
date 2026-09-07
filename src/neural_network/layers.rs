//! Neural network layers: the layer subsystem aggregator
//!
//! Declares every layer submodule and glob-re-exports the public layer types. It also defines
//! the shared infrastructure used across the subsystem: the
//! [`ParamCounts`](crate::neural_network::layers::ParamCounts) report (how many parameter
//! elements a layer holds, split into trainable and non-trainable), and the 2 macros that
//! give a layer its weight methods. `no_trainable_parameters_layer_functions` emits the stubs
//! of a parameter-free layer, and `named_weight_layer_functions` builds the named array list
//! of a layer that holds arrays.
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
//!   - [`merge`](crate::neural_network::layers::merge)
//!   - [`permute`](crate::neural_network::layers::permute)
//!   - [`pooling`](crate::neural_network::layers::pooling)
//!   - [`recurrent`](crate::neural_network::layers::recurrent)
//!   - [`regularization`](crate::neural_network::layers::regularization)
//!   - [`repeat_vector`](crate::neural_network::layers::repeat_vector)
//!   - [`rescaling`](crate::neural_network::layers::rescaling)
//!   - [`reshape`](crate::neural_network::layers::reshape)
//!   - [`upsampling`](crate::neural_network::layers::upsampling)
//! - Shared (private) helpers: `conv_op_helpers` (2D/4D convolution zero-padding) and
//!   `shape_helpers` (pooling/convolution output-shape calculators)
//! - Validation: `validation` (shared input/weight checks)
//! - Serialization: [`checkpoint`](crate::neural_network::layers::checkpoint)
//!   (the named on-disk format, and the load that applies it)

/// How many parameter elements a layer holds, split by whether training updates them
///
/// The 2 counts are independent, and a layer reports both. A Dense layer holds trainable
/// elements only. A pooling or activation layer holds none of either.
/// [`BatchNormalization`] holds both: `gamma` and `beta` are trainable, and the running mean
/// and the running variance are not. The running statistics move on every training forward pass, but no optimizer ever
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
/// The named checkpoint format: what a saved model holds, and how a load applies it
pub mod checkpoint;
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
/// Layers that take several inputs and reduce or join them into 1 output
pub mod merge;
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
/// A layer that reverses the order along 1 axis
pub mod reverse;
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
pub use merge::*;
pub use permute::*;
pub use pooling::*;
pub use recurrent::*;
pub use regularization::*;
pub use repeat_vector::*;
pub use rescaling::*;
pub use reshape::*;
pub use reverse::*;
pub use upsampling::*;

/// Generates the trait method stubs for layers without trainable parameters
///
/// Such layers rely on the default [`LayerBase::parameters_mut`] (an empty list, so the optimizer
/// skips them). This macro supplies the remaining required `param_count`, `weights`, and
/// `weights_mut` methods
///
/// It is path-exported via a `pub(in ...) use` re-export, so callers import it explicitly
/// rather than depending on textual macro ordering:
/// `use crate::neural_network::layers::no_trainable_parameters_layer_functions;`
///
/// The generated `param_count` returns `ParamCounts::none()`, and both weight methods return
/// the empty vector, so such a layer contributes no path to a checkpoint
///
/// [`LayerBase::parameters_mut`]: crate::neural_network::traits::LayerBase::parameters_mut
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> ParamCounts {
            ParamCounts::none()
        }

        fn weights(&self) -> Vec<$crate::neural_network::traits::WeightRef<'_>> {
            Vec::new()
        }

        fn weights_mut(&mut self) -> Vec<$crate::neural_network::traits::WeightMut<'_>> {
            Vec::new()
        }
    };
}
pub(in crate::neural_network::layers) use no_trainable_parameters_layer_functions;

/// Generates the `weights` and `weights_mut` methods of a layer that holds arrays
///
/// 1 list serves both directions, so the name, the kind, and the order of an array cannot
/// drift between the read path and the write path. The checkpoint format reads the name and
/// the kind from it, so this list is the layer half of every checkpoint path
///
/// Each entry reads `trainable "<name>" => <field>` or `non_trainable "<name>" => <field>`.
/// The name is the Keras 3 name of the array, and the field is the field of the layer struct
/// that holds it. A field of a nested struct is written with dots, such as `gates.kernel`.
/// The 2 halves stay next to each other, so a renamed field breaks this list instead of
/// leaving a stale name behind
///
/// An entry that ends in `if <flag>` is optional. The flag is a `bool` field of the same
/// layer, and the array reaches the list only while the flag is true. `Dense` writes
/// `trainable "bias" => bias if use_bias`, so a layer built with `use_bias` set to false
/// exposes the kernel alone and its checkpoint holds 1 path. An optional entry moves no other
/// entry, because a checkpoint addresses an array by name and never by position
///
/// Give an array the same name that [`LayerBase::parameters_mut`] gives it. The compiler binds
/// neither the name nor the storage. A parameter and the array of 1 name must be 1 storage,
/// because an optimizer writes through the parameter and a checkpoint reads the array. Some
/// tests hold that rule for 1 layer at a time, such as
/// `dropping_gamma_does_not_give_beta_the_optimizer_state_of_gamma` of
/// `tests/neural_network/optional_parameters.rs`. No test holds it for every layer type
///
/// It is path-exported like `no_trainable_parameters_layer_functions`:
/// `use crate::neural_network::layers::named_weight_layer_functions;`
///
/// [`LayerBase::parameters_mut`]: crate::neural_network::traits::LayerBase::parameters_mut
macro_rules! named_weight_layer_functions {
    ($($kind:ident $name:literal => $($field:ident).+ $(if $flag:ident)?),+ $(,)?) => {
        fn weights(&self) -> Vec<$crate::neural_network::traits::WeightRef<'_>> {
            [$($crate::neural_network::layers::optional_named_weight!(
                $crate::neural_network::traits::WeightRef::$kind(
                    $name,
                    self.$($field).+.view().into_dyn(),
                )
                $(, self.$flag)?
            )),+]
                .into_iter()
                .flatten()
                .collect()
        }

        fn weights_mut(&mut self) -> Vec<$crate::neural_network::traits::WeightMut<'_>> {
            [$($crate::neural_network::layers::optional_named_weight!(
                $crate::neural_network::traits::WeightMut::$kind(
                    $name,
                    self.$($field).+.view_mut().into_dyn(),
                )
                $(, self.$flag)?
            )),+]
                .into_iter()
                .flatten()
                .collect()
        }
    };
}
pub(in crate::neural_network::layers) use named_weight_layer_functions;

/// Generates the 2 build reports of a layer that holds a `built: Option<Shape>` field
///
/// [`Layer::known_input_shape`] gives the shape the layer built for, and
/// [`Layer::build_config`] gives the same shape with the batch axis freed, which is what a
/// checkpoint records. A layer whose displayed input shape comes from somewhere else uses
/// `build_config_function` instead and writes its own `known_input_shape`
///
/// It is path-exported like `no_trainable_parameters_layer_functions`:
/// `use crate::neural_network::layers::built_layer_shape_functions;`
///
/// [`Layer::known_input_shape`]: crate::neural_network::traits::Layer::known_input_shape
/// [`Layer::build_config`]: crate::neural_network::traits::UnaryLayer::build_config
macro_rules! built_layer_shape_functions {
    () => {
        fn known_input_shapes(&self) -> Option<Vec<$crate::neural_network::Shape>> {
            self.built
                .as_ref()
                .map(|shape| vec![$crate::neural_network::Shape::free_batch(shape)])
        }

        $crate::neural_network::layers::build_config_function!();
    };
}
pub(in crate::neural_network::layers) use built_layer_shape_functions;

/// Generates [`Layer::build_config`] of a layer that holds a `built: Option<Shape>` field
///
/// [`Layer::build_config`]: crate::neural_network::traits::UnaryLayer::build_config
macro_rules! build_config_function {
    () => {
        fn is_built(&self) -> bool {
            self.built.is_some()
        }

        fn build_config(&self) -> Option<$crate::neural_network::layers::checkpoint::BuildConfig> {
            self.built
                .as_ref()
                .map($crate::neural_network::layers::checkpoint::BuildConfig::unary)
        }
    };
}
pub(in crate::neural_network::layers) use build_config_function;

/// Wraps 1 named array as the `Option` that `named_weight_layer_functions` collects
///
/// The rule with 1 argument takes an array that the layer always holds, and it is always
/// `Some`. The rule with 2 takes an optional array and the flag that decides whether the layer
/// holds it. Splitting the 2 rules keeps the always-present case free of any run-time test
macro_rules! optional_named_weight {
    ($entry:expr) => {
        Some($entry)
    };
    ($entry:expr, $flag:expr) => {
        $flag.then(|| $entry)
    };
}
pub(in crate::neural_network::layers) use optional_named_weight;
