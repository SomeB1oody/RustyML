use super::*;
use ndarray::Zip;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};
use rayon::prelude::*;

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

/// A macro that generates standard function implementations for neural network layers
/// without trainable parameters.
///
/// This macro expands to implementations of the following functions:
/// - `param_count`: Returns 0 as the layer has no trainable parameters
/// - `update_parameters_sgd`: Empty implementation for SGD parameter updates
/// - `update_parameters_adam`: Empty implementation for Adam parameter updates
/// - `update_parameters_rmsprop`: Empty implementation for RMSProp parameter updates
/// - `get_weights`: Returns `LayerWeight::Empty` as the layer has no weights
macro_rules! no_trainable_parameters_layer_functions {
    () => {
        fn param_count(&self) -> TrainingParameters {
            // This layer has no trainable parameters
            TrainingParameters::NoTrainable
        }

        fn update_parameters_sgd(&mut self, _lr: f32) {
            // This layer have no trainable parameters
        }

        fn update_parameters_adam(
            &mut self,
            _lr: f32,
            _beta1: f32,
            _beta2: f32,
            _epsilon: f32,
            _t: u64,
        ) {
            // This layer have no trainable parameters
        }

        fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
            // This layer have no trainable parameters
        }

        fn update_parameters_ada_grad(&mut self, _lr: f32, _epsilon: f32) {
            // This layer have no trainable parameters
        }

        fn get_weights(&self) -> LayerWeight<'_> {
            // This layer has no weights
            LayerWeight::Empty
        }
    };
}

/// A macro that conditionally executes parallel or sequential computation based on a threshold.
///
/// This macro is designed to eliminate code duplication in both convolution and pooling layers
/// by providing a unified pattern for choosing between parallel and sequential execution strategies.
///
/// # Parameters
///
/// - `$batch_size` - Number of batches
/// - `$channels` - Number of channels
/// - `$threshold` - Threshold for parallel execution (when batch_size * channels >= threshold)
/// - `$compute_fn` - Closure/function to execute for each (batch, channel) pair
macro_rules! execute_parallel_or_sequential {
    ($batch_size:expr, $channels:expr, $threshold:expr, $compute_fn:expr) => {
        if $batch_size * $channels >= $threshold {
            // Parallel execution for large workloads
            (0..$batch_size)
                .into_par_iter()
                .flat_map(|b| {
                    (0..$channels)
                        .into_par_iter()
                        .map(move |c| $compute_fn(b, c))
                })
                .collect()
        } else {
            // Sequential execution for small workloads
            (0..$batch_size)
                .flat_map(|b| (0..$channels).map(move |c| $compute_fn(b, c)))
                .collect()
        }
    };
}

/// A macro that merges gradient results back into the gradient tensor for 1D spatial data.
///
/// This macro handles the common pattern of writing computed spatial gradients
/// back to the gradient tensor for 1D pooling and convolution layers.
///
/// # Parameters
///
/// - `$grad_tensor` - The gradient tensor to write to
/// - `$results` - Iterator of results in format ((batch, channel), spatial_grad)
/// - `$length` - Length of the spatial dimension
macro_rules! merge_gradients_1d {
    ($grad_tensor:expr, $results:expr, $length:expr) => {
        for ((b, c), spatial_grad) in $results {
            for (idx, grad_val) in spatial_grad.iter().enumerate() {
                $grad_tensor[[b, c, idx]] = *grad_val;
            }
        }
    };
}

/// A macro that merges gradient results back into the gradient tensor for 2D spatial data.
///
/// This macro handles the common pattern of writing computed spatial gradients
/// back to the gradient tensor for 2D pooling and convolution layers.
///
/// # Parameters
///
/// - `$grad_tensor` - The gradient tensor to write to
/// - `$results` - Iterator of results in format ((batch, channel), spatial_grad)
/// - `$height` - Height of the spatial dimension
/// - `$width` - Width of the spatial dimension
macro_rules! merge_gradients_2d {
    ($grad_tensor:expr, $results:expr, $height:expr, $width:expr) => {
        for ((b, c), spatial_grad) in $results {
            for h in 0..$height {
                for w in 0..$width {
                    let flat_idx = h * $width + w;
                    $grad_tensor[[b, c, h, w]] = spatial_grad[flat_idx];
                }
            }
        }
    };
}

/// A macro that merges gradient results back into the gradient tensor for 3D spatial data.
///
/// This macro handles the common pattern of writing computed spatial gradients
/// back to the gradient tensor for 3D pooling and convolution layers.
///
/// # Parameters
///
/// - `$grad_tensor` - The gradient tensor to write to
/// - `$results` - Iterator of results in format ((batch, channel), spatial_grad)
/// - `$depth` - Depth of the spatial dimension
/// - `$height` - Height of the spatial dimension
/// - `$width` - Width of the spatial dimension
macro_rules! merge_gradients_3d {
    ($grad_tensor:expr, $results:expr, $depth:expr, $height:expr, $width:expr) => {
        for ((b, c), spatial_grad) in $results {
            for d in 0..$depth {
                for h in 0..$height {
                    for w in 0..$width {
                        let flat_idx = d * ($height * $width) + h * $width + w;
                        $grad_tensor[[b, c, d, h, w]] = spatial_grad[flat_idx];
                    }
                }
            }
        }
    };
}

/// A module containing activation layer implementations for neural networks
pub mod activation_layer;
/// Convolutional layer for neural networks
pub mod convolution_layer;
/// Dense (Fully Connected) layer implementation for neural networks
pub mod dense;
/// A layer that flattens a 4D tensor into a 2D tensor
pub mod flatten;
/// A module containing helper functions for neural network layers
mod helper_function;
/// Container for different types of neural network layer weights
pub mod layer_weight;
/// Pooling layer for neural networks
pub mod pooling_layer;
/// Recurrent layer for neural networks
pub mod recurrent_layer;
/// A module containing regularization layers for neural networks
pub mod regularization_layer;
/// A module containing helper functions and structs for serializing neural network weights
pub mod serialize_weight;

pub use activation_layer::*;
pub use convolution_layer::*;
pub use dense::*;
pub use flatten::*;
use helper_function::*;
pub use layer_weight::*;
pub use pooling_layer::*;
pub use recurrent_layer::*;
pub use regularization_layer::*;
pub use serialize_weight::*;
