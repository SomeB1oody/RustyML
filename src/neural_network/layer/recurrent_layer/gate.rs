use crate::error::ModelError;
use crate::neural_network::layer::recurrent_layer::validation::validate_dimension_greater_than_zero;
use crate::neural_network::layer::recurrent_layer::{GRADIENT_CLIP_VALUE, orthogonal_init};
use crate::neural_network::neural_network_trait::ParamGrad;
use ndarray::{Array, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Gate parameters and gradients for recurrent cells.
///
/// Stores weights and gradients for a single GRU/LSTM gate. Optimizer state is held by the
/// optimizer, not the gate (see [`Gate::parameters`]).
///
/// # Fields
///
/// - `kernel` - Weight matrix for input connections with shape (input_dim, units)
/// - `recurrent_kernel` - Weight matrix for recurrent connections with shape (units, units)
/// - `bias` - Bias vector with shape (1, units)
/// - `grad_kernel` - Optional gradient for input weights, accumulated during backpropagation
/// - `grad_recurrent_kernel` - Optional gradient for recurrent weights, accumulated during backpropagation
/// - `grad_bias` - Optional gradient for bias terms, accumulated during backpropagation
pub struct Gate {
    pub kernel: Array2<f32>,
    pub recurrent_kernel: Array2<f32>,
    pub bias: Array2<f32>,
    pub grad_kernel: Option<Array2<f32>>,
    pub grad_recurrent_kernel: Option<Array2<f32>>,
    pub grad_bias: Option<Array2<f32>>,
}

impl Gate {
    /// Creates a gate with randomly initialized weights.
    ///
    /// Uses Xavier/Glorot initialization for the input kernel, Gram-Schmidt orthogonal
    /// initialization for the recurrent kernel, and a constant bias value.
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of the input features
    /// - `units` - Number of units (neurons) in this gate
    /// - `bias_init_value` - Initial value for bias (0.0 for most gates, 1.0 for LSTM forget gate)
    ///
    /// # Returns
    ///
    /// - `Result<Self, ModelError>` - A new gate instance with initialized parameters
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - If `input_dim` or `units` is 0
    pub fn new(input_dim: usize, units: usize, bias_init_value: f32) -> Result<Self, ModelError> {
        // Validate dimensions
        validate_dimension_greater_than_zero(input_dim, "input_dim")?;
        validate_dimension_greater_than_zero(units, "units")?;

        // Xavier/Glorot initialization for input kernel
        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random((input_dim, units), Uniform::new(-limit, limit).unwrap());

        // Orthogonal initialization for the recurrent kernel (true Gram-Schmidt, shared with
        // SimpleRNN), which keeps the recurrence norm-preserving across time steps.
        let recurrent_kernel = orthogonal_init(units);

        // Bias initialization
        let bias = Array::from_elem((1, units), bias_init_value);

        Ok(Self {
            kernel,
            recurrent_kernel,
            bias,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
        })
    }

    /// Exposes this gate's three trainable tensors (kernel, recurrent kernel, bias) and their
    /// gradients as flat [`ParamGrad`] slices, for the optimizer to update. Returns an empty
    /// vector if gradients have not been computed yet.
    pub fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            kernel,
            recurrent_kernel,
            bias,
            grad_kernel,
            grad_recurrent_kernel,
            grad_bias,
            ..
        } = self;
        let mut params = Vec::new();
        if let (Some(gk), Some(grk), Some(gb)) = (
            grad_kernel.as_ref(),
            grad_recurrent_kernel.as_ref(),
            grad_bias.as_ref(),
        ) {
            params.push(ParamGrad {
                value: kernel
                    .as_slice_mut()
                    .expect("gate kernel must be contiguous"),
                grad: gk
                    .as_slice()
                    .expect("gate kernel gradient must be contiguous"),
            });
            params.push(ParamGrad {
                value: recurrent_kernel
                    .as_slice_mut()
                    .expect("gate recurrent kernel must be contiguous"),
                grad: grk
                    .as_slice()
                    .expect("gate recurrent kernel gradient must be contiguous"),
            });
            params.push(ParamGrad {
                value: bias.as_slice_mut().expect("gate bias must be contiguous"),
                grad: gb
                    .as_slice()
                    .expect("gate bias gradient must be contiguous"),
            });
        }
        params
    }
}

/// Computes gate value: x_t @ kernel + h_prev @ recurrent_kernel + bias
///
/// This is the standard computation used by all gates in GRU and LSTM.
///
/// # Parameters
///
/// - `gate` - Gate parameters used for the computation
/// - `x_t` - Input at the current timestep with shape (batch, input_dim)
/// - `h_prev` - Previous hidden state with shape (batch, units)
///
/// # Returns
///
/// - `Array2<f32>` - Pre-activation gate values with shape (batch, units)
///
/// # Panics
///
/// - If matrix dimensions are incompatible for multiplication
#[inline]
pub fn compute_gate_value(gate: &Gate, x_t: &Array2<f32>, h_prev: &Array2<f32>) -> Array2<f32> {
    x_t.dot(&gate.kernel) + h_prev.dot(&gate.recurrent_kernel) + &gate.bias
}

/// Helper function to extract cache and return error if not available
///
/// This is used during backward pass to ensure forward pass has been run.
///
/// # Parameters
///
/// - `cache` - Cache container to take ownership from
/// - `error_msg` - Error message to use when cache is empty
///
/// # Returns
///
/// - `Result<T, ModelError>` - The cached value if present
///
/// # Errors
///
/// - `ModelError::ProcessingError` - If the cache is empty
#[inline]
pub fn take_cache<T>(cache: &mut Option<T>, error_msg: &str) -> Result<T, ModelError> {
    cache
        .take()
        .ok_or_else(|| ModelError::ProcessingError(error_msg.to_string()))
}

/// Stores gradients for a gate
///
/// This is a helper function to reduce code duplication when storing gradients
/// during backpropagation.
///
/// # Parameters
///
/// - `gate` - Gate to store gradients into
/// - `grad_kernel` - Gradient for input kernel weights
/// - `grad_recurrent` - Gradient for recurrent kernel weights
/// - `grad_bias` - Gradient for bias values
#[inline]
pub fn store_gate_gradients(
    gate: &mut Gate,
    grad_kernel: Array2<f32>,
    grad_recurrent: Array2<f32>,
    grad_bias: Array2<f32>,
) {
    // Clip gradients to curb exploding gradients in recurrent layers (matches SimpleRNN). Clipping
    // here, at gradient storage, keeps the optimizer update uniform across all layer types.
    let clip = |g: Array2<f32>| g.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
    gate.grad_kernel = Some(clip(grad_kernel));
    gate.grad_recurrent_kernel = Some(clip(grad_recurrent));
    gate.grad_bias = Some(clip(grad_bias));
}
