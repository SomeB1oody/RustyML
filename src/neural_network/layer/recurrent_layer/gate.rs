use super::*;

/// Gate parameters and optimizer state for recurrent cells.
///
/// Stores weights, gradients, and optimizer cache for a single GRU/LSTM gate.
///
/// # Fields
///
/// - `kernel` - Weight matrix for input connections with shape (input_dim, units)
/// - `recurrent_kernel` - Weight matrix for recurrent connections with shape (units, units)
/// - `bias` - Bias vector with shape (1, units)
/// - `grad_kernel` - Optional gradient for input weights, accumulated during backpropagation
/// - `grad_recurrent_kernel` - Optional gradient for recurrent weights, accumulated during backpropagation
/// - `grad_bias` - Optional gradient for bias terms, accumulated during backpropagation
/// - `optimizer_cache` - Cache for storing optimizer-specific state (Adam, RMSprop momentum, etc.)
pub struct Gate {
    pub kernel: Array2<f32>,
    pub recurrent_kernel: Array2<f32>,
    pub bias: Array2<f32>,
    pub grad_kernel: Option<Array2<f32>>,
    pub grad_recurrent_kernel: Option<Array2<f32>>,
    pub grad_bias: Option<Array2<f32>>,
    pub optimizer_cache: OptimizerCache,
}

impl Gate {
    /// Creates a gate with randomly initialized weights.
    ///
    /// Uses Xavier/Glorot initialization for the input kernel, a normalized orthogonal
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

        // Orthogonal initialization for recurrent kernel
        let mut recurrent_kernel = Array::random((units, units), Uniform::new(-1.0, 1.0).unwrap());
        if units > 0 {
            // Simplified orthogonalization using QR decomposition approximation
            // For better numerical stability, normalize each column
            for mut col in recurrent_kernel.columns_mut() {
                let norm = col.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    col /= norm;
                }
            }
        }

        // Bias initialization
        let bias = Array::from_elem((1, units), bias_init_value);

        Ok(Self {
            kernel,
            recurrent_kernel,
            bias,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
            optimizer_cache: OptimizerCache::default(),
        })
    }
}

/// Gradient clipping value to prevent exploding gradients in recurrent layers
const GRADIENT_CLIP_VALUE: f32 = 5.0;

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
    gate.grad_kernel = Some(grad_kernel);
    gate.grad_recurrent_kernel = Some(grad_recurrent);
    gate.grad_bias = Some(grad_bias);
}

/// Updates a single gate's parameters using SGD with gradient clipping
///
/// # Parameters
///
/// - `gate` - The gate to update
/// - `lr` - Learning rate
#[inline]
pub fn update_gate_sgd(gate: &mut Gate, lr: f32) {
    if let (Some(gk), Some(grk), Some(gb)) = (
        &gate.grad_kernel,
        &gate.grad_recurrent_kernel,
        &gate.grad_bias,
    ) {
        // Apply gradient clipping to prevent exploding gradients
        let gk_clipped = gk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let grk_clipped = grk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let gb_clipped = gb.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));

        gate.kernel = &gate.kernel - &(lr * &gk_clipped);
        gate.recurrent_kernel = &gate.recurrent_kernel - &(lr * &grk_clipped);
        gate.bias = &gate.bias - &(lr * &gb_clipped);
    }
}

/// Updates a single gate's parameters using Adam optimizer with gradient clipping
///
/// # Parameters
///
/// - `gate` - The gate to update
/// - `input_dim` - Input dimensionality (for initializing optimizer state if needed)
/// - `units` - Number of units (for initializing optimizer state if needed)
/// - `lr` - Learning rate
/// - `beta1` - Exponential decay rate for first moment estimates
/// - `beta2` - Exponential decay rate for second moment estimates
/// - `epsilon` - Small constant for numerical stability
/// - `t` - Current timestep (for bias correction)
#[inline]
pub fn update_gate_adam(
    gate: &mut Gate,
    input_dim: usize,
    units: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u64,
) {
    // Initialize Adam states if needed
    if gate.optimizer_cache.adam_states.is_none() {
        gate.optimizer_cache.adam_states = Some(AdamStates::new(
            (input_dim, units),
            Some((units, units)),
            (1, units),
        ));
    }

    if let (Some(gk), Some(grk), Some(gb)) = (
        &gate.grad_kernel,
        &gate.grad_recurrent_kernel,
        &gate.grad_bias,
    ) {
        // Apply gradient clipping before Adam update
        let gk_clipped = gk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let grk_clipped = grk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let gb_clipped = gb.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));

        let adam_states = gate.optimizer_cache.adam_states.as_mut().unwrap();
        let (k_update, rk_update, b_update) = adam_states.update_parameter(
            &gk_clipped,
            Some(&grk_clipped),
            &gb_clipped,
            beta1,
            beta2,
            epsilon,
            t,
            lr,
        );

        gate.kernel = &gate.kernel - &k_update;
        gate.recurrent_kernel = &gate.recurrent_kernel - &rk_update.unwrap();
        gate.bias = &gate.bias - &b_update;
    }
}

/// Updates a single gate's parameters using RMSprop optimizer with gradient clipping
///
/// # Parameters
///
/// - `gate` - The gate to update
/// - `input_dim` - Input dimensionality (for initializing optimizer state if needed)
/// - `units` - Number of units (for initializing optimizer state if needed)
/// - `lr` - Learning rate
/// - `rho` - Decay rate for moving average of squared gradients
/// - `epsilon` - Small constant for numerical stability
#[inline]
pub fn update_gate_rmsprop(
    gate: &mut Gate,
    input_dim: usize,
    units: usize,
    lr: f32,
    rho: f32,
    epsilon: f32,
) {
    // Initialize RMSprop cache if needed
    if gate.optimizer_cache.rmsprop_cache.is_none() {
        gate.optimizer_cache.rmsprop_cache = Some(RMSpropCache::new(
            (input_dim, units),
            Some((units, units)),
            (1, units),
        ));
    }

    if let (Some(gk), Some(grk), Some(gb)) = (
        &gate.grad_kernel,
        &gate.grad_recurrent_kernel,
        &gate.grad_bias,
    ) {
        // Apply gradient clipping before RMSprop update
        let gk_clipped = gk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let grk_clipped = grk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let gb_clipped = gb.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));

        if let Some(ref mut cache) = gate.optimizer_cache.rmsprop_cache {
            cache.update_parameters(
                &mut gate.kernel,
                Some(&mut gate.recurrent_kernel),
                &mut gate.bias,
                &gk_clipped,
                Some(&grk_clipped),
                &gb_clipped,
                rho,
                lr,
                epsilon,
            );
        }
    }
}

/// Updates a single gate's parameters using AdaGrad optimizer with gradient clipping
///
/// # Parameters
///
/// - `gate` - The gate to update
/// - `input_dim` - Input dimensionality (for initializing optimizer state if needed)
/// - `units` - Number of units (for initializing optimizer state if needed)
/// - `lr` - Learning rate
/// - `epsilon` - Small constant for numerical stability
#[inline]
pub fn update_gate_ada_grad(
    gate: &mut Gate,
    input_dim: usize,
    units: usize,
    lr: f32,
    epsilon: f32,
) {
    // Initialize AdaGrad cache if needed
    if gate.optimizer_cache.ada_grad_cache.is_none() {
        gate.optimizer_cache.ada_grad_cache = Some(AdaGradStates::new(
            (input_dim, units),
            Some((units, units)),
            (1, units),
        ));
    }

    if let (Some(gk), Some(grk), Some(gb)) = (
        &gate.grad_kernel,
        &gate.grad_recurrent_kernel,
        &gate.grad_bias,
    ) {
        // Apply gradient clipping before AdaGrad update
        let gk_clipped = gk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let grk_clipped = grk.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));
        let gb_clipped = gb.mapv(|x| x.clamp(-GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE));

        let ada_grad_cache = gate.optimizer_cache.ada_grad_cache.as_mut().unwrap();
        let (k_update, rk_update, b_update) = ada_grad_cache.update_parameter(
            &gk_clipped,
            Some(&grk_clipped),
            &gb_clipped,
            epsilon,
            lr,
        );

        gate.kernel = &gate.kernel - &k_update;
        gate.recurrent_kernel = &gate.recurrent_kernel - &rk_update.unwrap();
        gate.bias = &gate.bias - &b_update;
    }
}
