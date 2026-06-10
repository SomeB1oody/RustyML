use crate::error::Error;
use crate::neural_network::layers::recurrent::validation::validate_dimension_greater_than_zero;
use crate::neural_network::layers::recurrent::{GRADIENT_CLIP_VALUE, orthogonal_init};
use crate::neural_network::traits::ParamGrad;
use ndarray::{Array, Array2};
use ndarray_rand::rand::rngs::StdRng;
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
#[derive(Debug)]
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
    /// - `rng` - RNG threaded in by the owning layer so all of its gates share one reproducible
    ///   stream. See [`crate::random`].
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new gate instance with initialized parameters
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `input_dim` or `units` is 0
    pub fn new(
        input_dim: usize,
        units: usize,
        bias_init_value: f32,
        rng: &mut StdRng,
    ) -> Result<Self, Error> {
        // Validate dimensions
        validate_dimension_greater_than_zero(input_dim, "input_dim")?;
        validate_dimension_greater_than_zero(units, "units")?;

        // Xavier/Glorot initialization for input kernel
        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random_using(
            (input_dim, units),
            Uniform::new(-limit, limit).unwrap(),
            rng,
        );

        // Orthogonal initialization for the recurrent kernel (true Gram-Schmidt, shared with
        // SimpleRNN), which keeps the recurrence norm-preserving across time steps.
        let recurrent_kernel = orthogonal_init(units, rng);

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
/// - `layer` - Name of the layer, used to build the error when the cache is empty
///
/// # Returns
///
/// - `crate::error::RustymlResult<T>` - The cached value if present
///
/// # Errors
///
/// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If the cache is empty
#[inline]
pub fn take_cache<T>(cache: &mut Option<T>, layer: &'static str) -> crate::error::RustymlResult<T> {
    cache
        .take()
        .ok_or_else(|| Error::forward_pass_not_run(layer))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NnError;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    /// Builds a Gate manually with known weights so the test is fully deterministic:
    ///   kernel          = I_2  (2×2 identity)
    ///   recurrent_kernel = 0_2  (2×2 zeros)
    ///   bias            = [[1.0, 1.0]]
    ///
    /// With x_t = [[2.0, 3.0]] and h_prev = [[0.0, 0.0]] the formula
    ///   out = x_t @ kernel + h_prev @ recurrent_kernel + bias
    ///       = [[2,3]] @ I + [[0,0]] @ 0 + [[1,1]]
    ///       = [[2,3]] + [[0,0]] + [[1,1]]
    ///       = [[3, 4]]
    #[test]
    fn compute_gate_value_identity_kernel_zero_recurrent() {
        let gate = Gate {
            kernel: array![[1.0_f32, 0.0], [0.0, 1.0]],
            recurrent_kernel: Array2::<f32>::zeros((2, 2)),
            bias: array![[1.0_f32, 1.0]],
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
        };

        let x_t: Array2<f32> = array![[2.0, 3.0]];
        let h_prev: Array2<f32> = array![[0.0, 0.0]];

        let output = compute_gate_value(&gate, &x_t, &h_prev);

        assert_eq!(output.shape(), &[1, 2]);
        // Expected: [[3.0, 4.0]]
        assert_abs_diff_eq!(output[[0, 0]], 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[0, 1]], 4.0_f32, epsilon = 1e-6);
    }

    /// When the recurrent kernel is non-zero, the h_prev contribution must also
    /// be included.  With kernel = I, recurrent_kernel = I, bias = 0:
    ///   out = [[2,3]] @ I + [[1,1]] @ I + [[0,0]]
    ///       = [[2,3]] + [[1,1]] + [[0,0]]
    ///       = [[3, 4]]
    #[test]
    fn compute_gate_value_identity_recurrent_kernel_adds_h_prev() {
        let gate = Gate {
            kernel: array![[1.0_f32, 0.0], [0.0, 1.0]],
            recurrent_kernel: array![[1.0_f32, 0.0], [0.0, 1.0]],
            bias: Array2::<f32>::zeros((1, 2)),
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
        };

        let x_t: Array2<f32> = array![[2.0, 3.0]];
        let h_prev: Array2<f32> = array![[1.0, 1.0]];

        let output = compute_gate_value(&gate, &x_t, &h_prev);

        assert_eq!(output.shape(), &[1, 2]);
        // Expected: [[3.0, 4.0]]
        assert_abs_diff_eq!(output[[0, 0]], 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[0, 1]], 4.0_f32, epsilon = 1e-6);
    }
    // ── Gate::parameters (empty branch) ─────────────────────────────────────────

    /// A freshly-constructed Gate has all three gradients set to `None`, so
    /// `parameters()` must return an EMPTY vector (documented contract: "Returns an
    /// empty vector if gradients have not been computed yet").  The implementation
    /// only pushes ParamGrad entries inside `if let (Some, Some, Some) = (..grads..)`,
    /// which cannot fire when every gradient is `None`.
    #[test]
    fn parameters_empty_when_gradients_none() {
        let mut gate = Gate {
            kernel: array![[1.0_f32, 0.0], [0.0, 1.0]],
            recurrent_kernel: Array2::<f32>::zeros((2, 2)),
            bias: array![[0.0_f32, 0.0]],
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
        };

        let params = gate.parameters();
        assert!(
            params.is_empty(),
            "parameters() must be empty before gradients are computed, got {} entries",
            params.len()
        );
    }

    // ── store_gate_gradients (clipping) ─────────────────────────────────────────

    /// `store_gate_gradients` clamps every stored gradient to
    /// [-GRADIENT_CLIP_VALUE, +GRADIENT_CLIP_VALUE] = [-5.0, 5.0].  By definition of
    /// `f32::clamp`:  10 -> 5, -7 -> -5, 3 -> 3 (in range, unchanged), -100 -> -5,
    /// 5 -> 5 (boundary), -5 -> -5 (boundary).  This proves both that out-of-range
    /// magnitudes are pulled to the ±5 bounds and that in-range values pass through.
    #[test]
    fn store_gate_gradients_clips_to_pm_five() {
        let mut gate = Gate {
            kernel: Array2::<f32>::zeros((2, 2)),
            recurrent_kernel: Array2::<f32>::zeros((2, 2)),
            bias: Array2::<f32>::zeros((1, 2)),
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
        };

        // grad_kernel: 10 -> 5, -7 -> -5, 3 -> 3 (kept), -100 -> -5
        let grad_kernel = array![[10.0_f32, -7.0], [3.0, -100.0]];
        // grad_recurrent: 5 -> 5 (boundary), -5 -> -5 (boundary), 0 -> 0, 6 -> 5
        let grad_recurrent = array![[5.0_f32, -5.0], [0.0, 6.0]];
        // grad_bias: 8 -> 5, -8 -> -5
        let grad_bias = array![[8.0_f32, -8.0]];

        store_gate_gradients(&mut gate, grad_kernel, grad_recurrent, grad_bias);

        let gk = gate.grad_kernel.as_ref().expect("grad_kernel stored");
        assert_abs_diff_eq!(gk[[0, 0]], 5.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(gk[[0, 1]], -5.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(gk[[1, 0]], 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(gk[[1, 1]], -5.0_f32, epsilon = 1e-6);

        let grk = gate
            .grad_recurrent_kernel
            .as_ref()
            .expect("grad_recurrent_kernel stored");
        assert_abs_diff_eq!(grk[[0, 0]], 5.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(grk[[0, 1]], -5.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(grk[[1, 0]], 0.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(grk[[1, 1]], 5.0_f32, epsilon = 1e-6);

        let gb = gate.grad_bias.as_ref().expect("grad_bias stored");
        assert_abs_diff_eq!(gb[[0, 0]], 5.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(gb[[0, 1]], -5.0_f32, epsilon = 1e-6);

        // Every stored gradient must lie within the clip bounds.
        for g in [gk, grk, gb] {
            for &v in g.iter() {
                assert!(
                    (-GRADIENT_CLIP_VALUE..=GRADIENT_CLIP_VALUE).contains(&v),
                    "stored gradient {v} outside [-{GRADIENT_CLIP_VALUE}, {GRADIENT_CLIP_VALUE}]"
                );
            }
        }
    }

    // ── take_cache (None -> ForwardPassNotRun) ──────────────────────────────────

    /// `take_cache` is the guard used on the backward path: when the forward pass has
    /// not run, the cache is `None` and `take_cache` must map that `None` to
    /// `Error::NeuralNetwork(NnError::ForwardPassNotRun(layer))` carrying the layer
    /// name verbatim (per `Error::forward_pass_not_run`).
    #[test]
    fn take_cache_none_returns_forward_pass_not_run() {
        let mut cache: Option<Array2<f32>> = None;
        let err = take_cache(&mut cache, "Gate").unwrap_err();
        assert!(
            matches!(
                err,
                Error::NeuralNetwork(NnError::ForwardPassNotRun("Gate"))
            ),
            "expected ForwardPassNotRun(\"Gate\"), got: {err:?}"
        );
    }

    /// When the cache is populated, `take_cache` returns the inner value and leaves
    /// `None` behind (it takes ownership), so a subsequent call would error.  This
    /// pins the success/`take` semantics that the None-branch test complements.
    #[test]
    fn take_cache_some_returns_value_and_empties() {
        let mut cache: Option<Array2<f32>> = Some(array![[7.0_f32, 8.0]]);
        let value = take_cache(&mut cache, "Gate").expect("value present");
        assert_eq!(value.shape(), &[1, 2]);
        assert_abs_diff_eq!(value[[0, 0]], 7.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(value[[0, 1]], 8.0_f32, epsilon = 1e-6);
        assert!(cache.is_none(), "take_cache must leave None behind");
    }
}
