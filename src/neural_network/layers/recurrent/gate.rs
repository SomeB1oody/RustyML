//! Fused gate parameters and shared helpers for GRU/LSTM recurrent cells
//!
//! Defines the `FusedGates` weight/gradient container plus the batched input-projection and
//! cache helpers reused across recurrent layers

use crate::error::Error;
use crate::neural_network::layers::recurrent::orthogonal_init;
use crate::neural_network::layers::recurrent::validation::validate_dimension_greater_than_zero;
use crate::neural_network::traits::ParamGrad;
use ndarray::{Array, Array2, Array3, ArrayView3, s};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Fused gate parameters and gradients for recurrent cells
///
/// Packs every gate's weights side by side into single matrices. Column blocks follow a fixed,
/// layer-defined gate order (LSTM: `[i | f | g | o]`, GRU: `[r | z | h]`), so the batched input
/// projection and the per-timestep recurrent projection each run as one large GEMM instead of one
/// GEMM per gate:
///
/// - `kernel`: `[input_dim, n_gates * units]`
/// - `recurrent_kernel`: `[units, n_gates * units]`
/// - `bias`: `[1, n_gates * units]`
///
/// Optimizer state is held by the optimizer, not the gates (see [`FusedGates::parameters`])
#[derive(Debug)]
pub struct FusedGates {
    /// Fused input kernel with shape (input_dim, n_gates * units)
    pub kernel: Array2<f32>,
    /// Fused recurrent kernel with shape (units, n_gates * units)
    pub recurrent_kernel: Array2<f32>,
    /// Fused bias with shape (1, n_gates * units)
    pub bias: Array2<f32>,
    /// Optional gradient for the fused input kernel, stored during backpropagation
    pub grad_kernel: Option<Array2<f32>>,
    /// Optional gradient for the fused recurrent kernel, stored during backpropagation
    pub grad_recurrent_kernel: Option<Array2<f32>>,
    /// Optional gradient for the fused bias, stored during backpropagation
    pub grad_bias: Option<Array2<f32>>,
}

impl FusedGates {
    /// Creates fused gates with randomly initialized weights
    ///
    /// Initialization keeps the per-gate semantics of separate gates exactly: the input kernel is
    /// Xavier/Glorot uniform with the **per-gate** fan (`input_dim + units`, not
    /// `input_dim + n_gates * units`), each gate's recurrent block is an independent Gram-Schmidt
    /// orthogonal `[units, units]` matrix, and each gate's bias block is a per-gate constant
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of the input features
    /// - `units` - Number of units (neurons) per gate
    /// - `bias_init` - Initial bias value per gate, in gate-block order; its length determines the
    ///   number of gates (e.g. `&[0.0, 1.0, 0.0, 0.0]` for an LSTM with forget bias 1.0)
    /// - `rng` - RNG threaded in by the owning layer so the whole layer shares one reproducible
    ///   stream. See [`crate::random`]
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - Fused gates with initialized parameters
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `input_dim` or `units` is 0
    pub fn new(
        input_dim: usize,
        units: usize,
        bias_init: &[f32],
        rng: &mut StdRng,
    ) -> Result<Self, Error> {
        validate_dimension_greater_than_zero(input_dim, "input_dim")?;
        validate_dimension_greater_than_zero(units, "units")?;

        let n_gates = bias_init.len();
        let width = n_gates * units;

        // Xavier/Glorot with the per-gate fan
        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let kernel = Array::random_using(
            (input_dim, width),
            Uniform::new(-limit, limit).unwrap(),
            rng,
        );

        // One orthogonal [units, units] block per gate
        let mut recurrent_kernel = Array2::<f32>::zeros((units, width));
        for g in 0..n_gates {
            recurrent_kernel
                .slice_mut(s![.., g * units..(g + 1) * units])
                .assign(&orthogonal_init(units, rng));
        }

        let mut bias = Array2::<f32>::zeros((1, width));
        for (g, &b) in bias_init.iter().enumerate() {
            bias.slice_mut(s![.., g * units..(g + 1) * units]).fill(b);
        }

        Ok(Self {
            kernel,
            recurrent_kernel,
            bias,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
        })
    }

    /// Exposes the 3 fused trainable tensors (kernel, recurrent kernel, bias) and their gradients
    /// as flat [`ParamGrad`] slices for the optimizer to update
    ///
    /// Returns an empty vector if gradients have not been computed yet
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
            params.push(ParamGrad::weight(
                kernel
                    .as_slice_mut()
                    .expect("fused kernel must be contiguous"),
                gk.as_slice()
                    .expect("fused kernel gradient must be contiguous"),
            ));
            params.push(ParamGrad::weight(
                recurrent_kernel
                    .as_slice_mut()
                    .expect("fused recurrent kernel must be contiguous"),
                grk.as_slice()
                    .expect("fused recurrent kernel gradient must be contiguous"),
            ));
            params.push(ParamGrad::no_decay(
                bias.as_slice_mut().expect("fused bias must be contiguous"),
                gb.as_slice()
                    .expect("fused bias gradient must be contiguous"),
            ));
        }
        params
    }

    /// Stores gradients for the fused tensors (replace semantics, matching Dense/SimpleRNN)
    ///
    /// # Parameters
    ///
    /// - `grad_kernel` - Gradient for the fused input kernel `[input_dim, n_gates * units]`
    /// - `grad_recurrent` - Gradient for the fused recurrent kernel `[units, n_gates * units]`
    /// - `grad_bias` - Gradient for the fused bias `[1, n_gates * units]`
    #[inline]
    pub fn store_gradients(
        &mut self,
        grad_kernel: Array2<f32>,
        grad_recurrent: Array2<f32>,
        grad_bias: Array2<f32>,
    ) {
        self.grad_kernel = Some(grad_kernel);
        self.grad_recurrent_kernel = Some(grad_recurrent);
        self.grad_bias = Some(grad_bias);
    }
}

/// Batched fused input projection: `x3 [batch, timesteps, input_dim] @ kernel` for every timestep
/// and every gate in a single GEMM, returning `[batch, timesteps, n_gates * units]`
///
/// The input projection does not depend on the recurrence, so collapsing the (batch, timesteps)
/// axes and all gate columns into one matmul replaces `timesteps * n_gates` small gemms with one
/// large one (better cache/SIMD use). Only the `h_prev @ recurrent_kernel` term has to stay
/// sequential
pub fn project_input(kernel: &Array2<f32>, x3: &ArrayView3<f32>) -> Array3<f32> {
    let (batch, timesteps, input_dim) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
    let width = kernel.shape()[1];
    let x2 = x3
        .to_shape((batch * timesteps, input_dim))
        .expect("contiguous [batch*timesteps, input_dim] reshape");
    reshape_2d_to_3d(
        crate::math::matmul::gemm_internal(&x2, kernel),
        (batch, timesteps, width),
    )
}

/// Reshapes a GEMM result `[d0*d1, d2]` to `[d0, d1, d2]`, tolerating a column-major input
///
/// `ndarray`'s `dot` returns a column-major result when both operands have a row stride of 1,
/// which arrays with a length-1 axis can exhibit despite passing `is_standard_layout` (e.g. a
/// `[1, w]` kernel built by `concatenate`). A direct `into_shape_with_order` would panic with
/// `IncompatibleLayout` on such a result, so normalize the layout first when needed
pub fn reshape_2d_to_3d(m: Array2<f32>, dims: (usize, usize, usize)) -> Array3<f32> {
    let m = if m.is_standard_layout() {
        m
    } else {
        m.as_standard_layout().into_owned()
    };
    m.into_shape_with_order(dims)
        .expect("row-major [d0*d1, d2] reshapes to [d0, d1, d2]")
}

/// Extracts the cached value, erroring if it is absent
///
/// Used during the backward pass to ensure the forward pass has been run
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NnError;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, Array3, array};

    // FusedGates::new

    /// Fused shapes follow [input_dim, n*units] / [units, n*units] / [1, n*units]
    #[test]
    fn fused_gates_new_shapes() {
        let mut rng = crate::random::make_rng(Some(42));
        let gates = FusedGates::new(4, 3, &[0.0, 1.0, 0.0, 0.0], &mut rng).unwrap();
        assert_eq!(gates.kernel.shape(), &[4, 12]);
        assert_eq!(gates.recurrent_kernel.shape(), &[3, 12]);
        assert_eq!(gates.bias.shape(), &[1, 12]);
    }

    /// Each gate's bias block is filled with its per-gate constant
    #[test]
    fn fused_gates_new_bias_blocks() {
        let mut rng = crate::random::make_rng(Some(42));
        let gates = FusedGates::new(2, 2, &[0.0, 1.0, 0.5], &mut rng).unwrap();
        let b = &gates.bias;
        // Block 0 (cols 0..2) = 0.0, block 1 (cols 2..4) = 1.0, block 2 (cols 4..6) = 0.5
        for c in 0..2 {
            assert_abs_diff_eq!(b[[0, c]], 0.0_f32, epsilon = 1e-6);
        }
        for c in 2..4 {
            assert_abs_diff_eq!(b[[0, c]], 1.0_f32, epsilon = 1e-6);
        }
        for c in 4..6 {
            assert_abs_diff_eq!(b[[0, c]], 0.5_f32, epsilon = 1e-6);
        }
    }

    /// Each gate's recurrent block is orthogonal (R^T R = I per block), not the fused matrix
    /// as a whole
    #[test]
    fn fused_gates_new_recurrent_blocks_orthogonal() {
        let mut rng = crate::random::make_rng(Some(7));
        let units = 3;
        let gates = FusedGates::new(2, units, &[0.0, 0.0], &mut rng).unwrap();
        for g in 0..2 {
            let block = gates
                .recurrent_kernel
                .slice(s![.., g * units..(g + 1) * units]);
            let gram = block.t().dot(&block);
            for i in 0..units {
                for j in 0..units {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_abs_diff_eq!(gram[[i, j]], expected, epsilon = 1e-4);
                }
            }
        }
    }

    /// Zero dimensions are rejected
    #[test]
    fn fused_gates_new_rejects_zero_dims() {
        let mut rng = crate::random::make_rng(Some(1));
        assert!(FusedGates::new(0, 3, &[0.0], &mut rng).is_err());
        assert!(FusedGates::new(3, 0, &[0.0], &mut rng).is_err());
    }

    // FusedGates::parameters (empty branch)

    /// Freshly-constructed gates have all gradients `None`, so `parameters()` is empty
    #[test]
    fn parameters_empty_when_gradients_none() {
        let mut rng = crate::random::make_rng(Some(3));
        let mut gates = FusedGates::new(2, 2, &[0.0, 0.0], &mut rng).unwrap();
        assert!(
            gates.parameters().is_empty(),
            "parameters() must be empty before gradients are computed"
        );
    }

    /// After storing gradients, `parameters()` exposes the 3 fused tensors
    #[test]
    fn parameters_three_entries_after_gradients() {
        let mut rng = crate::random::make_rng(Some(3));
        let mut gates = FusedGates::new(2, 2, &[0.0, 0.0], &mut rng).unwrap();
        gates.store_gradients(
            Array2::zeros((2, 4)),
            Array2::zeros((2, 4)),
            Array2::zeros((1, 4)),
        );
        assert_eq!(gates.parameters().len(), 3);
    }

    // store_gradients (replace semantics, no clipping)

    /// `store_gradients` stores gradients exactly as computed, with no clipping
    #[test]
    fn store_gradients_stores_unchanged() {
        let mut rng = crate::random::make_rng(Some(5));
        let mut gates = FusedGates::new(2, 1, &[0.0, 0.0], &mut rng).unwrap();

        // Large values must be preserved verbatim (no clipping)
        let grad_kernel = array![[10.0_f32, -100.0], [3.0, 42.0]];
        let grad_recurrent = array![[5.0_f32, -5.0]];
        let grad_bias = array![[8.0_f32, -8.0]];

        gates.store_gradients(
            grad_kernel.clone(),
            grad_recurrent.clone(),
            grad_bias.clone(),
        );

        for (got, exp) in gates
            .grad_kernel
            .as_ref()
            .unwrap()
            .iter()
            .zip(grad_kernel.iter())
        {
            assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
        }
        for (got, exp) in gates
            .grad_recurrent_kernel
            .as_ref()
            .unwrap()
            .iter()
            .zip(grad_recurrent.iter())
        {
            assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
        }
        for (got, exp) in gates
            .grad_bias
            .as_ref()
            .unwrap()
            .iter()
            .zip(grad_bias.iter())
        {
            assert_abs_diff_eq!(got, exp, epsilon = 1e-6);
        }
    }

    // project_input

    /// The fused batched projection equals x_t @ kernel computed per timestep
    #[test]
    fn project_input_matches_per_timestep_gemm() {
        // x3: [batch=1, timesteps=2, input_dim=2]
        let x3 = Array3::from_shape_vec((1, 2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        // kernel: [2, 4] (2 gates, 2 units each)
        let kernel = array![[1.0_f32, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]];

        let out = project_input(&kernel, &x3.view());
        assert_eq!(out.shape(), &[1, 2, 4]);

        // t=0: [1, 2] @ kernel = [1, 2, 2, 4]
        assert_abs_diff_eq!(out[[0, 0, 0]], 1.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 0, 1]], 2.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 0, 2]], 2.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 0, 3]], 4.0_f32, epsilon = 1e-6);
        // t=1: [3, 4] @ kernel = [3, 4, 6, 8]
        assert_abs_diff_eq!(out[[0, 1, 0]], 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 1, 1]], 4.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 1, 2]], 6.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 1, 3]], 8.0_f32, epsilon = 1e-6);
    }

    // take_cache

    /// `take_cache` maps a `None` cache to `ForwardPassNotRun` carrying the layer name verbatim
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

    /// When the cache is populated, `take_cache` returns the inner value and leaves `None` behind
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
