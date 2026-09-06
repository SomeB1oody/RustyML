//! Fused gate parameters and shared helpers for the GRU and LSTM recurrent cells
//!
//! Defines the `FusedGates` weight container. Provides the batched input-projection and reshape
//! helpers shared by all 3 recurrent layers.

use crate::error::Error;
use crate::neural_network::layers::recurrent::validation::validate_dimension_greater_than_zero;
use crate::neural_network::traits::ParamRef;
use crate::neural_network::{Fans, Initializer};
use ndarray::{Array2, Array3, ArrayView3, s};
use ndarray_rand::rand::rngs::StdRng;

/// Fused gate parameters for recurrent cells
///
/// Packs every gate's weights side by side into single matrices. Column blocks follow a fixed,
/// layer-defined gate order: LSTM uses `[i | f | g | o]` and GRU uses `[z | r | h]`. This lets
/// the batched input projection run as 1 large GEMM for every gate, instead of 1 GEMM per gate.
///
/// The per-timestep recurrent projection benefits the same way when a layer's gates do not
/// depend on each other. GRU's candidate gate depends on the reset gate's output, so GRU still
/// needs 2 recurrent GEMMs per timestep instead of 1.
///
/// The optimizer holds its own state, not the gates. See [`FusedGates::parameters_mut`].
#[derive(Debug)]
pub struct FusedGates {
    /// Fused input kernel with shape (input_dim, n_gates * units)
    pub kernel: Array2<f32>,
    /// Fused recurrent kernel with shape (units, n_gates * units)
    pub recurrent_kernel: Array2<f32>,
    /// Fused bias with shape (1, n_gates * units)
    pub bias: Array2<f32>,
}

impl FusedGates {
    /// Fused gates that hold no weight at all
    ///
    /// A layer holds this until [`UnaryLayer::build`](crate::neural_network::traits::UnaryLayer::build)
    /// reads the feature count from the input shape and replaces it with a drawn set
    ///
    /// # Returns
    ///
    /// - `FusedGates` - Gates whose 3 matrices are empty
    pub fn empty() -> Self {
        Self {
            kernel: Array2::zeros((0, 0)),
            recurrent_kernel: Array2::zeros((0, 0)),
            bias: Array2::zeros((0, 0)),
        }
    }

    /// Creates fused gates with randomly initialized weights
    ///
    /// Initialization keeps the per-gate semantics of separate gates. The input kernel uses
    /// Xavier/Glorot uniform with the **per-gate** fan, `input_dim + units`, not
    /// `input_dim + n_gates * units`. Each gate's recurrent block is an independent Gram-Schmidt
    /// orthogonal `[units, units]` matrix, and each gate's bias block is a per-gate constant
    ///
    /// The draw order is part of the contract: the fused input kernel first, then 1 orthogonal
    /// block per gate in gate order, all against the 1 generator the caller supplies. A second
    /// generator, or a different order, changes every value from the second draw onward
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of the input features
    /// - `units` - Number of units (neurons) per gate
    /// - `bias_init` - Initial bias value per gate, in gate-block order. Its length sets the
    ///   number of gates (e.g. `&[0.0, 1.0, 0.0, 0.0]` for an LSTM with a forget bias of 1.0)
    /// - `rng` - Shared RNG from the owning layer, so the whole layer draws from a single
    ///   reproducible stream. See [`crate::random`]
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

        // The fan pair is the per-gate pair, not the fused width
        let kernel =
            Initializer::GlorotUniform.draw((input_dim, width), Fans::new(input_dim, units), rng);

        // A single orthogonal [units, units] block per gate
        let mut recurrent_kernel = Array2::<f32>::zeros((units, width));
        for g in 0..n_gates {
            recurrent_kernel
                .slice_mut(s![.., g * units..(g + 1) * units])
                .assign(&Initializer::Orthogonal.draw_orthogonal(units, Fans::NONE, rng));
        }

        let mut bias = Array2::<f32>::zeros((1, width));
        for (g, &b) in bias_init.iter().enumerate() {
            bias.slice_mut(s![.., g * units..(g + 1) * units]).fill(b);
        }

        Ok(Self {
            kernel,
            recurrent_kernel,
            bias,
        })
    }

    /// Number of units per gate
    ///
    /// The recurrent kernel is `[units, gate count * units]`, so its first axis is the unit
    /// count. A cell reads the count from here, and holds no second copy of it
    ///
    /// # Returns
    ///
    /// - `usize` - The unit count of 1 gate
    pub fn units(&self) -> usize {
        self.recurrent_kernel.shape()[0]
    }

    /// Exposes the 3 fused trainable tensors (kernel, recurrent kernel, bias) as flat
    /// [`ParamRef`] slices for the optimizer to update
    ///
    /// The gates hold no gradient. A backward pass adds every gradient to the store of the
    /// context, under these same 3 names
    ///
    /// # Returns
    ///
    /// - `Vec<ParamRef<'_>>` - The 3 tensors, in kernel, recurrent kernel, bias order
    pub fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let Self {
            kernel,
            recurrent_kernel,
            bias,
        } = self;
        vec![
            ParamRef::weight(
                "kernel",
                kernel
                    .as_slice_mut()
                    .expect("fused kernel must be contiguous"),
            ),
            ParamRef::weight(
                "recurrent_kernel",
                recurrent_kernel
                    .as_slice_mut()
                    .expect("fused recurrent kernel must be contiguous"),
            ),
            ParamRef::no_decay(
                "bias",
                bias.as_slice_mut().expect("fused bias must be contiguous"),
            ),
        ]
    }
}

/// Batched fused input projection: `x3 @ kernel` for every timestep and every gate in a single
/// GEMM
///
/// The input projection does not depend on the recurrence. Collapsing the (batch, timesteps) axes
/// and all gate columns into 1 matmul replaces `timesteps * n_gates` small GEMMs with 1 large
/// GEMM. This improves cache and SIMD use. Only the `h_prev @ recurrent_kernel` term must stay
/// sequential.
///
/// # Parameters
///
/// - `kernel` - Fused input kernel with shape `[input_dim, n_gates * units]`
/// - `x3` - Input tensor with shape `[batch, timesteps, input_dim]`
///
/// # Returns
///
/// - `Array3<f32>` - The projected input with shape `[batch, timesteps, n_gates * units]`
pub fn project_input(kernel: &Array2<f32>, x3: &ArrayView3<f32>) -> Array3<f32> {
    let (batch, timesteps, input_dim) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
    let width = kernel.shape()[1];
    let x2 = x3
        .to_shape((batch * timesteps, input_dim))
        .expect("contiguous [batch*timesteps, input_dim] reshape");
    reshape_2d_to_3d(gemmkit_ndarray::dot(&x2, kernel), (batch, timesteps, width))
}

/// Reshapes a GEMM result `[d0*d1, d2]` to `[d0, d1, d2]`, and tolerates a column-major input
///
/// `ndarray`'s `dot` returns a column-major result when both operands have a row stride of 1.
/// Arrays with a length-1 axis can have this stride even when `is_standard_layout` returns true,
/// for example a `[1, w]` kernel built by `concatenate`. A direct `into_shape_with_order` call
/// would then panic with `IncompatibleLayout`, so this function normalizes the layout first when
/// needed.
///
/// # Parameters
///
/// - `m` - GEMM result with shape `[d0*d1, d2]`, in either row-major or column-major layout
/// - `dims` - Target shape `(d0, d1, d2)`
///
/// # Returns
///
/// - `Array3<f32>` - `m` reshaped to `dims`
pub fn reshape_2d_to_3d(m: Array2<f32>, dims: (usize, usize, usize)) -> Array3<f32> {
    let m = if m.is_standard_layout() {
        m
    } else {
        m.as_standard_layout().into_owned()
    };
    m.into_shape_with_order(dims)
        .expect("row-major [d0*d1, d2] reshapes to [d0, d1, d2]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array3, array};

    // FusedGates::new

    /// Fused shapes are `[input_dim, w]`, `[units, w]`, and `[1, w]`, where `w = n_gates * units`
    #[test]
    fn fused_gates_new_shapes() {
        let mut rng = crate::random::make_rng(Some(42));
        let gates = FusedGates::new(4, 3, &[0.0, 1.0, 0.0, 0.0], &mut rng).unwrap();
        assert_eq!(gates.kernel.shape(), &[4, 12]);
        assert_eq!(gates.recurrent_kernel.shape(), &[3, 12]);
        assert_eq!(gates.bias.shape(), &[1, 12]);
    }

    /// `FusedGates::new` fills each gate's bias block with its per-gate constant
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

    /// `FusedGates::new` rejects a dimension of 0
    #[test]
    fn fused_gates_new_rejects_zero_dims() {
        let mut rng = crate::random::make_rng(Some(1));
        assert!(FusedGates::new(0, 3, &[0.0], &mut rng).is_err());
        assert!(FusedGates::new(3, 0, &[0.0], &mut rng).is_err());
    }

    // FusedGates::parameters_mut

    /// `parameters_mut` exposes the 3 fused tensors under their fixed names, on every call
    #[test]
    fn parameters_mut_three_named_entries() {
        let mut rng = crate::random::make_rng(Some(3));
        let mut gates = FusedGates::new(2, 2, &[0.0, 0.0], &mut rng).unwrap();
        let params = gates.parameters_mut();
        assert_eq!(params.len(), 3);
        let names: Vec<&str> = params.iter().map(|param| param.name).collect();
        assert_eq!(names, vec!["kernel", "recurrent_kernel", "bias"]);
    }

    /// Decoupled weight decay applies to the 2 kernels, and it skips the bias
    #[test]
    fn parameters_mut_decay_flags() {
        let mut rng = crate::random::make_rng(Some(3));
        let mut gates = FusedGates::new(2, 2, &[0.0, 0.0], &mut rng).unwrap();
        let decays: Vec<bool> = gates
            .parameters_mut()
            .iter()
            .map(|param| param.decays)
            .collect();
        assert_eq!(decays, vec![true, true, false]);
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
}
