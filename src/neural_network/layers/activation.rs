//! Element-wise activation functions and the standalone activation layers
//!
//! Defines the [`Activation`] enum (the single source of truth for each
//! activation's forward transform and derivative) along with the thin
//! [`Linear`], [`ReLU`], [`Sigmoid`], [`Tanh`], and [`Softmax`] layer wrappers

use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::parallel_gates::{cheap_map_parallel_threshold, exp_map_parallel_threshold};
use crate::{Deserialize, Serialize};
use ndarray::{Array2, ArrayView1, ArrayViewMut1, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Formats a shape slice as a parenthesized tuple, e.g. `"(2, 3)"`
///
/// # Parameters
///
/// - `shape` - Dimension sizes to format
///
/// # Returns
///
/// - `String` - The shape rendered as `"(d0, d1, ...)"`
fn format_shape(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Formats the cached output shape for activation layers
///
/// # Parameters
///
/// - `cached_tensor` - The layer's cached output, if any
///
/// # Returns
///
/// - `String` - The cached tensor's shape, or `"Unknown"` if nothing is cached
fn format_output_shape(cached_tensor: &Option<Tensor>) -> String {
    match cached_tensor {
        Some(tensor) => format_shape(tensor.shape()),
        None => "Unknown".to_string(),
    }
}

/// Linear (Identity) activation layer
pub mod linear;
/// ReLU (Rectified Linear Unit) activation layer
pub mod relu;
/// Sigmoid activation layer
pub mod sigmoid;
/// Softmax activation layer
pub mod softmax;
/// Tanh (Hyperbolic Tangent) activation layer
pub mod tanh;

pub use linear::Linear;
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;

/// Epsilon guarding the softmax normalizer against division by zero
const SOFTMAX_EPSILON: f32 = 1e-8;

/// The element-wise activation functions that trainable layers can embed
///
/// Dense, the convolutional layers, and the recurrent layers each carry an
/// `Activation` value instead of a generic activation type parameter. A runtime
/// enum keeps the host layers non-generic, which removes monomorphization bloat
/// and lets weight deserialization downcast every layer to a single concrete type
/// rather than probing each `Layer<Act>` pairing
///
/// The standalone activation *layers* ([`Linear`], [`ReLU`], [`Sigmoid`], [`Tanh`],
/// [`Softmax`]) delegate their math here, so this enum is the single source of
/// truth for both the forward transform and its derivative
///
/// # Adding a new activation
///
/// Implement the math **here on the enum**, not in a layer:
/// 1. Add a variant to this enum
/// 2. Handle it in [`Activation::forward`] (the transform) and [`Activation::backward`] (the
///    derivative, expressed in terms of the *activated output* `a`, not the pre-activation `z`)
/// 3. Add a thin standalone layer struct mirroring [`ReLU`] - a `{ output_cache }` field whose
///    `Layer` impl validates, caches the output, and delegates to this enum - plus a
///    `From<NewLayer> for Activation` impl so it works as an `impl Into<Activation>` argument
///
/// The algorithm lives on the enum, not in the layer `impl`s, because the trainable layers
/// (Dense, the convolutional layers, the recurrent layers) store an `Activation` *value* and call
/// these **pure, stateless** methods inside their own forward/backward. A stateful `Layer` impl
/// (which caches `output` and takes `&mut self`) cannot serve that embedded, value-typed use
/// without reintroducing a generic activation type parameter or `Box<dyn Layer>`, and would
/// duplicate the algorithm. The standalone structs are therefore thin wrappers, never the source
/// of truth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Identity activation, `f(x) = x`
    Linear,
    /// Rectified linear unit, `max(0, x)`
    ReLU,
    /// Logistic sigmoid, `1 / (1 + e^-x)`
    Sigmoid,
    /// Hyperbolic tangent, `tanh(x)`
    Tanh,
    /// Softmax over the last axis (input must be at least 2D)
    Softmax,
}

impl Activation {
    /// Applies the activation to a pre-activation tensor `z`, returning the activated output
    ///
    /// # Parameters
    ///
    /// - `z` - Pre-activation tensor (the linear output of the host layer)
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - Activated tensor with the same shape as `z`
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - Softmax received an input with fewer than 2 dimensions
    /// - `Error::Computation` - Softmax failed to reshape the input
    pub fn forward(&self, z: &Tensor) -> Result<Tensor, Error> {
        match self {
            Activation::Linear => Ok(z.clone()),
            Activation::ReLU => {
                let mut out = z.clone();
                let relu = |x: f32| if x <= 0.0 { 0.0 } else { x };
                if out.len() >= cheap_map_parallel_threshold() {
                    out.par_mapv_inplace(relu);
                } else {
                    out.mapv_inplace(relu);
                }
                Ok(out)
            }
            Activation::Sigmoid => {
                let mut out = z.clone();
                let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                if out.len() >= exp_map_parallel_threshold() {
                    out.par_mapv_inplace(sigmoid);
                } else {
                    out.mapv_inplace(sigmoid);
                }
                Ok(out)
            }
            Activation::Tanh => {
                let tanh = |x: f32| x.tanh();
                let out = if z.len() >= exp_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(tanh);
                    out
                } else {
                    z.mapv(tanh)
                };
                Ok(out)
            }
            Activation::Softmax => softmax_forward(z),
        }
    }

    /// Computes the gradient with respect to the pre-activation input
    ///
    /// Every supported activation's derivative is expressible in terms of its own
    /// output, so this takes the cached activated tensor rather than the original input
    ///
    /// This is pure math with no clamping or NaN/Inf sanitization
    ///
    /// # Parameters
    ///
    /// - `activated` - The activated output `a` produced by [`forward`](Activation::forward)
    /// - `grad_output` - Upstream gradient `dL/da`
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The gradient `dL/dz`, same shape as `activated`
    ///
    /// # Errors
    ///
    /// - `Error::Computation` - Softmax failed to reshape the tensors
    pub fn backward(&self, activated: &Tensor, grad_output: &Tensor) -> Result<Tensor, Error> {
        match self {
            Activation::Linear => Ok(grad_output.clone()),
            Activation::ReLU => {
                // ReLU'(z) = 1 where z > 0. Since a = max(0, z), `a > 0` iff `z > 0`
                let mut grad = grad_output.clone();
                let relu_grad = |g: &mut f32, &a: &f32| {
                    if a <= 0.0 {
                        *g = 0.0;
                    }
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad).and(activated).par_for_each(relu_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(relu_grad);
                }
                Ok(grad)
            }
            Activation::Sigmoid => {
                // sigmoid'(z) = a * (1 - a)
                let mut grad = grad_output.clone();
                let sigmoid_grad = |g: &mut f32, &a: &f32| {
                    *g *= a * (1.0 - a);
                };
                if grad.len() >= exp_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(sigmoid_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(sigmoid_grad);
                }
                Ok(grad)
            }
            Activation::Tanh => {
                // tanh'(z) = 1 - a^2
                let mut grad = grad_output.clone();
                let tanh_grad = |g: &mut f32, &a: &f32| {
                    *g *= 1.0 - a * a;
                };
                if activated.len() >= exp_map_parallel_threshold() {
                    Zip::from(&mut grad).and(activated).par_for_each(tanh_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(tanh_grad);
                }
                Ok(grad)
            }
            Activation::Softmax => softmax_backward(activated, grad_output),
        }
    }
}

impl From<Linear> for Activation {
    fn from(_: Linear) -> Self {
        Activation::Linear
    }
}
impl From<ReLU> for Activation {
    fn from(_: ReLU) -> Self {
        Activation::ReLU
    }
}
impl From<Sigmoid> for Activation {
    fn from(_: Sigmoid) -> Self {
        Activation::Sigmoid
    }
}
impl From<Tanh> for Activation {
    fn from(_: Tanh) -> Self {
        Activation::Tanh
    }
}
impl From<Softmax> for Activation {
    fn from(_: Softmax) -> Self {
        Activation::Softmax
    }
}

/// Softmax forward over the last axis, with the row-max shift for numerical stability
fn softmax_forward(input: &Tensor) -> Result<Tensor, Error> {
    let shape = input.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(Error::invalid_input(format!(
            "Softmax requires input with at least 2 dimensions, got shape: {:?}",
            shape
        )));
    }

    // Flatten to [batch, features]; softmax runs over the last axis
    let batch_size: usize = shape[..ndim - 1].iter().product();
    let num_features = shape[ndim - 1];

    let mut output_2d = input
        .to_owned()
        .into_shape_with_order((batch_size, num_features))
        .context("Failed to reshape for softmax computation")?;

    let apply_softmax = |mut row: ArrayViewMut1<f32>| {
        // Subtract the row max so every exp argument is <= 0 (no overflow)
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        row.map_inplace(|x| *x = (*x - max_val).exp());
        let sum = row.sum().max(SOFTMAX_EPSILON);
        row.map_inplace(|x| *x /= sum);
    };

    if batch_size * num_features >= exp_map_parallel_threshold() {
        output_2d
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(apply_softmax);
    } else {
        output_2d.axis_iter_mut(Axis(0)).for_each(apply_softmax);
    }

    Ok(output_2d
        .into_shape_with_order(shape)
        .context("Failed to reshape back after softmax computation")?
        .into_dyn())
}

/// Softmax backward using the Jacobian-vector product expressed via the cached output
fn softmax_backward(output: &Tensor, grad_output: &Tensor) -> Result<Tensor, Error> {
    let shape = output.shape();
    let ndim = shape.len();
    let batch_size: usize = shape[..ndim - 1].iter().product();
    let num_features = shape[ndim - 1];

    let output_2d = output
        .to_shape((batch_size, num_features))
        .context("Failed to reshape output for backward")?;

    let grad_output_2d = grad_output
        .to_shape((batch_size, num_features))
        .context("Failed to reshape grad_output for backward")?;

    let mut grad_input_2d = Array2::<f32>::zeros((batch_size, num_features));

    // grad_input[i] = a[i] * (grad_output[i] - sum_j(a[j] * grad_output[j]))
    let compute_gradient = |mut grad_row: ArrayViewMut1<f32>,
                            out_row: ArrayView1<f32>,
                            grad_out_row: ArrayView1<f32>| {
        let dot: f32 = out_row
            .iter()
            .zip(grad_out_row.iter())
            .map(|(&o, &g)| o * g)
            .sum();

        for j in 0..num_features {
            grad_row[j] = out_row[j] * (grad_out_row[j] - dot);
        }
    };

    if batch_size * num_features >= exp_map_parallel_threshold() {
        Zip::from(grad_input_2d.axis_iter_mut(Axis(0)))
            .and(output_2d.axis_iter(Axis(0)))
            .and(grad_output_2d.axis_iter(Axis(0)))
            .par_for_each(compute_gradient);
    } else {
        Zip::from(grad_input_2d.axis_iter_mut(Axis(0)))
            .and(output_2d.axis_iter(Axis(0)))
            .and(grad_output_2d.axis_iter(Axis(0)))
            .for_each(compute_gradient);
    }

    Ok(grad_input_2d
        .into_shape_with_order(shape)
        .context("Failed to reshape grad_input back")?
        .into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    // Helpers

    /// Build a 2-D Tensor (ArrayD<f32>) from a row-major `data` vec with shape `(rows, cols)`
    fn tensor2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
        Array2::from_shape_vec((rows, cols), data)
            .expect("shape/data mismatch")
            .into_dyn()
    }

    // softmax_forward

    /// Softmax of a single row matches the hand-computed distribution
    #[test]
    fn softmax_forward_basic_row() {
        let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
        let output = softmax_forward(&input).expect("softmax_forward failed");
        let vals = output.as_slice().expect("not contiguous");
        assert_abs_diff_eq!(vals[0], 0.09003_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[1], 0.24473_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[2], 0.66524_f32, epsilon = 1e-4);
    }

    /// Softmax outputs sum to 1.0
    #[test]
    fn softmax_forward_sums_to_one() {
        let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
        let output = softmax_forward(&input).expect("softmax_forward failed");
        let sum: f32 = output.iter().sum();
        assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-6);
    }

    /// A row of equal large values stays numerically stable, yielding a uniform distribution
    #[test]
    fn softmax_forward_large_equal_values_stable() {
        let input = tensor2(1, 3, vec![1000.0, 1000.0, 1000.0]);
        let output = softmax_forward(&input).expect("softmax_forward failed");
        let vals = output.as_slice().expect("not contiguous");
        let third = 1.0_f32 / 3.0;
        assert_abs_diff_eq!(vals[0], third, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[1], third, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[2], third, epsilon = 1e-6);
    }

    /// A single-element row maps to 1.0
    #[test]
    fn softmax_forward_single_element_row() {
        let input = tensor2(1, 1, vec![5.0]);
        let output = softmax_forward(&input).expect("softmax_forward failed");
        let vals = output.as_slice().expect("not contiguous");
        assert_abs_diff_eq!(vals[0], 1.0_f32, epsilon = 1e-6);
    }

    /// A 1-D input (ndim < 2) returns an error
    #[test]
    fn softmax_forward_rejects_1d_input() {
        use ndarray::Array1;
        let input = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn();
        assert!(
            softmax_forward(&input).is_err(),
            "1-D input should return Err"
        );
    }

    // softmax_backward

    /// Backward gradient matches the hand-computed Jacobian-vector product
    #[test]
    fn softmax_backward_jacobian_vector_product() {
        let output = tensor2(1, 3, vec![0.25, 0.25, 0.5]);
        let grad_output = tensor2(1, 3, vec![1.0, 0.0, 0.0]);
        let grad_input = softmax_backward(&output, &grad_output).expect("softmax_backward failed");
        let vals = grad_input.as_slice().expect("not contiguous");
        assert_abs_diff_eq!(vals[0], 0.1875_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[1], -0.0625_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[2], -0.125_f32, epsilon = 1e-6);
    }

    /// The gradient row sums to (approximately) zero, since softmax Jacobian rows sum to zero
    #[test]
    fn softmax_backward_row_sums_to_zero() {
        let output = tensor2(1, 3, vec![0.25, 0.25, 0.5]);
        let grad_output = tensor2(1, 3, vec![1.0, 0.0, 0.0]);
        let grad_input = softmax_backward(&output, &grad_output).expect("softmax_backward failed");
        let row_sum: f32 = grad_input.iter().sum();
        assert_abs_diff_eq!(row_sum, 0.0_f32, epsilon = 1e-6);
    }

    // Activation enum - round-trip through the public API

    /// Activation::Softmax forward delegates to softmax_forward
    #[test]
    fn activation_softmax_forward_via_enum() {
        let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
        let output = Activation::Softmax
            .forward(&input)
            .expect("Activation::Softmax forward failed");
        let vals = output.as_slice().expect("not contiguous");
        // Same expected values as softmax_forward_basic_row
        assert_abs_diff_eq!(vals[0], 0.09003_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[1], 0.24473_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[2], 0.66524_f32, epsilon = 1e-4);
    }

    /// Activation::Softmax backward delegates to softmax_backward
    #[test]
    fn activation_softmax_backward_via_enum() {
        let output = tensor2(1, 3, vec![0.25, 0.25, 0.5]);
        let grad_output = tensor2(1, 3, vec![1.0, 0.0, 0.0]);
        let grad_input = Activation::Softmax
            .backward(&output, &grad_output)
            .expect("Activation::Softmax backward failed");
        let vals = grad_input.as_slice().expect("not contiguous");
        assert_abs_diff_eq!(vals[0], 0.1875_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[1], -0.0625_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[2], -0.125_f32, epsilon = 1e-6);
    }
}
