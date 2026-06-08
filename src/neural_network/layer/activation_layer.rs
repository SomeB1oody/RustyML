use crate::error::ModelError;
use crate::neural_network::Tensor;
use crate::{Deserialize, Serialize};
use ndarray::{Array2, ArrayView1, ArrayViewMut1, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Helper function to format a shape slice as a parenthesized tuple, e.g. `"(2, 3)"`.
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

/// Helper function to format the output shape for activation layers.
///
/// Returns a formatted string representing the shape of the cached tensor,
/// or "Unknown" if no tensor has been cached yet.
fn format_output_shape(cached_tensor: &Option<Tensor>) -> String {
    match cached_tensor {
        Some(tensor) => format_shape(tensor.shape()),
        None => "Unknown".to_string(),
    }
}

/// Linear (Identity) activation layer.
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

/// Gradient clipping bound shared by the differentiable activations to curb exploding gradients.
const GRAD_CLIP_VALUE: f32 = 1e6;
/// Lower input clamp for sigmoid/tanh to keep `exp` from overflowing.
const INPUT_CLIP_MIN: f32 = -500.0;
/// Upper input clamp for sigmoid/tanh to keep `exp` from overflowing.
const INPUT_CLIP_MAX: f32 = 500.0;
/// Epsilon guarding the softmax normalizer against division by zero.
const SOFTMAX_EPSILON: f32 = 1e-8;
/// Element threshold above which ReLU switches to parallel evaluation.
const RELU_PARALLEL_THRESHOLD: usize = 10_000;
/// Element threshold above which Sigmoid switches to parallel evaluation.
const SIGMOID_PARALLEL_THRESHOLD: usize = 1000;
/// Element threshold above which Tanh switches to parallel evaluation.
const TANH_PARALLEL_THRESHOLD: usize = 2048;
/// Row threshold above which Softmax switches to parallel evaluation.
const SOFTMAX_PARALLEL_THRESHOLD: usize = 8;

/// Clamps a differentiable activation's gradient, zeroing non-finite values.
fn clip_grad(g: &mut f32) {
    if g.is_nan() || g.is_infinite() {
        *g = 0.0;
    } else {
        *g = g.clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
    }
}

/// The element-wise activation functions that trainable layers can embed.
///
/// Dense, the convolutional layers and the recurrent layers each carry an
/// `Activation` value instead of a `T: ActivationLayer` type parameter. Keeping
/// the activation as a runtime enum makes the host layers non-generic, which
/// removes monomorphization bloat and lets weight deserialization downcast every
/// layer to a single concrete type rather than probing each `Layer<Act>` pairing.
///
/// The standalone activation *layers* ([`Linear`], [`ReLU`], [`Sigmoid`], [`Tanh`],
/// [`Softmax`]) delegate their math here, so this enum is the single source of
/// truth for both the forward transform and its derivative.
///
/// # Adding a new activation
///
/// Implement the math **here on the enum**, not in a layer:
/// 1. Add a variant to this enum.
/// 2. Handle it in [`Activation::forward`] (the transform) and [`Activation::backward`] (the
///    derivative, expressed in terms of the *activated output* `a`, not the pre-activation `z`).
/// 3. Add a thin standalone layer struct mirroring [`ReLU`] — a `{ output_cache }` field whose
///    `Layer` impl validates, caches the output, and delegates to this enum — plus a
///    `From<NewLayer> for Activation` impl so it works as an `impl Into<Activation>` argument.
///
/// The algorithm must live on the enum (not in the layer `impl`s) because the trainable layers
/// (Dense, the convolutional layers, the recurrent layers) store an `Activation` *value* and call
/// these **pure, stateless** methods inside their own forward/backward. A stateful `Layer` impl
/// (which caches `output` and takes `&mut self`) cannot serve that embedded, value-typed use
/// without reintroducing the `T: ActivationLayer` generic or `Box<dyn Layer>` — both removed in the
/// Step-1 refactor — and would duplicate the algorithm. The standalone structs are therefore thin
/// wrappers, never the source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Identity activation, `f(x) = x`.
    Linear,
    /// Rectified linear unit, `max(0, x)`.
    ReLU,
    /// Logistic sigmoid, `1 / (1 + e^-x)`.
    Sigmoid,
    /// Hyperbolic tangent, `tanh(x)`.
    Tanh,
    /// Softmax over the last axis (input must be at least 2D).
    Softmax,
}

impl Activation {
    /// Applies the activation to a pre-activation tensor `z`, returning the activated output.
    ///
    /// # Parameters
    ///
    /// - `z` - Pre-activation tensor (the linear output of the host layer)
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, ModelError>` - Activated tensor with the same shape as `z`
    ///
    /// # Errors
    ///
    /// - `ModelError::InputValidationError` - Softmax received an input with fewer than 2 dimensions
    /// - `ModelError::ProcessingError` - Softmax failed to reshape the input
    pub fn forward(&self, z: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            Activation::Linear => Ok(z.clone()),
            Activation::ReLU => {
                let mut out = z.clone();
                let relu = |x: f32| if x > 0.0 { x } else { 0.0 };
                if out.len() >= RELU_PARALLEL_THRESHOLD {
                    out.par_mapv_inplace(relu);
                } else {
                    out.mapv_inplace(relu);
                }
                Ok(out)
            }
            Activation::Sigmoid => {
                let mut out = z.clone();
                let sigmoid = |x: f32| {
                    let clipped = x.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX);
                    1.0 / (1.0 + (-clipped).exp())
                };
                if out.len() >= SIGMOID_PARALLEL_THRESHOLD {
                    out.par_mapv_inplace(sigmoid);
                } else {
                    out.mapv_inplace(sigmoid);
                }
                Ok(out)
            }
            Activation::Tanh => {
                let tanh = |x: f32| x.clamp(INPUT_CLIP_MIN, INPUT_CLIP_MAX).tanh();
                let out = if z.len() >= TANH_PARALLEL_THRESHOLD {
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

    /// Computes the gradient with respect to the pre-activation input.
    ///
    /// Every supported activation's derivative is expressible in terms of its own
    /// output, so this takes the cached activated tensor rather than the original input.
    ///
    /// # Parameters
    ///
    /// - `activated` - The activated output `a` produced by [`forward`](Activation::forward)
    /// - `grad_output` - Upstream gradient `dL/da`
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, ModelError>` - The gradient `dL/dz`, same shape as `activated`
    ///
    /// # Errors
    ///
    /// - `ModelError::ProcessingError` - Softmax failed to reshape the tensors
    pub fn backward(&self, activated: &Tensor, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            // Derivative is 1, so the gradient passes through unchanged.
            Activation::Linear => Ok(grad_output.clone()),
            Activation::ReLU => {
                // ReLU'(z) = 1 where z > 0. Since a = max(0, z), `a > 0` iff `z > 0`.
                let mut grad = grad_output.clone();
                let relu_grad = |g: &mut f32, &a: &f32| {
                    if a <= 0.0 {
                        *g = 0.0;
                    } else {
                        *g = g.clamp(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE);
                    }
                };
                if activated.len() >= RELU_PARALLEL_THRESHOLD {
                    Zip::from(&mut grad).and(activated).par_for_each(relu_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(relu_grad);
                }
                Ok(grad)
            }
            Activation::Sigmoid => {
                // sigmoid'(z) = a * (1 - a).
                let mut grad = grad_output.clone();
                let sigmoid_grad = |g: &mut f32, &a: &f32| {
                    *g *= a * (1.0 - a);
                    clip_grad(g);
                };
                if grad.len() >= SIGMOID_PARALLEL_THRESHOLD {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(sigmoid_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(sigmoid_grad);
                }
                Ok(grad)
            }
            Activation::Tanh => {
                // tanh'(z) = 1 - a^2.
                let mut grad = grad_output.clone();
                let tanh_grad = |g: &mut f32, &a: &f32| {
                    *g *= 1.0 - a * a;
                    clip_grad(g);
                };
                if activated.len() >= TANH_PARALLEL_THRESHOLD {
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

/// Softmax forward over the last axis, with the row-max shift for numerical stability.
fn softmax_forward(input: &Tensor) -> Result<Tensor, ModelError> {
    let shape = input.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(ModelError::InputValidationError(format!(
            "Softmax requires input with at least 2 dimensions, got shape: {:?}",
            shape
        )));
    }

    // Flatten to [batch, features]; softmax runs over the last axis.
    let batch_size: usize = shape[..ndim - 1].iter().product();
    let num_features = shape[ndim - 1];

    let mut output_2d = input
        .to_owned()
        .into_shape_with_order((batch_size, num_features))
        .map_err(|e| {
            ModelError::ProcessingError(format!("Failed to reshape for softmax computation: {}", e))
        })?;

    let apply_softmax = |mut row: ArrayViewMut1<f32>| {
        // Subtract the row max so every exp argument is <= 0 (no overflow).
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        row.map_inplace(|x| *x = (*x - max_val).exp());
        let sum = row.sum().max(SOFTMAX_EPSILON);
        row.map_inplace(|x| *x /= sum);
    };

    if batch_size > SOFTMAX_PARALLEL_THRESHOLD {
        output_2d
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(apply_softmax);
    } else {
        output_2d.axis_iter_mut(Axis(0)).for_each(apply_softmax);
    }

    output_2d
        .into_shape_with_order(shape)
        .map_err(|e| {
            ModelError::ProcessingError(format!(
                "Failed to reshape back after softmax computation: {}",
                e
            ))
        })
        .map(|out| out.into_dyn())
}

/// Softmax backward using the Jacobian-vector product expressed via the cached output.
fn softmax_backward(output: &Tensor, grad_output: &Tensor) -> Result<Tensor, ModelError> {
    let shape = output.shape();
    let ndim = shape.len();
    let batch_size: usize = shape[..ndim - 1].iter().product();
    let num_features = shape[ndim - 1];

    let output_2d = output
        .to_owned()
        .into_shape_with_order((batch_size, num_features))
        .map_err(|e| {
            ModelError::ProcessingError(format!("Failed to reshape output for backward: {}", e))
        })?;

    let grad_output_2d = grad_output
        .to_owned()
        .into_shape_with_order((batch_size, num_features))
        .map_err(|e| {
            ModelError::ProcessingError(format!(
                "Failed to reshape grad_output for backward: {}",
                e
            ))
        })?;

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
            clip_grad(&mut grad_row[j]);
        }
    };

    if batch_size > SOFTMAX_PARALLEL_THRESHOLD {
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

    grad_input_2d
        .into_shape_with_order(shape)
        .map_err(|e| {
            ModelError::ProcessingError(format!("Failed to reshape grad_input back: {}", e))
        })
        .map(|grad| grad.into_dyn())
}
