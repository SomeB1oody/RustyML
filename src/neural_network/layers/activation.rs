//! Element-wise activation functions and the standalone activation layers
//!
//! Defines the [`Activation`] enum, the single source of truth for each activation's forward
//! transform and derivative. The thin layer wrappers delegate their math to it: [`Linear`],
//! [`ReLU`], [`LeakyReLU`], [`ELU`], [`SELU`], [`Sigmoid`], [`HardSigmoid`], [`Tanh`],
//! [`Softplus`], [`Softsign`], [`Exponential`], and [`Softmax`]

use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::parallel_gates::{cheap_map_parallel_threshold, exp_map_parallel_threshold};
use crate::{Deserialize, Serialize};
use ndarray::{Array2, ArrayView1, ArrayViewMut1, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Formats a shape slice as a parenthesized tuple, for example `"(2, 3)"`
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

/// Formats the cached output shape for activation layers, or "Unknown" if nothing is cached
fn format_output_shape(cached_tensor: &Option<Tensor>) -> String {
    match cached_tensor {
        Some(tensor) => format_shape(tensor.shape()),
        None => "Unknown".to_string(),
    }
}

/// ELU (Exponential Linear Unit) activation layer
pub mod elu;
/// Exponential activation layer
pub mod exponential;
/// Hard sigmoid activation layer, a piecewise-linear approximation of the logistic sigmoid
pub mod hard_sigmoid;
/// Leaky ReLU activation layer
pub mod leaky_relu;
/// Linear (Identity) activation layer
pub mod linear;
/// ReLU (Rectified Linear Unit) activation layer
pub mod relu;
/// SELU (Scaled Exponential Linear Unit) activation layer
pub mod selu;
/// Sigmoid activation layer
pub mod sigmoid;
/// Softmax activation layer
pub mod softmax;
/// Softplus activation layer
pub mod softplus;
/// Softsign activation layer
pub mod softsign;
/// Tanh (Hyperbolic Tangent) activation layer
pub mod tanh;

pub use elu::ELU;
pub use exponential::Exponential;
pub use hard_sigmoid::HardSigmoid;
pub use leaky_relu::LeakyReLU;
pub use linear::Linear;
pub use relu::ReLU;
pub use selu::SELU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use softplus::Softplus;
pub use softsign::Softsign;
pub use tanh::Tanh;

/// SELU's fixed `alpha`, from Klambauer et al. (2017)
///
/// The published constant is 1.6732632423543772848170429916717. This literal is the f32 it
/// rounds to
const SELU_ALPHA: f32 = 1.673_263_2;

/// SELU's fixed `scale`, from Klambauer et al. (2017)
///
/// The published constant is 1.0507009873554804934193349852946. This literal is the f32 it
/// rounds to
const SELU_SCALE: f32 = 1.050_701;

/// The product `SELU_SCALE * SELU_ALPHA`, about 1.7580993
///
/// The negative branch multiplies by this single constant rather than by the 2 factors in
/// sequence. The backward pass then recovers the derivative as `a + SELU_SCALE_ALPHA`, which
/// is exact for the value the forward pass wrote
const SELU_SCALE_ALPHA: f32 = SELU_SCALE * SELU_ALPHA;

/// The slope of the hard sigmoid's linear segment, `1/6`
const HARD_SIGMOID_SLOPE: f32 = 1.0 / 6.0;

/// The element-wise activation functions that trainable layers can embed
///
/// Dense, the convolutional layers, and the recurrent layers each carry an `Activation` value
/// instead of a generic activation type parameter. A runtime enum keeps the host layers
/// non-generic. This removes monomorphization bloat and lets weight deserialization downcast
/// every layer to a single concrete type, instead of probing each `Layer<Act>` pairing
///
/// Every standalone activation *layer* in this module delegates its math here. This enum is
/// the single source of truth for both the forward transform and its derivative
///
/// # The output-only derivative contract
///
/// [`Activation::backward`] receives the *activated output* `a`, never the pre-activation `z`.
/// Every variant's derivative is therefore expressed through `a` alone, and a host layer caches
/// 1 tensor rather than 2.
///
/// This admits every activation whose derivative has a closed form in `a`, which covers the
/// full family below. It excludes GELU, SiLU (Swish), and Mish, whose `a = z * g(z)` shape has
/// no closed-form inverse. Adding those needs a wider contract that also hands the backward
/// pass the pre-activation.
///
/// The 2 saturating variants pay a small accuracy cost for the contract. `ELU` and `SELU`
/// recover their negative branch as `a + alpha`, a subtraction of 2 near-equal values far down
/// the tail. The absolute error stays inside 1 unit in the last place of `alpha`, at inputs
/// where the derivative is already close to 0.
///
/// # Adding a new activation
///
/// Implement the math on the enum, not in a layer.
///
/// 1. Add a variant to this enum
/// 2. Handle it in [`Activation::forward`] (the transform) and [`Activation::backward`] (the
///    derivative, expressed by the *activated output* `a`, not the pre-activation `z`)
/// 3. Reject any unusable parameter in [`Activation::validate`], which every trainable layer's
///    constructor calls
/// 4. Add a thin standalone layer struct that mirrors [`ReLU`]. Give it an `output_cache`
///    field. Its `Layer` impl validates the input, caches the output, and delegates to this
///    enum. Add a `From<NewLayer> for Activation` impl so the layer works as an
///    `impl Into<Activation>` argument
///
/// The algorithm lives on the enum, not in the layer `impl` blocks. The trainable layers
/// (Dense, the convolutional layers, the recurrent layers) store an `Activation` value. Each
/// layer calls these pure, stateless methods inside its own forward and backward passes.
///
/// A stateful `Layer` impl caches `output` and takes `&mut self`. It cannot serve that
/// embedded, value-typed use without a generic activation type parameter or `Box<dyn Layer>`.
/// Either option would duplicate the algorithm. The standalone structs stay thin wrappers,
/// never the source of truth
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    /// Leaky rectified linear unit, `x` for `x >= 0` and `negative_slope * x` below it
    ///
    /// Unlike [`Activation::ReLU`], the negative side keeps a non-zero gradient, so a unit
    /// whose pre-activation stays negative can still recover
    LeakyReLU {
        /// Slope applied below 0. Must be finite and greater than 0. Keras defaults the
        /// layer form to `0.3`. Use [`Activation::ReLU`] for a slope of 0
        negative_slope: f32,
    },
    /// Exponential linear unit, `x` for `x > 0` and `alpha * (e^x - 1)` below it
    ELU {
        /// Scale of the saturating negative branch. Must be finite and greater than 0.
        /// Keras defaults to `1.0`
        alpha: f32,
    },
    /// Scaled exponential linear unit, `scale * x` for `x > 0` and
    /// `scale * alpha * (e^x - 1)` below it
    ///
    /// `alpha` and `scale` are the fixed constants of Klambauer et al. (2017). Pair it with
    /// Lecun-normal initialization for the self-normalizing property to hold
    SELU,
    /// Softplus, `ln(1 + e^x)`, a smooth approximation of [`Activation::ReLU`]
    Softplus,
    /// Softsign, `x / (1 + |x|)`, a bounded activation that saturates polynomially rather
    /// than exponentially
    Softsign,
    /// Hard sigmoid, `clip(x/6 + 0.5, 0, 1)`, a piecewise-linear approximation of
    /// [`Activation::Sigmoid`] with no exponential
    HardSigmoid,
    /// Exponential, `e^x`
    Exponential,
}

impl Activation {
    /// Applies the activation to a pre-activation tensor `z` and returns the activated output
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
            Activation::LeakyReLU { negative_slope } => {
                let slope = *negative_slope;
                let leaky_relu = |x: f32| if x >= 0.0 { x } else { slope * x };
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(leaky_relu);
                    out
                } else {
                    z.mapv(leaky_relu)
                };
                Ok(out)
            }
            Activation::ELU { alpha } => {
                let alpha = *alpha;
                // `exp_m1` keeps the full precision of `e^x - 1` as x approaches 0
                let elu = |x: f32| if x > 0.0 { x } else { alpha * x.exp_m1() };
                let out = if z.len() >= exp_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(elu);
                    out
                } else {
                    z.mapv(elu)
                };
                Ok(out)
            }
            Activation::SELU => {
                let selu = |x: f32| {
                    if x > 0.0 {
                        SELU_SCALE * x
                    } else {
                        SELU_SCALE_ALPHA * x.exp_m1()
                    }
                };
                let out = if z.len() >= exp_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(selu);
                    out
                } else {
                    z.mapv(selu)
                };
                Ok(out)
            }
            Activation::Softplus => {
                // Factoring `e^x` out of the logarithm above 0 keeps both tails in range.
                // The direct form overflows for large x, and loses every digit for small x
                let softplus = |x: f32| {
                    if x > 0.0 {
                        x + (-x).exp().ln_1p()
                    } else {
                        x.exp().ln_1p()
                    }
                };
                let out = if z.len() >= exp_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(softplus);
                    out
                } else {
                    z.mapv(softplus)
                };
                Ok(out)
            }
            Activation::Softsign => {
                let softsign = |x: f32| x / (1.0 + x.abs());
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(softsign);
                    out
                } else {
                    z.mapv(softsign)
                };
                Ok(out)
            }
            Activation::HardSigmoid => {
                let hard_sigmoid = |x: f32| (x + 3.0).clamp(0.0, 6.0) / 6.0;
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(hard_sigmoid);
                    out
                } else {
                    z.mapv(hard_sigmoid)
                };
                Ok(out)
            }
            Activation::Exponential => {
                let exponential = |x: f32| x.exp();
                let out = if z.len() >= exp_map_parallel_threshold() {
                    let mut out = z.clone();
                    out.par_mapv_inplace(exponential);
                    out
                } else {
                    z.mapv(exponential)
                };
                Ok(out)
            }
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
                // ReLU'(z) = 1 when z > 0. Since a = max(0, z), `a > 0` exactly when `z > 0`
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
            Activation::LeakyReLU { negative_slope } => {
                // LeakyReLU'(z) = 1 when z >= 0, and `negative_slope` below it. A positive
                // slope keeps the sign of z, so `a < 0` marks exactly the elements with z < 0
                let slope = *negative_slope;
                let mut grad = grad_output.clone();
                let leaky_relu_grad = |g: &mut f32, &a: &f32| {
                    if a < 0.0 {
                        *g *= slope;
                    }
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(leaky_relu_grad);
                } else {
                    Zip::from(&mut grad)
                        .and(activated)
                        .for_each(leaky_relu_grad);
                }
                Ok(grad)
            }
            Activation::ELU { alpha } => {
                // ELU'(z) = 1 when z > 0. Below that a = alpha * (e^z - 1), which rearranges
                // to alpha * e^z = a + alpha, so the derivative needs no exponential
                let alpha = *alpha;
                let mut grad = grad_output.clone();
                let elu_grad = |g: &mut f32, &a: &f32| {
                    if a <= 0.0 {
                        *g *= a + alpha;
                    }
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad).and(activated).par_for_each(elu_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(elu_grad);
                }
                Ok(grad)
            }
            Activation::SELU => {
                // SELU'(z) = scale when z > 0, and scale * alpha * e^z below it. The same
                // rearrangement as ELU turns that into a + SELU_SCALE_ALPHA
                let mut grad = grad_output.clone();
                let selu_grad = |g: &mut f32, &a: &f32| {
                    *g *= if a > 0.0 {
                        SELU_SCALE
                    } else {
                        a + SELU_SCALE_ALPHA
                    };
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad).and(activated).par_for_each(selu_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(selu_grad);
                }
                Ok(grad)
            }
            Activation::Softplus => {
                // softplus'(z) = sigmoid(z). With a = ln(1 + e^z), that is 1 - e^-a.
                // `exp_m1` holds the tiny values the far negative tail produces
                let mut grad = grad_output.clone();
                let softplus_grad = |g: &mut f32, &a: &f32| {
                    *g *= -(-a).exp_m1();
                };
                if activated.len() >= exp_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(softplus_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(softplus_grad);
                }
                Ok(grad)
            }
            Activation::Softsign => {
                // softsign'(z) = 1 / (1 + |z|)^2. With a = z / (1 + |z|), that is (1 - |a|)^2
                let mut grad = grad_output.clone();
                let softsign_grad = |g: &mut f32, &a: &f32| {
                    let t = 1.0 - a.abs();
                    *g *= t * t;
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(softsign_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(softsign_grad);
                }
                Ok(grad)
            }
            Activation::HardSigmoid => {
                // The slope is 1/6 on the linear segment and 0 on both saturated ends. An end
                // is exactly where the forward clamp wrote 0 or 1
                let mut grad = grad_output.clone();
                let hard_sigmoid_grad = |g: &mut f32, &a: &f32| {
                    *g *= if a > 0.0 && a < 1.0 {
                        HARD_SIGMOID_SLOPE
                    } else {
                        0.0
                    };
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(hard_sigmoid_grad);
                } else {
                    Zip::from(&mut grad)
                        .and(activated)
                        .for_each(hard_sigmoid_grad);
                }
                Ok(grad)
            }
            Activation::Exponential => {
                // exp'(z) = e^z = a
                let mut grad = grad_output.clone();
                let exponential_grad = |g: &mut f32, &a: &f32| {
                    *g *= a;
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(exponential_grad);
                } else {
                    Zip::from(&mut grad)
                        .and(activated)
                        .for_each(exponential_grad);
                }
                Ok(grad)
            }
        }
    }

    /// Checks that a parameterized variant carries a usable parameter
    ///
    /// Every trainable layer's constructor calls this. An unusable slope or scale therefore
    /// fails where you build the model, not on the first forward pass. The parameter-free
    /// variants always pass.
    ///
    /// Both bounds are strict for the same reason. [`Activation::backward`] reads only the
    /// activated output, and it separates the 2 branches by the sign of that output. A slope
    /// or scale of 0 collapses the whole negative side onto `a = 0`, which erases the branch.
    /// A negative one inverts the sign, which reads the wrong branch
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - Ok when the activation is usable
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - `LeakyReLU`'s `negative_slope` or `ELU`'s `alpha` is not
    ///   finite and greater than 0
    pub fn validate(&self) -> Result<(), Error> {
        let (name, value, reason) = match self {
            Activation::LeakyReLU { negative_slope } => (
                "negative_slope",
                *negative_slope,
                "must be finite and greater than 0 (use Activation::ReLU for 0)",
            ),
            Activation::ELU { alpha } => ("alpha", *alpha, "must be finite and greater than 0"),
            _ => return Ok(()),
        };

        if !value.is_finite() || value <= 0.0 {
            return Err(Error::invalid_parameter(name, reason));
        }
        Ok(())
    }
}

impl From<Linear> for Activation {
    #[inline]
    fn from(_: Linear) -> Self {
        Activation::Linear
    }
}
impl From<ReLU> for Activation {
    #[inline]
    fn from(_: ReLU) -> Self {
        Activation::ReLU
    }
}
impl From<Sigmoid> for Activation {
    #[inline]
    fn from(_: Sigmoid) -> Self {
        Activation::Sigmoid
    }
}
impl From<Tanh> for Activation {
    #[inline]
    fn from(_: Tanh) -> Self {
        Activation::Tanh
    }
}
impl From<Softmax> for Activation {
    #[inline]
    fn from(_: Softmax) -> Self {
        Activation::Softmax
    }
}
impl From<LeakyReLU> for Activation {
    #[inline]
    fn from(layer: LeakyReLU) -> Self {
        Activation::LeakyReLU {
            negative_slope: layer.negative_slope,
        }
    }
}
impl From<ELU> for Activation {
    #[inline]
    fn from(layer: ELU) -> Self {
        Activation::ELU { alpha: layer.alpha }
    }
}
impl From<SELU> for Activation {
    #[inline]
    fn from(_: SELU) -> Self {
        Activation::SELU
    }
}
impl From<Softplus> for Activation {
    #[inline]
    fn from(_: Softplus) -> Self {
        Activation::Softplus
    }
}
impl From<Softsign> for Activation {
    #[inline]
    fn from(_: Softsign) -> Self {
        Activation::Softsign
    }
}
impl From<HardSigmoid> for Activation {
    #[inline]
    fn from(_: HardSigmoid) -> Self {
        Activation::HardSigmoid
    }
}
impl From<Exponential> for Activation {
    #[inline]
    fn from(_: Exponential) -> Self {
        Activation::Exponential
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

    // Flatten to [batch, features]. Softmax runs over the last axis
    let batch_size: usize = shape[..ndim - 1].iter().product();
    let num_features = shape[ndim - 1];

    // `to_owned` keeps the strides of an array that is not in C order, and
    // `into_shape_with_order` refuses such an array. `as_standard_layout` settles the layout
    // first, for the same 1 copy this line needs anyway
    let mut output_2d = input
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch_size, num_features))
        .context("Failed to reshape for softmax computation")?;

    let apply_softmax = |mut row: ArrayViewMut1<f32>| {
        // Subtract the row max so every exp argument is <= 0 (no overflow)
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        row.map_inplace(|x| *x = (*x - max_val).exp());
        // The max-shift guarantees one of the terms is exp(0)=1.0, so the sum is always >= 1.0
        let sum = row.sum();
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

/// Unit tests for the activation layer helpers and the `Activation` enum
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

    /// A row of equal large values stays numerically stable and produces a uniform distribution
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

    /// An input that is not in C order gives the same result as the same values in C order
    ///
    /// `Permute` produces C order on purpose, but a caller can hand a transposed tensor
    /// straight to this layer. Every other layer accepts one
    #[test]
    fn softmax_forward_accepts_input_that_is_not_in_c_order() {
        use ndarray::IxDyn;

        // `permuted_axes` reorders the strides only, and `to_owned` keeps them. The result
        // owns a contiguous buffer while `is_standard_layout` stays false
        let base = tensor2(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let transposed = base.view().permuted_axes(IxDyn(&[1, 0])).to_owned();
        assert!(
            !transposed.is_standard_layout(),
            "the test input must not be in C order"
        );

        let output = softmax_forward(&transposed).expect("softmax_forward must accept it");
        assert_eq!(output.shape(), &[3, 2]);

        let c_order: Tensor = transposed.as_standard_layout().into_owned();
        let want = softmax_forward(&c_order).expect("softmax_forward failed");
        for (got, expected) in output.iter().zip(want.iter()) {
            assert_abs_diff_eq!(*got, *expected, epsilon = 1e-6);
        }
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

    /// The gradient row sums to about zero, since the softmax Jacobian rows sum to zero
    #[test]
    fn softmax_backward_row_sums_to_zero() {
        let output = tensor2(1, 3, vec![0.25, 0.25, 0.5]);
        let grad_output = tensor2(1, 3, vec![1.0, 0.0, 0.0]);
        let grad_input = softmax_backward(&output, &grad_output).expect("softmax_backward failed");
        let row_sum: f32 = grad_input.iter().sum();
        assert_abs_diff_eq!(row_sum, 0.0_f32, epsilon = 1e-6);
    }

    // Round-trip tests for the Activation enum's public API

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

    // Forward and derivative tables pinned against Keras 3.15 on the jax backend, evaluated in
    // float32. These assert agreement with the reference implementation, not merely that
    // `forward` and `backward` agree with each other

    /// The probe inputs shared by every pinned table below
    const PROBES: [f32; 11] = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0];

    /// Asserts that `actual` matches the pinned `expected` row
    ///
    /// The tolerance mixes a relative and an absolute bound, because the tables span 5
    /// orders of magnitude. `Exponential` reaches 148, where 1 f32 ulp is already about 1e-5
    fn assert_pinned(name: &str, actual: &Tensor, expected: &[f32]) {
        let got: Vec<f32> = actual.iter().cloned().collect();
        assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-5 * e.abs().max(1.0);
            assert!(
                (g - e).abs() <= tol,
                "{name}[{i}] at x = {}: got {g}, Keras gives {e}, tolerance {tol}",
                PROBES[i]
            );
        }
    }

    /// Runs `activation` over [`PROBES`] and checks both the output and the derivative
    ///
    /// An all-ones upstream gradient makes the backward result the derivative itself
    fn check_against_keras(name: &str, activation: Activation, fwd: &[f32], grad: &[f32]) {
        let input = tensor2(1, PROBES.len(), PROBES.to_vec());
        let output = activation.forward(&input).expect("forward failed");
        assert_pinned(&format!("{name} forward"), &output, fwd);

        let ones = tensor2(1, PROBES.len(), vec![1.0; PROBES.len()]);
        let derivative = activation
            .backward(&output, &ones)
            .expect("backward failed");
        assert_pinned(&format!("{name} backward"), &derivative, grad);
    }

    /// LeakyReLU with the Keras layer default slope of 0.3. The derivative at exactly 0 is 1,
    /// because the positive branch is `x >= 0`
    #[test]
    fn leaky_relu_matches_keras() {
        check_against_keras(
            "LeakyReLU(0.3)",
            Activation::LeakyReLU {
                negative_slope: 0.3,
            },
            &[
                -1.5,
                -0.90000004,
                -0.6,
                -0.3,
                -0.15,
                0.0,
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
            ],
            &[0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
    }

    /// ELU with the Keras default alpha of 1.0
    #[test]
    fn elu_matches_keras() {
        check_against_keras(
            "ELU(1.0)",
            Activation::ELU { alpha: 1.0 },
            &[
                -0.99326205,
                -0.95021296,
                -0.86466473,
                -0.63212055,
                -0.39346933,
                0.0,
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
            ],
            &[
                0.0067379475,
                0.049787045,
                0.13533527,
                0.36787945,
                0.60653067,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        );
    }

    /// ELU with alpha = 0.5, the case that pins the branch convention
    ///
    /// At exactly 0 the derivative is `alpha`, not 1, because the positive branch is `x > 0`.
    /// The default alpha of 1.0 hides that, since both branches then give 1
    #[test]
    fn elu_half_alpha_matches_keras() {
        check_against_keras(
            "ELU(0.5)",
            Activation::ELU { alpha: 0.5 },
            &[
                -0.49663103,
                -0.47510648,
                -0.43233237,
                -0.31606027,
                -0.19673467,
                0.0,
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
            ],
            &[
                0.0033689737,
                0.024893522,
                0.067_667_63,
                0.18393973,
                0.30326533,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        );
    }

    /// SELU. The derivative at exactly 0 is `scale * alpha`, the same `x > 0` convention as ELU
    #[test]
    fn selu_matches_keras() {
        check_against_keras(
            "SELU",
            Activation::SELU,
            &[
                -1.7462534,
                -1.6705688,
                -1.5201665,
                -1.1113307,
                -0.691_758_2,
                0.0,
                0.525_350_5,
                1.050701,
                2.101402,
                3.152_103,
                5.253_505,
            ],
            &[
                0.011845981,
                0.087_530_57,
                0.23793285,
                0.646_768_6,
                1.0663412,
                1.7580993,
                1.050701,
                1.050701,
                1.050701,
                1.050701,
                1.050701,
            ],
        );
    }

    /// Softplus. The value at 0 is ln(2), and the derivative there is 0.5
    #[test]
    fn softplus_matches_keras() {
        check_against_keras(
            "Softplus",
            Activation::Softplus,
            &[
                0.0067153485,
                0.048587352,
                0.126928,
                0.313_261_7,
                0.474_077,
                // softplus(0) = ln(1 + 1), so the pinned value is exactly this constant
                std::f32::consts::LN_2,
                0.974_077,
                1.3132617,
                2.126_928,
                3.0485873,
                5.0067153,
            ],
            &[
                0.006692851,
                0.047425874,
                0.11920291,
                0.2689414,
                0.37754068,
                0.5,
                0.62245935,
                0.73105854,
                0.880_797,
                0.95257413,
                0.993_307_2,
            ],
        );
    }

    /// Softsign
    #[test]
    fn softsign_matches_keras() {
        check_against_keras(
            "Softsign",
            Activation::Softsign,
            &[
                -0.833_333_3,
                -0.75,
                -0.666_666_7,
                -0.5,
                -0.33333334,
                0.0,
                0.33333334,
                0.5,
                0.666_666_7,
                0.75,
                0.833_333_3,
            ],
            &[
                0.027777776,
                0.0625,
                0.11111112,
                0.25,
                0.44444448,
                1.0,
                0.44444448,
                0.25,
                0.11111112,
                0.0625,
                0.027777776,
            ],
        );
    }

    /// HardSigmoid. Both saturated ends are exact, and their derivative is 0
    #[test]
    fn hard_sigmoid_matches_keras() {
        check_against_keras(
            "HardSigmoid",
            Activation::HardSigmoid,
            &[
                0.0,
                0.0,
                0.16666667,
                0.33333334,
                0.416_666_7,
                0.5,
                0.583_333_4,
                0.666_666_7,
                0.833_333_4,
                1.0,
                1.0,
            ],
            &[
                0.0, 0.0, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
                0.16666667, 0.0, 0.0,
            ],
        );
    }

    /// Exponential, whose derivative equals its output
    #[test]
    fn exponential_matches_keras() {
        let table = [
            0.006737947,
            0.049787067,
            0.13533528,
            0.36787945,
            0.60653067,
            1.0,
            1.6487212,
            2.7182817,
            7.389_056,
            20.085537,
            148.41316,
        ];
        check_against_keras("Exponential", Activation::Exponential, &table, &table);
    }

    /// The softplus derivative survives the far negative tail
    ///
    /// At x = -40 the derivative is about 4.25e-18. Computing it as `1 - e^-a` rounds to
    /// exactly 0, because `e^-a` is 1 at f32 precision. `exp_m1` keeps the value
    #[test]
    fn softplus_backward_keeps_the_far_negative_tail() {
        let input = tensor2(1, 1, vec![-40.0]);
        let output = Activation::Softplus.forward(&input).expect("forward");
        let ones = tensor2(1, 1, vec![1.0]);
        let derivative = Activation::Softplus
            .backward(&output, &ones)
            .expect("backward");

        let got = derivative.iter().next().copied().expect("1 element");
        let expected = 4.248_354e-18_f32;
        assert!(
            (got - expected).abs() <= 1e-5 * expected,
            "softplus derivative at x = -40: got {got}, Keras gives {expected}"
        );
    }

    /// `validate` rejects a slope of 0 or below, since the backward pass reads the branch
    /// off the sign of the output
    #[test]
    fn validate_rejects_unusable_leaky_relu_slope() {
        for slope in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            let result = Activation::LeakyReLU {
                negative_slope: slope,
            }
            .validate();
            assert!(
                matches!(result, Err(Error::InvalidParameter { .. })),
                "slope {slope} must be rejected, got {result:?}"
            );
        }
    }

    /// The same bound applies to ELU's alpha
    #[test]
    fn validate_rejects_unusable_elu_alpha() {
        for alpha in [0.0, -1.0, f32::NAN, f32::NEG_INFINITY] {
            let result = Activation::ELU { alpha }.validate();
            assert!(
                matches!(result, Err(Error::InvalidParameter { .. })),
                "alpha {alpha} must be rejected, got {result:?}"
            );
        }
    }

    /// Every usable activation passes validation
    #[test]
    fn validate_accepts_usable_activations() {
        let usable = [
            Activation::Linear,
            Activation::ReLU,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax,
            Activation::LeakyReLU {
                negative_slope: 0.3,
            },
            Activation::ELU { alpha: 1.0 },
            Activation::SELU,
            Activation::Softplus,
            Activation::Softsign,
            Activation::HardSigmoid,
            Activation::Exponential,
        ];
        for activation in usable {
            assert!(
                activation.validate().is_ok(),
                "{activation:?} must pass validation"
            );
        }
    }
}
