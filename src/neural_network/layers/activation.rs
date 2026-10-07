//! Element-wise activation functions and the standalone activation layers
//!
//! Defines the [`Activation`] enum, the single source of truth for each activation's forward
//! transform and derivative. The thin layer wrappers delegate their math to it: [`Linear`],
//! [`ReLU`], [`LeakyReLU`], [`ELU`], [`SELU`], [`Sigmoid`], [`HardSigmoid`], [`Tanh`],
//! [`Softplus`], [`Softsign`], [`Exponential`], [`Softmax`], [`GELU`], [`SiLU`], and [`Mish`]
//!
//! [`PReLU`] stands apart. Its negative-side slope is a trainable array, so it holds its own
//! math and its own weights instead of delegating to the enum

use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::parallel_gates::{cheap_map_parallel_threshold, exp_map_parallel_threshold};
use crate::{Deserialize, Serialize};
use ndarray::{Array2, ArrayView1, ArrayViewMut1, Axis, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// ELU (Exponential Linear Unit) activation layer
pub mod elu;
/// Exponential activation layer
pub mod exponential;
/// GELU (Gaussian Error Linear Unit) activation layer
pub mod gelu;
/// Hard sigmoid activation layer, a piecewise-linear approximation of the logistic sigmoid
pub mod hard_sigmoid;
/// Leaky ReLU activation layer
pub mod leaky_relu;
/// Linear (Identity) activation layer
pub mod linear;
/// Mish activation layer
pub mod mish;
/// PReLU activation layer, whose negative-side slope is a trainable parameter
pub mod p_relu;
/// ReLU (Rectified Linear Unit) activation layer
pub mod relu;
/// SELU (Scaled Exponential Linear Unit) activation layer
pub mod selu;
/// Sigmoid activation layer
pub mod sigmoid;
/// SiLU (Sigmoid Linear Unit) activation layer, also named Swish
pub mod silu;
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
pub use gelu::GELU;
pub use hard_sigmoid::HardSigmoid;
pub use leaky_relu::LeakyReLU;
pub use linear::Linear;
pub use mish::Mish;
pub use p_relu::PReLU;
pub use relu::ReLU;
pub use selu::SELU;
pub use sigmoid::Sigmoid;
pub use silu::SiLU;
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
/// The negative branch multiplies by this single constant. The backward pass then recovers
/// the derivative as `a + SELU_SCALE_ALPHA`, which is exact for the value the forward pass
/// wrote
const SELU_SCALE_ALPHA: f32 = SELU_SCALE * SELU_ALPHA;

/// The slope of the hard sigmoid's linear segment, `1/6`
const HARD_SIGMOID_SLOPE: f32 = 1.0 / 6.0;

/// The scale `sqrt(2 / pi)` of the tanh approximation of GELU
const GELU_TANH_SCALE: f32 = 0.797_884_6;

/// The cubic coefficient of the tanh approximation of GELU, from Hendrycks and Gimpel (2016)
const GELU_TANH_CUBIC: f32 = 0.044_715;

/// The density scale `1 / sqrt(2 * pi)` of the standard normal distribution
const NORMAL_DENSITY_SCALE: f32 = 0.398_942_3;

/// The default softmax axis, the last axis of the input
///
/// A negative axis counts back from the end, so `-1` holds for every rank
pub const DEFAULT_SOFTMAX_AXIS: i32 = -1;

/// The element-wise activation functions that trainable layers can embed
///
/// Dense, the convolutional layers, and the recurrent layers each carry an `Activation` value
/// instead of a generic activation type parameter. A runtime enum keeps the host layers
/// non-generic. This removes monomorphization bloat and lets weight deserialization downcast
/// every layer to a single concrete type, instead of probing each `Layer<Act>` pairing
///
/// Every standalone activation *layer* in this module delegates its math here. This enum is
/// the single source of truth for both the forward transform and its derivative. The
/// algorithm lives here, and not in a layer `impl` block
///
/// # Notes
///
/// The backward pass of each variant reads 1 tensor. [`Activation::saves`] names it: the
/// activated output `a`, or the pre-activation `z`. A host layer therefore caches 1 tensor, not
/// 2. [`Activation::forward_train`] returns that tensor in an [`ActivationCache`], and
/// [`Activation::backward`] refuses a cache that holds the other tensor.
///
/// A variant reads `a` when its derivative has a closed form in `a`. GELU, SiLU, and Mish have
/// the shape `a = z * g(z)`, which has no closed-form inverse, so these 3 variants read `z`.
///
/// The 2 saturating variants pay a small accuracy cost for the contract. `ELU` recovers its
/// negative branch as `a + alpha`, and `SELU` recovers its negative branch as
/// `a + SELU_SCALE_ALPHA`. Both computations subtract 2 near-equal values far down the tail.
/// The absolute error stays inside 1 unit in the last place of that constant, at inputs where
/// the derivative is already close to 0.
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
    /// Softmax over 1 axis, `e^x / sum(e^x)` along that axis
    ///
    /// The lanes along the chosen axis each become a probability distribution that sums
    /// to 1. The whole tensor keeps its shape
    Softmax {
        /// Axis to normalize. A negative value counts back from the end, so the default
        /// `-1` always means the last axis
        ///
        /// The axis resolves against the rank of the input on each call, not when the
        /// value is built. The same value therefore normalizes the last axis of a rank-2
        /// input and the last axis of a rank-4 input. A resolved index outside the rank
        /// of the input fails the call
        ///
        /// A trainable host layer accepts `-1` alone. See [`Activation::validate`]
        axis: i32,
    },
    /// Leaky rectified linear unit, `x` for `x >= 0` and `negative_slope * x` below it
    ///
    /// Unlike [`Activation::ReLU`], the negative side keeps a non-zero gradient, so a unit
    /// whose pre-activation stays negative can still recover
    LeakyReLU {
        /// Slope applied below 0. Must be finite and greater than 0. The layer form
        /// defaults to `0.3`. Use [`Activation::ReLU`] for a slope of 0
        negative_slope: f32,
    },
    /// Exponential linear unit, `x` for `x > 0` and `alpha * (e^x - 1)` below it
    ELU {
        /// Scale of the saturating negative branch. Must be finite and greater than 0.
        /// The layer form defaults to `1.0`
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
    /// Gaussian error linear unit, `x * Phi(x)`, where `Phi` is the standard normal
    /// cumulative distribution function, from Hendrycks and Gimpel (2016)
    GELU {
        /// When `true`, the variant uses the tanh approximation
        /// `x * sigmoid(2 * sqrt(2 / pi) * (x + 0.044715 * x^3))`. When `false`, it uses the
        /// exact `Phi`. The layer form defaults to `false`
        approximate: bool,
    },
    /// Sigmoid linear unit, `x * sigmoid(x)`, also named Swish
    SiLU,
    /// Mish, `x * tanh(softplus(x))`, from Misra (2019)
    Mish,
}

/// The tensor that the backward pass of an [`Activation`] reads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationInput {
    /// The activated output `a`
    Output,
    /// The pre-activation `z`
    PreActivation,
}

/// The tensor that a training forward pass saves for the backward pass of an [`Activation`]
///
/// [`Activation::forward_train`] makes the cache. The cache records which tensor it holds,
/// and [`Activation::backward`] refuses a cache that holds the other tensor
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationCache {
    /// The tensor that the cache holds
    holds: ActivationInput,
    /// The saved tensor
    tensor: Tensor,
}

impl ActivationCache {
    /// Wraps a tensor that a host layer saved without [`Activation::forward_train`]
    ///
    /// A host layer that fuses an activation into a matrix product gets the output from the
    /// product. It wraps that output here as an [`ActivationInput::Output`] cache
    ///
    /// # Parameters
    ///
    /// - `holds` - The tensor that `tensor` is
    /// - `tensor` - The saved tensor
    ///
    /// # Returns
    ///
    /// - `ActivationCache` - The cache
    pub(crate) fn new(holds: ActivationInput, tensor: Tensor) -> Self {
        Self { holds, tensor }
    }

    /// The tensor that the cache holds
    ///
    /// # Returns
    ///
    /// - `ActivationInput` - [`ActivationInput::Output`] or [`ActivationInput::PreActivation`]
    pub fn holds(&self) -> ActivationInput {
        self.holds
    }

    /// The saved tensor
    ///
    /// # Returns
    ///
    /// - `&Tensor` - The activated output or the pre-activation, as [`ActivationCache::holds`]
    ///   names
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// The shape of the saved tensor, which is also the shape of the activated output
    ///
    /// # Returns
    ///
    /// - `&[usize]` - The shape
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Gives back the saved tensor
    ///
    /// # Returns
    ///
    /// - `Tensor` - The activated output or the pre-activation, as [`ActivationCache::holds`]
    ///   names
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

/// Logistic sigmoid of 1 value
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus of 1 value, `ln(1 + e^x)`
///
/// Above 0 the function factors `e^x` out of the logarithm. This keeps both tails in range.
/// The direct form overflows for a large `x`, and it loses every digit for a small `x`
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Exact GELU of 1 value, `x * Phi(x)`
///
/// `Phi(x) = erfc(-x / sqrt(2)) / 2`. The complementary error function keeps the relative
/// precision of `Phi` in the negative tail, where `1 + erf` would cancel to 0
#[inline]
fn gelu_exact(x: f32) -> f32 {
    x * 0.5 * libm::erfcf(-x * std::f32::consts::FRAC_1_SQRT_2)
}

/// Derivative of the exact GELU at `x`, `Phi(x) + x * phi(x)`
#[inline]
fn gelu_exact_grad(x: f32) -> f32 {
    let cdf = 0.5 * libm::erfcf(-x * std::f32::consts::FRAC_1_SQRT_2);
    let pdf = NORMAL_DENSITY_SCALE * (-0.5 * x * x).exp();
    cdf + x * pdf
}

/// The sigmoid argument `2 * sqrt(2 / pi) * (x + 0.044715 * x^3)` of the tanh approximation
///
/// `1 + tanh(u) = 2 * sigmoid(2 * u)`. The sigmoid form does not cancel in the negative tail
#[inline]
fn gelu_tanh_argument(x: f32) -> f32 {
    2.0 * GELU_TANH_SCALE * (x + GELU_TANH_CUBIC * x * x * x)
}

/// Tanh approximation of GELU at `x`
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    x * sigmoid(gelu_tanh_argument(x))
}

/// Derivative of the tanh approximation of GELU at `x`
#[inline]
///
/// Where the sigmoid saturates, `s * (1 - s)` is 0 and the derivative is `s`. The function
/// returns `s` there, because `x * x` overflows to infinity for `|x|` above about 1.8e19, and
/// `0 * infinity` is NaN
fn gelu_tanh_grad(x: f32) -> f32 {
    let s = sigmoid(gelu_tanh_argument(x));
    let slope = s * (1.0 - s);
    if slope == 0.0 {
        return s;
    }
    let argument_grad = 2.0 * GELU_TANH_SCALE * (1.0 + 3.0 * GELU_TANH_CUBIC * x * x);
    s + x * slope * argument_grad
}

/// SiLU of 1 value, `x * sigmoid(x)`
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Derivative of SiLU at `x`, `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`
#[inline]
fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Mish of 1 value, `x * tanh(softplus(x))`
#[inline]
fn mish(x: f32) -> f32 {
    x * softplus(x).tanh()
}

/// Derivative of Mish at `x`, `t + x * (1 - t^2) * sigmoid(x)` with `t = tanh(softplus(x))`
#[inline]
fn mish_grad(x: f32) -> f32 {
    let t = softplus(x).tanh();
    t + x * (1.0 - t * t) * sigmoid(x)
}

/// Applies an element-wise map, in parallel when the tensor reaches `threshold` elements
fn map_elements(z: &Tensor, threshold: usize, f: impl Fn(f32) -> f32 + Sync + Send) -> Tensor {
    if z.len() >= threshold {
        Zip::from(z).par_map_collect(|&x| f(x))
    } else {
        z.mapv(f)
    }
}

/// Multiplies each upstream gradient by the derivative at the matching pre-activation
///
/// # Errors
///
/// - `Error::ShapeMismatch` - `grad_output` and `z` differ in shape
fn scale_by_derivative(
    z: &Tensor,
    grad_output: &Tensor,
    derivative: impl Fn(f32) -> f32 + Sync + Send,
) -> Result<Tensor, Error> {
    if grad_output.shape() != z.shape() {
        return Err(Error::shape_mismatch(z.shape(), grad_output.shape()));
    }
    let mut grad = grad_output.clone();
    let scale = |g: &mut f32, &x: &f32| *g *= derivative(x);
    if z.len() >= exp_map_parallel_threshold() {
        Zip::from(&mut grad).and(z).par_for_each(scale);
    } else {
        Zip::from(&mut grad).and(z).for_each(scale);
    }
    Ok(grad)
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
    /// - `Error::InvalidInput` - The Softmax axis resolves outside the rank of `z`
    /// - `Error::Computation` - Softmax failed to reshape the input
    pub fn forward(&self, z: &Tensor) -> Result<Tensor, Error> {
        match self {
            Activation::Linear => Ok(z.clone()),
            Activation::ReLU => {
                let relu = |x: f32| if x <= 0.0 { 0.0 } else { x };
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| relu(x))
                } else {
                    z.mapv(relu)
                };
                Ok(out)
            }
            Activation::Sigmoid => {
                let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                let out = if z.len() >= exp_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| sigmoid(x))
                } else {
                    z.mapv(sigmoid)
                };
                Ok(out)
            }
            Activation::Tanh => {
                let tanh = |x: f32| x.tanh();
                let out = if z.len() >= exp_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| tanh(x))
                } else {
                    z.mapv(tanh)
                };
                Ok(out)
            }
            Activation::Softmax { axis } => softmax_forward(z, *axis),
            Activation::LeakyReLU { negative_slope } => {
                let slope = *negative_slope;
                let leaky_relu = |x: f32| if x >= 0.0 { x } else { slope * x };
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| leaky_relu(x))
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
                    Zip::from(z).par_map_collect(|&x| elu(x))
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
                    Zip::from(z).par_map_collect(|&x| selu(x))
                } else {
                    z.mapv(selu)
                };
                Ok(out)
            }
            Activation::Softplus => Ok(map_elements(z, exp_map_parallel_threshold(), softplus)),
            Activation::Softsign => {
                let softsign = |x: f32| x / (1.0 + x.abs());
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| softsign(x))
                } else {
                    z.mapv(softsign)
                };
                Ok(out)
            }
            Activation::HardSigmoid => {
                let hard_sigmoid = |x: f32| (x + 3.0).clamp(0.0, 6.0) / 6.0;
                let out = if z.len() >= cheap_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| hard_sigmoid(x))
                } else {
                    z.mapv(hard_sigmoid)
                };
                Ok(out)
            }
            Activation::Exponential => {
                let exponential = |x: f32| x.exp();
                let out = if z.len() >= exp_map_parallel_threshold() {
                    Zip::from(z).par_map_collect(|&x| exponential(x))
                } else {
                    z.mapv(exponential)
                };
                Ok(out)
            }
            Activation::GELU { approximate: false } => {
                Ok(map_elements(z, exp_map_parallel_threshold(), gelu_exact))
            }
            Activation::GELU { approximate: true } => {
                Ok(map_elements(z, exp_map_parallel_threshold(), gelu_tanh))
            }
            Activation::SiLU => Ok(map_elements(z, exp_map_parallel_threshold(), silu)),
            Activation::Mish => Ok(map_elements(z, exp_map_parallel_threshold(), mish)),
        }
    }

    /// The tensor that the backward pass of this activation reads
    ///
    /// # Returns
    ///
    /// - `ActivationInput` - [`ActivationInput::PreActivation`] for GELU, SiLU, and Mish, and
    ///   [`ActivationInput::Output`] for every other variant
    pub fn saves(&self) -> ActivationInput {
        match self {
            Activation::GELU { .. } | Activation::SiLU | Activation::Mish => {
                ActivationInput::PreActivation
            }
            _ => ActivationInput::Output,
        }
    }

    /// Applies the activation and saves the tensor that the backward pass reads
    ///
    /// The cache holds a copy of the output, or `z` itself, as [`Activation::saves`] names
    ///
    /// # Parameters
    ///
    /// - `z` - Pre-activation tensor (the linear output of the host layer)
    ///
    /// # Returns
    ///
    /// - `Result<(Tensor, ActivationCache), Error>` - The activated tensor, and the cache for
    ///   [`Activation::backward`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - The Softmax axis resolves outside the rank of `z`
    /// - `Error::Computation` - Softmax failed to reshape the input
    pub fn forward_train(&self, z: Tensor) -> Result<(Tensor, ActivationCache), Error> {
        let output = self.forward(&z)?;
        let cache = match self.saves() {
            ActivationInput::Output => {
                ActivationCache::new(ActivationInput::Output, output.clone())
            }
            ActivationInput::PreActivation => {
                ActivationCache::new(ActivationInput::PreActivation, z)
            }
        };
        Ok((output, cache))
    }

    /// The activated output that a cache stands for
    ///
    /// An [`ActivationInput::Output`] cache gives a copy of its tensor. An
    /// [`ActivationInput::PreActivation`] cache gives the forward pass of its tensor
    ///
    /// # Parameters
    ///
    /// - `cache` - A cache that [`Activation::forward_train`] of this activation made
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The activated output
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - The cache holds another tensor than [`Activation::saves`]
    ///   names, or the Softmax axis resolves outside the rank of the tensor
    /// - `Error::Computation` - Softmax failed to reshape the tensor
    pub fn output_of(&self, cache: &ActivationCache) -> Result<Tensor, Error> {
        self.check_cache(cache)?;
        match cache.holds {
            ActivationInput::Output => Ok(cache.tensor.clone()),
            ActivationInput::PreActivation => self.forward(&cache.tensor),
        }
    }

    /// Refuses a cache that holds another tensor than this activation reads
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - The cache holds another tensor than [`Activation::saves`]
    ///   names
    fn check_cache(&self, cache: &ActivationCache) -> Result<(), Error> {
        if cache.holds != self.saves() {
            return Err(Error::invalid_input(format!(
                "{self:?} reads {:?}, but the cache holds {:?}",
                self.saves(),
                cache.holds
            )));
        }
        Ok(())
    }

    /// Computes the gradient with respect to the pre-activation input
    ///
    /// This is pure math with no clamping or NaN/Inf sanitization
    ///
    /// # Parameters
    ///
    /// - `cache` - The cache that [`Activation::forward_train`] of this activation made
    /// - `grad_output` - Upstream gradient `dL/da`
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - The gradient `dL/dz`, with the shape of the cached tensor
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - The cache holds another tensor than [`Activation::saves`]
    ///   names, or the Softmax axis resolves outside the rank of the cached tensor
    /// - `Error::ShapeMismatch` - `grad_output` and the cached tensor differ in shape
    /// - `Error::Computation` - Softmax failed to reshape the tensors
    pub fn backward(&self, cache: &ActivationCache, grad_output: &Tensor) -> Result<Tensor, Error> {
        self.check_cache(cache)?;
        let activated = &cache.tensor;
        if grad_output.shape() != activated.shape() {
            return Err(Error::shape_mismatch(
                activated.shape(),
                grad_output.shape(),
            ));
        }
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
                let mut grad = grad_output.clone();
                let sigmoid_grad = |g: &mut f32, &a: &f32| {
                    *g *= a * (1.0 - a);
                };
                if grad.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad)
                        .and(activated)
                        .par_for_each(sigmoid_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(sigmoid_grad);
                }
                Ok(grad)
            }
            Activation::Tanh => {
                let mut grad = grad_output.clone();
                let tanh_grad = |g: &mut f32, &a: &f32| {
                    *g *= 1.0 - a * a;
                };
                if activated.len() >= cheap_map_parallel_threshold() {
                    Zip::from(&mut grad).and(activated).par_for_each(tanh_grad);
                } else {
                    Zip::from(&mut grad).and(activated).for_each(tanh_grad);
                }
                Ok(grad)
            }
            Activation::Softmax { axis } => softmax_backward(activated, grad_output, *axis),
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
            Activation::GELU { approximate: false } => {
                scale_by_derivative(activated, grad_output, gelu_exact_grad)
            }
            Activation::GELU { approximate: true } => {
                scale_by_derivative(activated, grad_output, gelu_tanh_grad)
            }
            Activation::SiLU => scale_by_derivative(activated, grad_output, silu_grad),
            Activation::Mish => scale_by_derivative(activated, grad_output, mish_grad),
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
    /// # Notes
    ///
    /// An embedded [`Activation::Softmax`] accepts the default axis `-1` alone. This
    /// restriction is a deliberate simplification, not a limit of the math
    ///
    /// The host layers do not agree on the rank of the tensor they hand to the activation.
    /// [`Dense`](crate::neural_network::layers::dense::Dense) and the convolution layers apply
    /// the activation to the full output tensor. The recurrent layers apply the activation to
    /// 1 rank-2 state per timestep, and that state has no time axis
    ///
    /// The axis `-1` names the same lane in all of these hosts. Any other axis names a
    /// different tensor axis in a recurrent layer than in its output. A per-host axis rule is
    /// deferred, not impossible. Use the standalone [`Softmax`] layer for a different axis
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - `LeakyReLU`'s `negative_slope` or `ELU`'s `alpha` is not
    ///   finite and greater than 0, or `Softmax`'s `axis` is not `-1`
    pub fn validate(&self) -> Result<(), Error> {
        let (name, value, reason) = match self {
            Activation::LeakyReLU { negative_slope } => (
                "negative_slope",
                *negative_slope,
                "must be finite and greater than 0 (use Activation::ReLU for 0)",
            ),
            Activation::ELU { alpha } => ("alpha", *alpha, "must be finite and greater than 0"),
            Activation::Softmax { axis } => {
                if *axis != DEFAULT_SOFTMAX_AXIS {
                    return Err(Error::invalid_parameter(
                        "axis",
                        "must be -1 in a trainable layer (use the Softmax layer for another axis)",
                    ));
                }
                return Ok(());
            }
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
    fn from(layer: Softmax) -> Self {
        Activation::Softmax { axis: layer.axis }
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
impl From<GELU> for Activation {
    #[inline]
    fn from(layer: GELU) -> Self {
        Activation::GELU {
            approximate: layer.approximate,
        }
    }
}
impl From<SiLU> for Activation {
    #[inline]
    fn from(_: SiLU) -> Self {
        Activation::SiLU
    }
}
impl From<Mish> for Activation {
    #[inline]
    fn from(_: Mish) -> Self {
        Activation::Mish
    }
}

/// Resolves a softmax axis against the rank of the tensor it applies to
///
/// A negative axis counts back from the end. The resolution happens on each call, never when
/// the [`Activation`] value or the [`Softmax`] layer is built. A value that holds a resolved
/// index would reduce the wrong axis as soon as the rank of the input changed
///
/// # Parameters
///
/// - `axis` - Axis to normalize, which can be negative
/// - `ndim` - Rank of the tensor
///
/// # Returns
///
/// - `Result<usize, Error>` - The resolved axis, from 0 through `ndim - 1`
///
/// # Errors
///
/// - `Error::InvalidInput` - The resolved axis falls outside the rank
fn resolve_softmax_axis(axis: i32, ndim: usize) -> Result<usize, Error> {
    // The widening to i64 keeps the sum in range for every i32 input, `i32::MIN` included
    let resolved = if axis < 0 {
        axis as i64 + ndim as i64
    } else {
        axis as i64
    };

    if resolved < 0 || resolved >= ndim as i64 {
        return Err(Error::invalid_input(format!(
            "Softmax axis {axis} is out of bounds for an input of rank {ndim}"
        )));
    }
    Ok(resolved as usize)
}

/// Softmax forward over 1 axis, with the lane-max shift for numerical stability
///
/// The last axis takes the fold-to-2D path, whose lanes are contiguous rows. Every other axis
/// takes the lane path, because the lanes along a non-final axis are strided and a fold would
/// mix them
fn softmax_forward(input: &Tensor, axis: i32) -> Result<Tensor, Error> {
    let shape = input.shape();
    let ndim = shape.len();
    let axis = resolve_softmax_axis(axis, ndim)?;

    let apply_softmax = |mut lane: ArrayViewMut1<f32>| {
        // Subtract the lane max so every exp argument is <= 0 (no overflow)
        let max_val = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        lane.map_inplace(|x| *x = (*x - max_val).exp());
        // The max-shift guarantees 1 of the terms is exp(0)=1.0, so the sum is always >= 1.0
        let sum = lane.sum();
        lane.map_inplace(|x| *x /= sum);
    };

    if axis + 1 == ndim {
        // Flatten to [batch, features]. A fold of the leading axes keeps every last-axis lane
        // whole, so the 2D rows are exactly the lanes
        let batch_size: usize = shape[..ndim - 1].iter().product();
        let num_features = shape[ndim - 1];

        // `to_owned` keeps a non-C-order array's strides, so `into_shape_with_order` then
        // refuses it. `as_standard_layout` puts the array in C order first.
        let mut output_2d = input
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((batch_size, num_features))
            .context("Failed to reshape for softmax computation")?;

        // The task is 1 row, so a single-row input has nothing to spread and must stay serial
        if batch_size > 1 && batch_size * num_features >= exp_map_parallel_threshold() {
            output_2d
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(apply_softmax);
        } else {
            output_2d.axis_iter_mut(Axis(0)).for_each(apply_softmax);
        }

        return Ok(output_2d
            .into_shape_with_order(shape)
            .context("Failed to reshape back after softmax computation")?
            .into_dyn());
    }

    // A fresh C-order destination takes the input values, and the lanes then hold the strided
    // groups that the chosen axis selects. This keeps the result in C order for every input
    // layout, which no reshape of a strided view can promise
    let mut output = Tensor::zeros(input.raw_dim());
    output.assign(input);

    let lane_len = shape[axis];
    let lane_count: usize = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis)
        .map(|(_, &length)| length)
        .product();

    // The task is 1 lane, so a single-lane input has nothing to spread and must stay serial
    if lane_count > 1 && lane_count * lane_len >= exp_map_parallel_threshold() {
        Zip::from(output.lanes_mut(Axis(axis))).par_for_each(apply_softmax);
    } else {
        Zip::from(output.lanes_mut(Axis(axis))).for_each(apply_softmax);
    }

    Ok(output)
}

/// Softmax backward using the Jacobian-vector product expressed via the cached output
///
/// The product `dL/dz = a * (g - sum_over_axis(a * g))` is exact for every axis, so the
/// output-only derivative contract holds as long as the value carries the axis
fn softmax_backward(output: &Tensor, grad_output: &Tensor, axis: i32) -> Result<Tensor, Error> {
    let shape = output.shape();
    let ndim = shape.len();
    let axis = resolve_softmax_axis(axis, ndim)?;

    // Softmax keeps the shape, so the 2 tensors walk the same lanes. The lane path pairs the
    // 2 shapes directly, and a mismatch there is a panic rather than an error
    if grad_output.shape() != shape {
        return Err(Error::shape_mismatch(shape, grad_output.shape()));
    }

    // grad_input[i] = a[i] * (grad_output[i] - sum_j(a[j] * grad_output[j]))
    let compute_gradient = |mut grad_lane: ArrayViewMut1<f32>,
                            out_lane: ArrayView1<f32>,
                            grad_out_lane: ArrayView1<f32>| {
        let dot: f32 = out_lane
            .iter()
            .zip(grad_out_lane.iter())
            .map(|(&o, &g)| o * g)
            .sum();

        Zip::from(&mut grad_lane)
            .and(&out_lane)
            .and(&grad_out_lane)
            .for_each(|grad, &o, &g| *grad = o * (g - dot));
    };

    if axis + 1 == ndim {
        let batch_size: usize = shape[..ndim - 1].iter().product();
        let num_features = shape[ndim - 1];

        let output_2d = output
            .to_shape((batch_size, num_features))
            .context("Failed to reshape output for backward")?;

        let grad_output_2d = grad_output
            .to_shape((batch_size, num_features))
            .context("Failed to reshape grad_output for backward")?;

        let mut grad_input_2d = Array2::<f32>::zeros((batch_size, num_features));

        // The backward pass is a row dot and a scale, with no `exp`, so it is a cheap map. The
        // task is 1 row, so a single-row input stays serial
        if batch_size > 1 && batch_size * num_features >= cheap_map_parallel_threshold() {
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

        return Ok(grad_input_2d
            .into_shape_with_order(shape)
            .context("Failed to reshape grad_input back")?
            .into_dyn());
    }

    let mut grad_input = Tensor::zeros(output.raw_dim());

    let lane_len = shape[axis];
    let lane_count: usize = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis)
        .map(|(_, &length)| length)
        .product();

    // The task is 1 lane, so a single-lane input stays serial
    if lane_count > 1 && lane_count * lane_len >= cheap_map_parallel_threshold() {
        Zip::from(grad_input.lanes_mut(Axis(axis)))
            .and(output.lanes(Axis(axis)))
            .and(grad_output.lanes(Axis(axis)))
            .par_for_each(compute_gradient);
    } else {
        Zip::from(grad_input.lanes_mut(Axis(axis)))
            .and(output.lanes(Axis(axis)))
            .and(grad_output.lanes(Axis(axis)))
            .for_each(compute_gradient);
    }

    Ok(grad_input)
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
        let output = softmax_forward(&input, DEFAULT_SOFTMAX_AXIS).expect("softmax_forward failed");
        let vals = output.as_slice().expect("not contiguous");
        assert_abs_diff_eq!(vals[0], 0.09003_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[1], 0.24473_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[2], 0.66524_f32, epsilon = 1e-4);
    }

    /// Softmax outputs sum to 1.0
    #[test]
    fn softmax_forward_sums_to_one() {
        let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
        let output = softmax_forward(&input, DEFAULT_SOFTMAX_AXIS).expect("softmax_forward failed");
        let sum: f32 = output.iter().sum();
        assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-6);
    }

    /// A row of equal large values stays numerically stable and produces a uniform distribution
    #[test]
    fn softmax_forward_large_equal_values_stable() {
        let input = tensor2(1, 3, vec![1000.0, 1000.0, 1000.0]);
        let output = softmax_forward(&input, DEFAULT_SOFTMAX_AXIS).expect("softmax_forward failed");
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
        let output = softmax_forward(&input, DEFAULT_SOFTMAX_AXIS).expect("softmax_forward failed");
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

        let output = softmax_forward(&transposed, DEFAULT_SOFTMAX_AXIS)
            .expect("softmax_forward must accept it");
        assert_eq!(output.shape(), &[3, 2]);

        let c_order: Tensor = transposed.as_standard_layout().into_owned();
        let want = softmax_forward(&c_order, DEFAULT_SOFTMAX_AXIS).expect("softmax_forward failed");
        for (got, expected) in output.iter().zip(want.iter()) {
            assert_abs_diff_eq!(*got, *expected, epsilon = 1e-6);
        }
    }

    /// A 1-D input normalizes its single axis
    #[test]
    fn softmax_forward_accepts_1d_input() {
        use ndarray::Array1;
        let input = Array1::from_vec(vec![0.0_f32, 1.0, 2.0]).into_dyn();
        let output =
            softmax_forward(&input, DEFAULT_SOFTMAX_AXIS).expect("1-D input must be accepted");
        assert_eq!(output.shape(), &[3]);
        let vals: Vec<f32> = output.iter().cloned().collect();
        assert_abs_diff_eq!(vals[0], 0.09003_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[1], 0.24473_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(vals[2], 0.66524_f32, epsilon = 1e-4);
    }

    /// A 0-D input has no axis to normalize, so every axis is out of bounds
    #[test]
    fn softmax_forward_rejects_0d_input() {
        let input: Tensor = ndarray::arr0(1.0_f32).into_dyn();
        assert!(
            matches!(
                softmax_forward(&input, DEFAULT_SOFTMAX_AXIS),
                Err(Error::InvalidInput(_))
            ),
            "0-D input must return InvalidInput"
        );
    }

    /// An axis outside the rank of the input fails the call, at both ends
    #[test]
    fn softmax_forward_rejects_out_of_range_axis() {
        let input = tensor2(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        for axis in [2, 7, -3, -5, i32::MIN, i32::MAX] {
            assert!(
                matches!(softmax_forward(&input, axis), Err(Error::InvalidInput(_))),
                "axis {axis} must be rejected for a rank-2 input"
            );
        }
    }

    /// The axis resolves against the rank of the input, on every call
    ///
    /// 1 value normalizes the last axis of a rank-2 input and the middle axis of a rank-3
    /// input. A resolved index held from an earlier call would reduce the wrong axis
    #[test]
    fn softmax_forward_resolves_a_negative_axis_at_call_time() {
        use ndarray::{Array, Ix3};

        let rank_2 = tensor2(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let last = softmax_forward(&rank_2, -1).expect("rank-2 forward");
        let same = softmax_forward(&rank_2, 1).expect("rank-2 forward");
        assert_eq!(
            last.iter().cloned().collect::<Vec<f32>>(),
            same.iter().cloned().collect::<Vec<f32>>(),
            "-1 and 1 name the same axis of a rank-2 input"
        );

        let rank_3: Tensor =
            Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f32 * 0.25)
                .into_dyn();
        let middle = softmax_forward(&rank_3, -2).expect("rank-3 forward");
        let by_index = softmax_forward(&rank_3, 1).expect("rank-3 forward");
        assert_eq!(
            middle.iter().cloned().collect::<Vec<f32>>(),
            by_index.iter().cloned().collect::<Vec<f32>>(),
            "-2 and 1 name the same axis of a rank-3 input"
        );

        // Every lane of the resolved axis sums to 1, and no other axis does
        let view = middle
            .view()
            .into_dimensionality::<Ix3>()
            .expect("the result is rank 3");
        for lane in view.lanes(Axis(1)) {
            assert_abs_diff_eq!(lane.sum(), 1.0_f32, epsilon = 1e-6);
        }
    }

    /// The lane path and the fold-to-2D path give the same values for the same lanes
    ///
    /// A softmax over axis 0 of a rank-2 input runs the lane path over strided lanes. The same
    /// values transposed, normalized over the last axis, run the fold path. The 2 results must
    /// agree, which is what makes the strided lane walk trustworthy
    #[test]
    fn softmax_forward_lane_path_agrees_with_the_fold_path() {
        use ndarray::IxDyn;

        let base = tensor2(3, 4, (0..12).map(|v| v as f32 * 0.3 - 1.5).collect());
        let by_lane = softmax_forward(&base, 0).expect("axis 0 forward");

        let transposed: Tensor = {
            let view = base.view().permuted_axes(IxDyn(&[1, 0]));
            let mut owned = Tensor::zeros(view.raw_dim());
            owned.assign(&view);
            owned
        };
        let by_fold = softmax_forward(&transposed, -1).expect("axis -1 forward");

        for (row, column) in (0..3).flat_map(|r| (0..4).map(move |c| (r, c))) {
            assert_abs_diff_eq!(
                by_lane[[row, column]],
                by_fold[[column, row]],
                epsilon = 1e-7
            );
        }
    }

    // softmax_backward

    /// Backward gradient matches the hand-computed Jacobian-vector product
    #[test]
    fn softmax_backward_jacobian_vector_product() {
        let output = tensor2(1, 3, vec![0.25, 0.25, 0.5]);
        let grad_output = tensor2(1, 3, vec![1.0, 0.0, 0.0]);
        let grad_input = softmax_backward(&output, &grad_output, DEFAULT_SOFTMAX_AXIS)
            .expect("softmax_backward failed");
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
        let grad_input = softmax_backward(&output, &grad_output, DEFAULT_SOFTMAX_AXIS)
            .expect("softmax_backward failed");
        let row_sum: f32 = grad_input.iter().sum();
        assert_abs_diff_eq!(row_sum, 0.0_f32, epsilon = 1e-6);
    }

    /// A grad_output that does not match the output shape is an error, not a panic
    ///
    /// The lane path pairs the 2 shapes with a `Zip`, which panics on a mismatch. The guard
    /// must run before that, so the caller gets `Error::ShapeMismatch`. The fold path takes
    /// the same guard, and this test pins both paths
    #[test]
    fn softmax_backward_rejects_a_mismatched_grad_output() {
        use ndarray::IxDyn;

        let output = Tensor::zeros(IxDyn(&[2, 3, 4]));
        let grad_output = Tensor::zeros(IxDyn(&[2, 3, 5]));

        // Axis 1 is not the last axis, so this call takes the lane path
        match softmax_backward(&output, &grad_output, 1) {
            Err(Error::ShapeMismatch { expected, found }) => {
                assert_eq!(expected, vec![2, 3, 4]);
                assert_eq!(found, vec![2, 3, 5]);
            }
            other => panic!("the lane path must return ShapeMismatch, got {other:?}"),
        }

        assert!(
            matches!(
                softmax_backward(&output, &grad_output, DEFAULT_SOFTMAX_AXIS),
                Err(Error::ShapeMismatch { .. })
            ),
            "the fold path must return ShapeMismatch as well"
        );
    }

    // Round-trip tests for the Activation enum's public API

    /// Activation::Softmax forward delegates to softmax_forward
    #[test]
    fn activation_softmax_forward_via_enum() {
        let input = tensor2(1, 3, vec![0.0, 1.0, 2.0]);
        let output = Activation::Softmax {
            axis: DEFAULT_SOFTMAX_AXIS,
        }
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
        let cache = ActivationCache::new(ActivationInput::Output, output);
        let grad_input = Activation::Softmax {
            axis: DEFAULT_SOFTMAX_AXIS,
        }
        .backward(&cache, &grad_output)
        .expect("Activation::Softmax backward failed");
        let vals = grad_input.as_slice().expect("not contiguous");
        assert_abs_diff_eq!(vals[0], 0.1875_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[1], -0.0625_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(vals[2], -0.125_f32, epsilon = 1e-6);
    }

    // The pinned float32 tables below confirm agreement with reference values, not just
    // that `forward` and `backward` match each other

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
                "{name}[{i}] at x = {}: got {g}, want {e}, tolerance {tol}",
                PROBES[i]
            );
        }
    }

    /// Runs `activation` over [`PROBES`] and checks both the output and the derivative
    ///
    /// An all-ones upstream gradient makes the backward result the derivative itself
    fn check_against_reference(name: &str, activation: Activation, fwd: &[f32], grad: &[f32]) {
        let input = tensor2(1, PROBES.len(), PROBES.to_vec());
        let (output, cache) = activation.forward_train(input).expect("forward failed");
        assert_pinned(&format!("{name} forward"), &output, fwd);

        let ones = tensor2(1, PROBES.len(), vec![1.0; PROBES.len()]);
        let derivative = activation.backward(&cache, &ones).expect("backward failed");
        assert_pinned(&format!("{name} backward"), &derivative, grad);
    }

    /// LeakyReLU with the layer's default slope of 0.3. The derivative at exactly 0 is 1,
    /// because the positive branch is `x >= 0`
    #[test]
    fn leaky_relu_matches_reference() {
        check_against_reference(
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

    /// ELU with the default alpha of 1.0
    #[test]
    fn elu_matches_reference() {
        check_against_reference(
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
    fn elu_half_alpha_matches_reference() {
        check_against_reference(
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
    fn selu_matches_reference() {
        check_against_reference(
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
    fn softplus_matches_reference() {
        check_against_reference(
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
    fn softsign_matches_reference() {
        check_against_reference(
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
    fn hard_sigmoid_matches_reference() {
        check_against_reference(
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
    fn exponential_matches_reference() {
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
        check_against_reference("Exponential", Activation::Exponential, &table, &table);
    }

    /// The softplus derivative survives the far negative tail
    ///
    /// At x = -40 the derivative is about 4.25e-18. The direct form `1 - e^-a` rounds to
    /// exactly 0, because `e^-a` is 1 at f32 precision. `exp_m1` keeps the value
    #[test]
    fn softplus_backward_keeps_the_far_negative_tail() {
        let input = tensor2(1, 1, vec![-40.0]);
        let (_, cache) = Activation::Softplus.forward_train(input).expect("forward");
        let ones = tensor2(1, 1, vec![1.0]);
        let derivative = Activation::Softplus
            .backward(&cache, &ones)
            .expect("backward");

        let got = derivative.iter().next().copied().expect("1 element");
        let expected = 4.248_354e-18_f32;
        assert!(
            (got - expected).abs() <= 1e-5 * expected,
            "softplus derivative at x = -40: got {got}, want {expected}"
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
            Activation::Softmax {
                axis: DEFAULT_SOFTMAX_AXIS,
            },
            Activation::LeakyReLU {
                negative_slope: 0.3,
            },
            Activation::ELU { alpha: 1.0 },
            Activation::SELU,
            Activation::Softplus,
            Activation::Softsign,
            Activation::HardSigmoid,
            Activation::Exponential,
            Activation::GELU { approximate: false },
            Activation::GELU { approximate: true },
            Activation::SiLU,
            Activation::Mish,
        ];
        for activation in usable {
            assert!(
                activation.validate().is_ok(),
                "{activation:?} must pass validation"
            );
        }
    }

    /// Exact GELU. The value at 0 is 0, and the derivative there is `Phi(0) = 0.5`
    #[test]
    fn gelu_matches_reference() {
        check_against_reference(
            "GELU",
            Activation::GELU { approximate: false },
            &[
                -1.433_257_9e-6,
                -0.004_049_694,
                -0.045_500_264,
                -0.158_655_25,
                -0.154_268_77,
                0.0,
                0.345_731_23,
                0.841_344_8,
                1.954_499_7,
                2.995_950_3,
                4.999_998_6,
            ],
            &[
                -7.146_946e-6,
                -0.011_945_647,
                -0.085_231_8,
                -0.083_315_47,
                0.132_504_88,
                0.5,
                0.867_495_1,
                1.083_315_5,
                1.085_231_8,
                1.011_945_6,
                1.000_007_1,
            ],
        );
    }

    /// The tanh approximation of GELU
    #[test]
    fn gelu_tanh_matches_reference() {
        check_against_reference(
            "GELU(approximate)",
            Activation::GELU { approximate: true },
            &[
                -2.291_796_2e-7,
                -0.003_637_392,
                -0.045_402_306,
                -0.158_808_01,
                -0.154_286,
                0.0,
                0.345_714,
                0.841_192,
                1.954_597_7,
                2.996_362_6,
                5.0,
            ],
            &[
                -1.546_362e-6,
                -0.011_584_167,
                -0.086_099_26,
                -0.082_964_084,
                0.132_630_1,
                0.5,
                0.867_369_9,
                1.082_964_1,
                1.086_099_3,
                1.011_584_2,
                1.000_001_5,
            ],
        );
    }

    /// SiLU. The derivative at 0 is `sigmoid(0) = 0.5`
    #[test]
    fn silu_matches_reference() {
        check_against_reference(
            "SiLU",
            Activation::SiLU,
            &[
                -0.033_464_255,
                -0.142_277_62,
                -0.238_405_84,
                -0.268_941_42,
                -0.188_770_33,
                0.0,
                0.311_229_67,
                0.731_058_6,
                1.761_594_2,
                2.857_722_4,
                4.966_535_6,
            ],
            &[
                -0.026_547_432,
                -0.088_104_11,
                -0.090_784_25,
                0.072_329_49,
                0.260_038_8,
                0.5,
                0.739_961_2,
                0.927_670_5,
                1.090_784_2,
                1.088_104_1,
                1.026_547_4,
            ],
        );
    }

    /// Mish. The derivative at 0 is `tanh(ln 2) = 0.6`
    #[test]
    fn mish_matches_reference() {
        check_against_reference(
            "Mish",
            Activation::Mish,
            &[
                -0.033_576_24,
                -0.145_647_46,
                -0.252_501_5,
                -0.303_401_46,
                -0.220_743_77,
                0.0,
                0.375_245_2,
                0.865_098_4,
                1.943_959,
                2.986_535,
                4.999_552,
            ],
            &[
                -0.026_747_498,
                -0.093_393_12,
                -0.108_355_09,
                0.059_216_756,
                0.289_510_68,
                0.6,
                0.886_424_4,
                1.049_036_2,
                1.069_317_9,
                1.021_107,
                1.000_800_2,
            ],
        );
    }

    /// The exact GELU keeps its relative precision in the negative tail
    ///
    /// At x = -12 the value is about -2.13e-32. The form `x * (1 + erf(x / sqrt(2))) / 2`
    /// gives exactly 0, because `erf` rounds to -1 at f32 precision
    #[test]
    fn gelu_keeps_the_negative_tail() {
        let input = tensor2(1, 1, vec![-12.0]);
        let activation = Activation::GELU { approximate: false };
        let (output, cache) = activation.forward_train(input).expect("forward");
        let derivative = activation
            .backward(&cache, &tensor2(1, 1, vec![1.0]))
            .expect("backward");

        for (name, got, want) in [
            ("value", output[[0, 0]], -2.131_778_5e-32_f32),
            ("derivative", derivative[[0, 0]], -2.557_895_7e-31_f32),
        ] {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs(),
                "GELU {name} at x = -12: got {got}, want {want}"
            );
        }
    }

    /// The derivative of the tanh approximation stays finite where the sigmoid saturates
    ///
    /// At `|x| = 1e30` the term `x * x` overflows to infinity. The derivative is 1 on the
    /// positive side and 0 on the negative side
    #[test]
    fn gelu_tanh_derivative_stays_finite_at_huge_inputs() {
        let activation = Activation::GELU { approximate: true };
        let input = tensor2(1, 4, vec![1e30, -1e30, f32::MAX, f32::MIN]);
        let (_, cache) = activation.forward_train(input).expect("forward");
        let derivative = activation
            .backward(&cache, &tensor2(1, 4, vec![1.0; 4]))
            .expect("backward");
        let got: Vec<f32> = derivative.iter().copied().collect();
        assert_eq!(got, vec![1.0, 0.0, 1.0, 0.0]);
    }

    /// The derivative of each pre-activation variant matches a central difference
    ///
    /// The difference runs in f64 over the f32 forward pass, so its error stays far below the
    /// tolerance
    #[test]
    fn pre_activation_variants_match_a_central_difference() {
        let points: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.125).collect();
        let step = 1e-2_f32;
        for activation in [
            Activation::GELU { approximate: false },
            Activation::GELU { approximate: true },
            Activation::SiLU,
            Activation::Mish,
        ] {
            let input = tensor2(1, points.len(), points.clone());
            let (_, cache) = activation.forward_train(input).expect("forward");
            let ones = tensor2(1, points.len(), vec![1.0; points.len()]);
            let derivative = activation.backward(&cache, &ones).expect("backward");

            let shifted = |delta: f32| {
                let moved = tensor2(1, points.len(), points.iter().map(|x| x + delta).collect());
                activation.forward(&moved).expect("forward")
            };
            let (above, below) = (shifted(step), shifted(-step));
            for (i, &x) in points.iter().enumerate() {
                let difference =
                    (above[[0, i]] as f64 - below[[0, i]] as f64) / (2.0 * step as f64);
                let got = derivative[[0, i]] as f64;
                assert!(
                    (got - difference).abs() <= 1e-3,
                    "{activation:?} at x = {x}: backward {got}, central difference {difference}"
                );
            }
        }
    }

    /// Each variant names the tensor that its backward pass reads
    #[test]
    fn only_gelu_silu_and_mish_read_the_pre_activation() {
        for activation in [
            Activation::GELU { approximate: false },
            Activation::GELU { approximate: true },
            Activation::SiLU,
            Activation::Mish,
        ] {
            assert_eq!(activation.saves(), ActivationInput::PreActivation);
        }
        for activation in [
            Activation::Linear,
            Activation::ReLU,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax {
                axis: DEFAULT_SOFTMAX_AXIS,
            },
            Activation::LeakyReLU {
                negative_slope: 0.3,
            },
            Activation::ELU { alpha: 1.0 },
            Activation::SELU,
            Activation::Softplus,
            Activation::Softsign,
            Activation::HardSigmoid,
            Activation::Exponential,
        ] {
            assert_eq!(activation.saves(), ActivationInput::Output);
        }
    }

    /// The backward pass refuses a cache that holds the other tensor
    ///
    /// A pre-activation read as an output, or an output read as a pre-activation, gives a
    /// wrong gradient with no other sign of the fault
    #[test]
    fn backward_refuses_a_cache_of_the_other_tensor() {
        let tensor = tensor2(1, 3, vec![-1.0, 0.0, 1.0]);
        let ones = tensor2(1, 3, vec![1.0; 3]);
        let output_cache = ActivationCache::new(ActivationInput::Output, tensor.clone());
        let input_cache = ActivationCache::new(ActivationInput::PreActivation, tensor);

        for (activation, cache) in [
            (Activation::SiLU, &output_cache),
            (Activation::Tanh, &input_cache),
        ] {
            assert!(
                matches!(
                    activation.backward(cache, &ones),
                    Err(Error::InvalidInput(_))
                ),
                "{activation:?} must refuse a {:?} cache",
                cache.holds()
            );
            assert!(matches!(
                activation.output_of(cache),
                Err(Error::InvalidInput(_))
            ));
        }
    }

    /// The backward pass refuses an upstream gradient of another shape, for every variant
    #[test]
    fn backward_refuses_a_gradient_of_another_shape() {
        let ones = tensor2(1, 4, vec![1.0; 4]);
        for activation in [Activation::ReLU, Activation::Mish] {
            let (_, cache) = activation
                .forward_train(tensor2(1, 3, vec![-1.0, 0.0, 1.0]))
                .expect("forward");
            assert!(matches!(
                activation.backward(&cache, &ones),
                Err(Error::ShapeMismatch { .. })
            ));
        }
    }

    /// `output_of` gives the forward output for both kinds of cache
    #[test]
    fn output_of_recovers_the_forward_output() {
        let input = tensor2(1, 4, vec![-2.0, -0.5, 0.5, 2.0]);
        for activation in [Activation::Tanh, Activation::GELU { approximate: false }] {
            let (output, cache) = activation.forward_train(input.clone()).expect("forward");
            assert_eq!(activation.output_of(&cache).expect("output_of"), output);
        }
    }
}
