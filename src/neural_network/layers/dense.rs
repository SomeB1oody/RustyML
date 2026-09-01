//! Dense (fully connected) layer: a linear transform followed by an optional activation

use crate::error::{Context, Error};
use crate::neural_network::Tensor;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::layer_weight::{DenseLayerWeight, LayerWeight};
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::traits::{Layer, ParamGrad};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Activation as FusedActivation, Bias, Parallelism};
use ndarray::{Array, Array2, ArrayView2, Axis, CowArray, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::borrow::Cow;

/// Dense (fully connected) layer for neural networks
///
/// Applies a linear transform with a weight matrix and a bias vector, then an optional
/// activation: `output = activation(input * weights + bias)`
///
/// The layer contracts the last axis only. The input shape is
/// `(batch_size, ..., input_dim)` with rank 2 or more, and the output shape is the same shape
/// with the last axis replaced by `output_dim`. The kernel stays `(input_dim, output_dim)`
/// for every rank, and every leading position shares it. Rank 3 is the common case for
/// sequence data, where the layer transforms each timestep with the same weights
///
/// Weights start from Xavier/Glorot initialization, and biases start at 0. During training,
/// the layer caches intermediate values for the backward pass
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::{Activation, Dense};
/// use rustyml::neural_network::optimizers::SGD;
/// use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
///
/// // Create input and target tensors: input dim 4, output dim 3, batch_size 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 3, Activation::ReLU).unwrap())
///     .add(Dense::new(3, 1, Activation::ReLU).unwrap());
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Print model structure
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 3).unwrap();
///
/// // Run forward prediction
/// let prediction = model.predict(&x);
/// println!("Prediction results: {:?}", prediction);
/// ```
///
/// An input of rank 3 or more keeps 1 kernel for every leading position:
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::layers::{Activation, Dense};
/// use rustyml::neural_network::traits::Layer;
///
/// // 2 sequences of 5 timesteps, and 7 features for each timestep
/// let x = Array::ones((2, 5, 7)).into_dyn();
/// let mut layer = Dense::new(7, 4, Activation::ReLU).unwrap();
///
/// let output = layer.forward(&x).unwrap();
/// assert_eq!(output.shape(), &[2, 5, 4]);
/// ```
#[derive(Debug)]
pub struct Dense {
    /// Input dimension size
    input_dim: usize,
    /// Output dimension size
    output_dim: usize,
    /// Weight matrix with shape (input_dim, output_dim)
    weights: Array2<f32>,
    /// Bias vector with shape (1, output_dim)
    bias: Array2<f32>,
    /// Cache of the folded input `[rows, input_dim]` from the forward pass
    input_cache: Option<Array2<f32>>,
    /// Shape of the last input that the forward pass received
    ///
    /// The backward pass restores the rank of the input gradient from it, and
    /// [`Layer::output_shape`] reports the real output rank from it
    input_shape: Option<Vec<usize>>,
    /// Cache of the activated output, used to backprop through the activation
    output_cache: Option<Tensor>,
    /// Stored weight gradients
    grad_weights: Option<Array2<f32>>,
    /// Stored bias gradients
    grad_bias: Option<Array2<f32>>,
    /// Activation function applied to the linear output
    activation: Activation,
}

impl Dense {
    /// Creates a new dense layer with an activation function
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Dimensionality of input features (number of features per timestep)
    /// - `units` - Number of units/neurons in the layer (determines output dimensionality)
    /// - `activation` - Activation applied to the linear output (any value convertible into
    ///   [`Activation`], e.g. `Activation::ReLU` or a standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - New `Dense` layer instance with initialized parameters
    ///
    /// # Notes
    ///
    /// Weights are seeded from the global seed or entropy by default. For reproducible
    /// initialization, set a seed with [`Dense::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `input_dim` or `units` is zero
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(
        input_dim: usize,
        units: usize,
        activation: impl Into<Activation>,
    ) -> Result<Self, Error> {
        // Validate that dimensions are greater than zero
        if input_dim == 0 {
            return Err(Error::invalid_parameter(
                "input_dim",
                "must be greater than 0",
            ));
        }
        if units == 0 {
            return Err(Error::invalid_parameter("units", "must be greater than 0"));
        }
        let activation = activation.into();
        activation.validate()?;

        Ok(Self {
            input_dim,
            output_dim: units,
            weights: Self::init_weights_array(input_dim, units, None),
            bias: Array::zeros((1, units)),
            input_cache: None,
            input_shape: None,
            output_cache: None,
            grad_weights: None,
            grad_bias: None,
            activation,
        })
    }

    /// Sets the seed used to initialize the weights and re-initializes them deterministically
    ///
    /// By default the weights are seeded from the global seed or entropy (see [`crate::random`]).
    /// This re-runs Xavier/Glorot uniform initialization with `random_state`, so call it before
    /// assigning custom weights or training. The bias stays zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.weights =
            Self::init_weights_array(self.input_dim, self.output_dim, Some(random_state));
        self
    }

    /// Xavier/Glorot uniform weight initialization for the given dimensions and seed
    fn init_weights_array(
        input_dim: usize,
        units: usize,
        random_state: Option<u64>,
    ) -> Array2<f32> {
        let limit = (6.0 / (input_dim + units) as f32).sqrt();
        let mut rng = crate::random::make_rng(random_state);
        Array::random_using(
            (input_dim, units),
            Uniform::new(-limit, limit).unwrap(),
            &mut rng,
        )
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - Weight matrix with shape (input_dim, output_dim)
    /// - `bias` - Bias vector with shape (1, output_dim)
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - Ok when `weights` and `bias` match the layer's configured shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` do not match the
    ///   layer's configured shape
    pub fn set_weights(&mut self, weights: Array2<f32>, bias: Array2<f32>) -> Result<(), Error> {
        validate_weight_shape("weight", self.weights.shape(), weights.shape())?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;

        self.weights = weights.as_standard_layout().into_owned();
        self.bias = bias.as_standard_layout().into_owned();
        Ok(())
    }

    /// Folds a tensor of rank 2 or more into a `[rows, columns]` matrix
    ///
    /// The layer contracts the last axis only, so every leading axis is a batch axis. `rows`
    /// is the product of all the leading axes. The fold lets 1 matrix product serve an input
    /// of any rank, and it makes each leading position share the same kernel. It also turns a
    /// sum over every leading axis into a sum over the folded row axis
    ///
    /// The result borrows the tensor when the tensor is reshapeable in place. It holds a
    /// C-order copy in the other case
    ///
    /// # Parameters
    ///
    /// - `tensor` - Tensor with rank 2 or more, whose last axis has `columns` elements
    /// - `columns` - Expected length of the last axis
    /// - `role` - Name of the tensor, used in the error messages
    ///
    /// # Returns
    ///
    /// - `Result<CowArray<f32, Ix2>, Error>` - The folded matrix
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - The rank is less than 2, or the last axis does not have
    ///   `columns` elements
    /// - `Error::Computation` - The fold failed
    fn fold<'a>(
        tensor: &'a Tensor,
        columns: usize,
        role: &str,
    ) -> Result<CowArray<'a, f32, Ix2>, Error> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(Error::invalid_input(format!(
                "Dense {role} must have rank 2 or more, got shape {shape:?}"
            )));
        }
        if shape[shape.len() - 1] != columns {
            return Err(Error::invalid_input(format!(
                "Dense {role} must have {columns} elements on the last axis, got shape {shape:?}"
            )));
        }

        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        tensor
            .to_shape((rows, columns))
            .context("Failed to fold the leading axes of a Dense tensor")
    }

    /// The layer's full forward transform, `activation(input * weights + bias)`. Shared by
    /// [`Layer::forward`] and [`Layer::predict`]
    ///
    /// The bias add rides the GEMM epilogue, so the pre-activation is written exactly once,
    /// with no separate broadcast add of the bias. The bias is the per-column addend, so it
    /// lowers to [`Bias::PerCol`]
    ///
    /// The activation stays a fused epilogue only where the backend has a vectorized one,
    /// `ReLU` ([`FusedActivation::Relu`]). `Linear` needs no separate pass, since it changes
    /// nothing. Every other activation runs as a separate [`Activation::forward`] pass
    ///
    /// A fused `f32` epilogue matches the unfused product plus scalar activation bit for bit,
    /// with 1 exception. gemmkit's fused `Relu` maps `NaN` to `0.0`, while this crate's
    /// scalar `relu` propagates `NaN`. A `NaN` pre-activation already means a diverged model,
    /// and the cached-output ReLU derivative then treats the resulting `0.0` as a dead unit
    ///
    /// The input arrives folded, and the result goes back to the rank of `input_shape`. A
    /// fold keeps each last-axis lane whole, so the activation gives the same result on the
    /// folded matrix as on the restored tensor. This holds for `Softmax` because an embedded
    /// `Softmax` must carry the last axis. [`Activation::validate`] rejects any other axis
    ///
    /// # Parameters
    ///
    /// - `input` - Folded input view with shape `[rows, input_dim]`. Any stride works, since
    ///   gemmkit reads the operand in place rather than copying it
    /// - `input_shape` - Shape of the tensor that `input` was folded from
    ///
    /// # Returns
    ///
    /// - `Result<Tensor, Error>` - Activated output. It has the shape of `input_shape` with
    ///   the last axis replaced by `output_dim`
    ///
    /// # Panics
    ///
    /// - If the input column count differs from `input_dim`
    /// - If the bias is not contiguous (it is always stored in standard layout)
    /// - If `input_shape` is empty
    ///
    /// # Errors
    ///
    /// - `Error::Computation` - Softmax failed to reshape the fused pre-activation, or the
    ///   result failed to go back to the rank of `input_shape`
    fn project(&self, input: &ArrayView2<'_, f32>, input_shape: &[usize]) -> Result<Tensor, Error> {
        let bias = self.bias.as_slice().expect("bias must be contiguous");
        // `beta == 0` means the fill value is never read. This only allocates the destination
        // the epilogue writes into
        let mut output = Array2::from_elem((input.nrows(), self.output_dim), 0.0);
        let fused_act = match self.activation {
            Activation::ReLU => Some(FusedActivation::Relu),
            _ => None,
        };
        gemmkit_ndarray::gemm_fused(
            1.0,
            input,
            &self.weights,
            0.0,
            &mut output,
            Some(Bias::PerCol(bias)),
            fused_act,
            Parallelism::Rayon(0),
        );

        let output = output.into_dyn();
        let activated = match self.activation {
            Activation::Linear | Activation::ReLU => output,
            _ => self.activation.forward(&output)?,
        };

        // The fused product writes C order, and so does every activation, so this call only
        // relabels the axes. It never copies
        let mut output_shape = input_shape.to_vec();
        let last_axis = output_shape.len() - 1;
        output_shape[last_axis] = self.output_dim;
        activated
            .into_shape_with_order(output_shape)
            .context("Failed to restore the rank of the Dense output")
    }
}

impl Layer for Dense {
    /// Training forward: caches the input and the activated output for the backward pass
    ///
    /// Fuses the linear product, bias add, and (for `ReLU`) the activation into one gemmkit
    /// call. See `Dense::project` for the `NaN` handling of the fused `ReLU`
    ///
    /// The input has rank 2 or more. The leading axes fold into 1 row axis, so a rank-3 input
    /// costs the same 1 matrix product as a rank-2 input with the same number of rows
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        let input_2d = Self::fold(input, self.input_dim, "input")?;

        // Fused linear + bias + activation, then cache the activated output for backpropagation
        let output = self.project(&input_2d.view(), input.shape())?;
        self.output_cache = Some(output.clone());

        // Cache the folded input [rows, input_dim] and the input shape for the backward pass
        self.input_shape = Some(input.shape().to_vec());
        self.input_cache = Some(input_2d.into_owned());

        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). Same fused projection as
    /// [`forward`](Layer::forward). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_2d = Self::fold(input, self.input_dim, "input")?;

        self.project(&input_2d.view(), input.shape())
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        // Backprop through the activation using the cached activated output
        let activated = self
            .output_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Dense"))?;
        // Upstream gradient must match the cached output shape
        if grad_output.shape() != activated.shape() {
            return Err(Error::shape_mismatch(
                activated.shape(),
                grad_output.shape(),
            ));
        }
        let grad_upstream = self.activation.backward(&activated, grad_output)?;

        // Fold the upstream gradient to [rows, output_dim]. Both operands of the weight
        // gradient are then 2D, and the bias gradient sums over 1 row axis that already
        // holds every leading position
        let grad_upstream_2d = Self::fold(&grad_upstream, self.output_dim, "gradient")?;

        let input = self
            .input_cache
            .take()
            .ok_or_else(|| Error::forward_pass_not_run("Dense"))?;
        let input_shape = self
            .input_shape
            .clone()
            .ok_or_else(|| Error::forward_pass_not_run("Dense"))?;

        // Weight gradients
        let grad_w = dot(&input.t(), &grad_upstream_2d);

        // Bias gradients: sum over every axis except the last one
        let grad_b = grad_upstream_2d.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Store gradients in a contiguous layout for `parameters()`
        self.grad_weights = Some(grad_w.as_standard_layout().to_owned());
        self.grad_bias = Some(grad_b.as_standard_layout().to_owned());

        // Gradient with respect to the input, back at the rank of the cached input
        let grad_input = dot(&grad_upstream_2d, &self.weights.t());

        grad_input
            .into_dyn()
            .into_shape_with_order(input_shape)
            .context("Failed to restore the rank of the Dense input gradient")
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn output_shape(&self) -> String {
        // The last axis becomes the unit count, and the axes between the batch axis and the
        // last axis pass through. Before the first forward pass, only the unit count is known
        match &self.input_shape {
            Some(shape) => {
                // Element 0 is the batch axis, which `summary()` prints as "None"
                let mut axes: Vec<String> = shape[1..shape.len() - 1]
                    .iter()
                    .map(|e| e.to_string())
                    .collect();
                axes.push(self.output_dim.to_string());
                format!("(None, {})", axes.join(", "))
            }
            None => format!("(None, {})", self.output_dim),
        }
    }

    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(self.input_dim * self.output_dim + self.output_dim)
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
        let Self {
            weights,
            bias,
            grad_weights,
            grad_bias,
            ..
        } = self;
        let mut params = Vec::new();
        // Each tensor is pushed on its own, so a tensor without a gradient holds back no other
        if let Some(grad) = grad_weights.as_ref() {
            params.push(ParamGrad::weight(
                "kernel",
                weights.as_slice_mut().expect("weights must be contiguous"),
                grad.as_slice().expect("grad_weights must be contiguous"),
            ));
        }
        if let Some(grad) = grad_bias.as_ref() {
            params.push(ParamGrad::no_decay(
                "bias",
                bias.as_slice_mut().expect("bias must be contiguous"),
                grad.as_slice().expect("grad_bias must be contiguous"),
            ));
        }
        params
    }

    fn get_weights(&self) -> LayerWeight<'_> {
        LayerWeight::Dense(DenseLayerWeight {
            weight: Cow::Borrowed(&self.weights),
            bias: Cow::Borrowed(&self.bias),
        })
    }
}
