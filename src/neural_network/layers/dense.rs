//! Dense (fully connected) layer: a linear transform followed by an optional activation

use crate::error::{Context, Error};
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::validation::{
    start_build, validate_optional_weight, validate_weight_shape,
};
use crate::neural_network::layers::{build_config_function, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Fans, Initializer, Shape, Tensor};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Activation as FusedActivation, Bias, Parallelism};
use ndarray::{Array, Array2, ArrayView2, Axis, CowArray, Ix2};

/// Dense (fully connected) layer for neural networks
///
/// Applies a linear transform with a weight matrix and a bias vector, then an optional
/// activation: `output = activation(input * weights + bias)`
///
/// [`with_use_bias(false)`](Dense::with_use_bias) drops the bias, and the layer then computes
/// `output = activation(input * weights)` and holds the kernel alone
///
/// The layer contracts the last axis only. The input shape is
/// `(batch_size, ..., input_dim)` with rank 2 or more, and the output shape is the same shape
/// with the last axis replaced by `output_dim`. The kernel stays `(input_dim, output_dim)`
/// for every rank, and every leading position shares it. Rank 3 is the common case for
/// sequence data, where the layer transforms each timestep with the same weights
///
/// Weights start from Xavier/Glorot initialization, and biases start at 0. During training,
/// the forward pass parks the values that the backward pass needs in the
/// [`Ctx`]
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::{Activation, Dense};
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::optimizers::SGD;
/// use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
///
/// // Create input and target tensors: input dim 4, output dim 3, batch_size 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = SequentialBuilder::new()
///     .add(Dense::new(3, Activation::ReLU).unwrap())
///     .add(Dense::new(1, Activation::ReLU).unwrap())
///     .build(&Shape::known(&[2, 4]))
///     .unwrap();
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
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::layers::{Activation, Dense};
/// use rustyml::neural_network::traits::UnaryLayer;
///
/// // 2 sequences of 5 timesteps, and 7 features for each timestep
/// let x = Array::ones((2, 5, 7)).into_dyn();
/// let mut layer = Dense::new(4, Activation::ReLU).unwrap();
///
/// let mut ctx = Ctx::training();
/// let output = layer.forward_mut(&x, &mut ctx).unwrap();
/// assert_eq!(output.shape(), &[2, 5, 4]);
/// ```
#[derive(Debug)]
pub struct Dense {
    /// Feature count of the last axis, which [`UnaryLayer::build`] reads from the input shape
    input_dim: usize,
    /// Shape the kernel depends on, which is `(None, input_dim)`. `None` before the build
    built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Output dimension size
    output_dim: usize,
    /// Weight matrix with shape (input_dim, output_dim)
    weights: Array2<f32>,
    /// Bias vector with shape (1, output_dim)
    ///
    /// The array stays allocated when `use_bias` is false, and nothing reads it in that case.
    /// The forward pass drops the bias epilogue, `weights` hides the array, and `parameters`
    /// never yields it, so a bias-free layer holds it and no more
    bias: Array2<f32>,
    /// Activation function applied to the linear output
    activation: Activation,
    /// Whether the layer adds a bias to the linear output
    use_bias: bool,
}

impl Dense {
    /// Creates a new dense layer with an activation function
    ///
    /// # Parameters
    ///
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
    /// The constructor draws nothing. [`UnaryLayer::build`] reads the feature count from the input
    /// shape and draws the kernel then. The draw takes the global seed or entropy by default.
    /// For reproducible initialization, set a seed with [`Dense::with_random_state`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `units` is zero
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(units: usize, activation: impl Into<Activation>) -> Result<Self, Error> {
        // Validate that dimensions are greater than zero
        if units == 0 {
            return Err(Error::invalid_parameter("units", "must be greater than 0"));
        }
        let activation = activation.into();
        activation.validate()?;

        Ok(Self {
            input_dim: 0,
            built: None,
            random_state: None,
            output_dim: units,
            weights: Array::zeros((0, 0)),
            bias: Array::zeros((0, 0)),
            activation,
            use_bias: true,
        })
    }

    /// Draws the kernel and zeroes the bias, at the extents the build settled
    ///
    /// The layer reports its own 2 fans. The kernel holds 1 weight per input and unit pair, so
    /// `fan_in` is the input width and `fan_out` is the unit count
    fn draw_parameters(&mut self) {
        let mut rng = crate::random::make_rng(self.random_state);
        self.weights = Initializer::GlorotUniform.draw(
            (self.input_dim, self.output_dim),
            Fans::new(self.input_dim, self.output_dim),
            &mut rng,
        );
        self.bias = Array::zeros((1, self.output_dim));
    }

    /// Sets the seed that the weight draw of [`UnaryLayer::build`] uses
    ///
    /// By default the draw takes the global seed or entropy (see [`crate::random`]). An unbuilt
    /// layer holds no kernel, so this records the seed and draws nothing. A layer that is
    /// already built draws its kernel again from the new seed, so the order of the 2 calls does
    /// not matter. The bias stays zero-initialized
    ///
    /// # Parameters
    ///
    /// - `random_state` - Seed for weight initialization
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        if self.built.is_some() {
            self.draw_parameters();
        }
        self
    }

    /// Sets whether the layer adds a bias to the linear output (defaults to `true`)
    ///
    /// With `use_bias` set to false the layer computes `activation(input * weights)`. It holds
    /// the kernel alone: `param_count` counts the kernel alone, `parameters` yields the kernel
    /// alone, and a checkpoint of the layer holds 1 array under the path `<position>.kernel`.
    /// A checkpoint written by a layer that has a bias therefore fails to load into a layer
    /// that has none, and the refusal names the path
    ///
    /// # Parameters
    ///
    /// - `use_bias` - `true` to add a bias, `false` to leave it out
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets the weights and bias for this layer
    ///
    /// # Parameters
    ///
    /// - `weights` - Weight matrix with shape (input_dim, output_dim)
    /// - `bias` - Bias vector with shape (1, output_dim), or `None` for a layer built with
    ///   [`with_use_bias(false)`](Dense::with_use_bias)
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - Ok when `weights` and `bias` match the layer's configured shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer holds no array yet
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `weights` or `bias` do not match the
    ///   layer's configured shape
    /// - `Error::InvalidParameter` - If a bias is given to a layer that holds none, or none is
    ///   given to a layer that holds one
    pub fn set_weights(
        &mut self,
        weights: Array2<f32>,
        bias: impl Into<Option<Array2<f32>>>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Dense"));
        }
        validate_weight_shape("kernel", self.weights.shape(), weights.shape())?;
        let bias = validate_optional_weight("bias", "use_bias", self.use_bias, bias.into())?;
        if let Some(bias) = bias.as_ref() {
            validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        }

        self.weights = weights.as_standard_layout().into_owned();
        if let Some(bias) = bias {
            self.bias = bias.as_standard_layout().into_owned();
        }
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
    /// [`UnaryLayer::forward`] and [`Layer::predict`]
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
        // A bias-free layer passes no epilogue at all, so the product is exactly the product.
        // A zero addend would give the same value for every input except a negative zero
        let bias = self
            .use_bias
            .then(|| self.bias.as_slice().expect("bias must be contiguous"));
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
            bias.map(Bias::PerCol),
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

/// What the forward pass of [`Dense`] parks for its backward pass
struct DenseCache {
    /// The input folded to `[rows, input_dim]`
    input: Array2<f32>,
    /// Shape of the tensor the forward pass received, to restore the rank of the gradient
    input_shape: Vec<usize>,
    /// The activated output, to backpropagate through the activation
    output: Tensor,
}

impl LayerBase for Dense {
    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so dropping the bias
        // corrects the count with no second formula to keep in step
        let bias = if self.use_bias { self.bias.len() } else { 0 };
        ParamCounts::trainable(self.weights.len() + bias)
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let Self {
            weights,
            bias,
            use_bias,
            ..
        } = self;
        let mut params = vec![ParamRef::weight(
            "kernel",
            weights.as_slice_mut().expect("weights must be contiguous"),
        )];
        if *use_bias {
            params.push(ParamRef::no_decay(
                "bias",
                bias.as_slice_mut().expect("bias must be contiguous"),
            ));
        }
        params
    }

    fn known_input_shapes(&self) -> Option<Vec<Shape>> {
        self.built.as_ref().map(|shape| vec![shape.free_batch()])
    }

    build_config_function!();

    named_weight_layer_functions!(
        trainable "kernel" => weights,
        trainable "bias" => bias if use_bias,
    );
}

impl UnaryLayer for Dense {
    /// Reads the feature count from the last axis, and draws the kernel and the bias
    ///
    /// The kernel is `(input_dim, units)` for every input rank, and every leading position
    /// shares it. The build shape therefore records the last axis alone, and the layer accepts
    /// any rank of 2 or more whose last axis matches
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        input.check_min_rank("Dense", 2)?;
        let Some(input_dim) = input.axes()[input.rank() - 1] else {
            return Err(Error::invalid_input(format!(
                "Dense needs a fixed extent on the last axis, and the shape {input} leaves that \
                 axis free"
            )));
        };
        if input_dim == 0 {
            return Err(Error::invalid_input(
                "Dense needs a positive extent on the last axis, got 0",
            ));
        }
        let canonical = Shape::new(vec![None, Some(input_dim)]);
        let Some(built) = start_build(&self.built, "Dense", &canonical)? else {
            return Ok(());
        };
        self.input_dim = input_dim;
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    /// Fuses the linear product, bias add, and (for `ReLU`) the activation into one gemmkit
    /// call. See `Dense::project` for the `NaN` handling of the fused `ReLU`
    ///
    /// The input has rank 2 or more. The leading axes fold into 1 row axis, so a rank-3 input
    /// costs the same 1 matrix product as a rank-2 input with the same number of rows
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("Dense"));
        }
        let input_2d = Self::fold(input, self.input_dim, "input")?;
        let output = self.project(&input_2d.view(), input.shape())?;

        if ctx.is_training() {
            ctx.push_cache(DenseCache {
                input: input_2d.into_owned(),
                input_shape: input.shape().to_vec(),
                output: output.clone(),
            });
        }

        Ok(output)
    }

    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let cache: DenseCache = ctx.pop_cache("Dense")?;

        // Upstream gradient must match the cached output shape
        if grad_output.shape() != cache.output.shape() {
            return Err(Error::shape_mismatch(
                cache.output.shape(),
                grad_output.shape(),
            ));
        }
        let grad_upstream = self.activation.backward(&cache.output, grad_output)?;

        // Fold the upstream gradient to [rows, output_dim]. Both operands of the weight
        // gradient are then 2D, and the bias gradient sums over 1 row axis that already
        // holds every leading position
        let grad_upstream_2d = Self::fold(&grad_upstream, self.output_dim, "gradient")?;

        let grad_w = dot(&cache.input.t(), &grad_upstream_2d);
        ctx.add_grad("kernel", grad_w.as_standard_layout().to_owned().into_dyn())?;

        // A bias-free layer computes no bias gradient, so the store holds none and no
        // optimizer state is ever keyed on one
        if self.use_bias {
            let grad_b = grad_upstream_2d.sum_axis(Axis(0)).insert_axis(Axis(0));
            ctx.add_grad("bias", grad_b.as_standard_layout().to_owned().into_dyn())?;
        }

        // Gradient with respect to the input, back at the rank of the cached input
        let grad_input = dot(&grad_upstream_2d, &self.weights.t());

        grad_input
            .into_dyn()
            .into_shape_with_order(cache.input_shape)
            .context("Failed to restore the rank of the Dense input gradient")
    }

    /// The last axis becomes the unit count, and every axis before it passes through
    ///
    /// The answer needs the unit count alone, so an unbuilt layer gives it. A built layer
    /// holds a kernel of a fixed width, and it refuses a last axis that the kernel cannot
    /// contract
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_min_rank("Dense", 2)?;
        let mut axes = input.axes().to_vec();
        let last = axes.len() - 1;
        if self.built.is_some() {
            match axes[last] {
                Some(extent) if extent == self.input_dim => {}
                Some(extent) => {
                    return Err(Error::invalid_input(format!(
                        "Dense input must have {} elements on the last axis, got {extent}",
                        self.input_dim
                    )));
                }
                None => {
                    return Err(Error::invalid_input(format!(
                        "Dense input must have {} elements on the last axis, and that axis is \
                         free",
                        self.input_dim
                    )));
                }
            }
        }
        axes[last] = Some(self.output_dim);
        Ok(Shape::new(axes))
    }
}
