//! SimpleRNN layer: a basic recurrent layer that returns the last hidden state, or every
//! timestep's hidden state

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::activation::Activation;
use crate::neural_network::layers::named_weight_layer_functions;
use crate::neural_network::layers::recurrent::gate::take_cache;
use crate::neural_network::layers::recurrent::input_step;
use crate::neural_network::layers::recurrent::validation::{
    split_grad_output, validate_dimension_greater_than_zero, validate_input_3d,
    validate_recurrent_dimensions,
};
use crate::neural_network::layers::validation::start_build;
use crate::neural_network::layers::validation::validate_weight_shape;
use crate::neural_network::layers::{build_config_function, build_on_forward};
use crate::neural_network::traits::{Layer, ParamGrad};
use crate::neural_network::{Fans, Initializer, Shape, Tensor};
use gemmkit_ndarray::dot;
use gemmkit_ndarray::{Activation as FusedActivation, Bias, Parallelism};
use ndarray::{Array, Array2, Array3, Axis};

/// Simple Recurrent Neural Network (SimpleRNN) layer
///
/// Processes a 3D input tensor with shape (batch_size, timesteps, input_dim) and returns
/// the last hidden state with shape (batch_size, units). It applies an activation from the
/// activation module at each timestep
///
/// [`SimpleRNN::with_return_sequences`] makes the layer return every timestep's hidden state,
/// with shape (batch_size, timesteps, units). [`SimpleRNN::with_go_backwards`] processes the
/// input timesteps from last to first.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array;
///
/// // Create input with batch_size=2, timesteps=5, input_dim=4,
/// // and target with batch_size=2, units=3 (same dimension as the last hidden state)
/// let x = Array::ones((2, 5, 4)).into_dyn();
/// let y = Array::ones((2, 3)).into_dyn();
///
/// // Build model: 1 SimpleRNN layer with Tanh activation
/// let mut model = SequentialBuilder::new()
///     .add(SimpleRNN::new(3, Activation::Tanh).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
/// // Print structure
/// model.summary();
///
/// // Train for 1 epoch
/// model.fit(&x, &y, 1).unwrap();
///
/// // Predict
/// let pred = model.predict(&x);
/// println!("SimpleRnn prediction:\n{:#?}\n", pred);
/// ```
#[derive(Debug)]
pub struct SimpleRNN {
    /// Feature count per timestep, which [`Layer::build`] reads from the input shape
    input_dim: usize,
    /// Shape the kernels depend on, which is `(None, None, input_dim)`. `None` before the
    /// build
    built: Option<Shape>,
    /// Seed of the weight draw, or `None` to take the global seed or entropy
    random_state: Option<u64>,
    /// Number of output units (neurons)
    units: usize,
    /// Weight matrix connecting inputs to the layer with shape (input_dim, units)
    kernel: Array2<f32>,
    /// Weight matrix connecting previous hidden states with shape (units, units)
    recurrent_kernel: Array2<f32>,
    /// Bias vector for the layer with shape (1, units)
    bias: Array2<f32>,
    /// Cached input tensor from the forward pass
    input_cache: Option<Array3<f32>>,
    /// Cached hidden states from the forward pass
    hidden_state_cache: Option<Vec<Array2<f32>>>,
    /// Gradient of the kernel weights
    grad_kernel: Option<Array2<f32>>,
    /// Gradient of the recurrent kernel weights
    grad_recurrent_kernel: Option<Array2<f32>>,
    /// Gradient of the bias
    grad_bias: Option<Array2<f32>>,
    /// Activation function applied at each timestep of the recurrence
    activation: Activation,
    /// Returns the full sequence of hidden states when true, or only the last one when false
    return_sequences: bool,
    /// Processes the input timesteps from last to first when true
    go_backwards: bool,
}

impl SimpleRNN {
    /// Creates a SimpleRNN layer with the specified dimensions and activation
    ///
    /// # Parameters
    ///
    /// - `input_dim` - Size of each input sample
    /// - `units` - Number of output units
    /// - `activation` - Activation function from the activation module (any [`Activation`]
    ///   variant, or any standalone activation layer)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new SimpleRNN layer instance
    ///
    /// # Notes
    ///
    /// By default, the constructor seeds weights from the global seed or entropy. Set a seed
    /// with [`SimpleRNN::with_random_state`] for reproducible initialization.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `units` is 0
    /// - `Error::InvalidParameter` - If the activation carries an unusable parameter (see
    ///   [`Activation::validate`])
    pub fn new(units: usize, activation: impl Into<Activation>) -> Result<Self, Error> {
        validate_dimension_greater_than_zero(units, "units")?;
        let activation = activation.into();
        activation.validate()?;

        Ok(SimpleRNN {
            input_dim: 0,
            built: None,
            random_state: None,
            units,
            kernel: Array::zeros((0, 0)),
            recurrent_kernel: Array::zeros((0, 0)),
            bias: Array::zeros((0, 0)),
            input_cache: None,
            hidden_state_cache: None,
            grad_kernel: None,
            grad_recurrent_kernel: None,
            grad_bias: None,
            activation,
            return_sequences: false,
            go_backwards: false,
        })
    }

    /// Sets the seed used to initialize the weights and re-initializes them deterministically
    ///
    /// By default, `SimpleRNN::new` seeds the weights from the global seed or entropy (see
    /// [`crate::random`]). This method re-runs the kernel (Xavier/Glorot) and recurrent-kernel
    /// (orthogonal) initialization with `random_state`, so call it before assigning custom
    /// weights or training. The bias stays zero-initialized.
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

    /// Sets whether the layer returns every timestep's hidden state
    ///
    /// The default is false, which returns only the last hidden state, with shape
    /// (batch_size, units). With true, the layer returns all hidden states, with shape
    /// (batch_size, timesteps, units). Slot `k` of the time axis holds the state after
    /// processing step `k`. The backward pass then expects a gradient of the same rank-3 shape.
    ///
    /// # Parameters
    ///
    /// - `return_sequences` - True to return every timestep's hidden state
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    ///
    /// # Notes
    ///
    /// The last slot of the returned sequence always equals the output of the same layer with
    /// `return_sequences` set to false.
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Sets whether the layer processes the input timesteps from last to first
    ///
    /// The default is false. With true, processing step 0 consumes input timestep
    /// `timesteps` - 1, and the output stays in processing order. The layer does not reverse the
    /// output back to input order, so slot 0 of a returned sequence holds the state that came
    /// from the last input timestep. This flag changes no shape.
    ///
    /// # Parameters
    ///
    /// - `go_backwards` - True to process the input timesteps from last to first
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_go_backwards(mut self, go_backwards: bool) -> Self {
        self.go_backwards = go_backwards;
        self
    }

    /// Initializes the input kernel (Xavier/Glorot) and recurrent kernel (orthogonal) from a seed
    ///
    /// Both draws share a single RNG, kernel first and then recurrent kernel, so a given seed
    /// reproduces the exact same pair of matrices. The order is part of the contract of this
    /// layer. A second generator, or the reverse order, changes the recurrent kernel
    fn draw_parameters(&mut self) {
        let mut rng = crate::random::make_rng(self.random_state);

        self.kernel = Initializer::GlorotUniform.draw(
            (self.input_dim, self.units),
            Fans::new(self.input_dim, self.units),
            &mut rng,
        );

        // Orthonormal columns keep the hidden-state transition norm-preserving
        self.recurrent_kernel =
            Initializer::Orthogonal.draw_orthogonal(self.units, Fans::NONE, &mut rng);

        self.bias = Array::zeros((1, self.units));
    }

    /// Sets the weights for this layer
    ///
    /// # Parameters
    ///
    /// - `kernel` - Weight matrix connecting inputs to the layer with shape (input_dim, units)
    /// - `recurrent_kernel` - Weight matrix connecting previous hidden states with shape
    ///   (units, units)
    /// - `bias` - Bias vector with shape (1, units)
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok(())` when every matrix matches the layer's existing shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If any supplied matrix does not match the
    ///   layer's existing shape
    pub fn set_weights(
        &mut self,
        kernel: Array2<f32>,
        recurrent_kernel: Array2<f32>,
        bias: Array2<f32>,
    ) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("SimpleRNN"));
        }
        validate_weight_shape("kernel", self.kernel.shape(), kernel.shape())?;
        validate_weight_shape(
            "recurrent_kernel",
            self.recurrent_kernel.shape(),
            recurrent_kernel.shape(),
        )?;
        validate_weight_shape("bias", self.bias.shape(), bias.shape())?;
        self.kernel = kernel;
        self.recurrent_kernel = recurrent_kernel;
        self.bias = bias;
        Ok(())
    }

    /// Batched input projection: `x3 [batch, timesteps, input_dim] @ kernel` for all timesteps in a
    /// single GEMM, returning `[batch, timesteps, units]`
    ///
    /// Collapsing the (batch, timesteps) axes into 1 matmul improves cache and SIMD use over
    /// `timesteps` separate small GEMMs.
    fn project_input(&self, x3: &ndarray::ArrayView3<f32>) -> Array3<f32> {
        crate::neural_network::layers::recurrent::gate::project_input(&self.kernel, x3)
    }

    /// Runs the recurrence and returns the layer output, the shared numeric body of
    /// [`Layer::forward`] and [`Layer::predict`]
    ///
    /// The output is the last hidden state, with shape (batch_size, units). With
    /// `return_sequences` set, it is instead every hidden state in processing order, with shape
    /// (batch_size, timesteps, units).
    ///
    /// When `hidden_states` is `Some`, this method records every hidden state, with `h_0 = 0`
    /// prepended, for the backward pass. `predict` passes `None` and skips both the recording
    /// and its clones. The record stays in processing order, so `hidden_states[k]` is the state
    /// that enters processing step `k`.
    ///
    /// A timestep needs 1 GEMM call and at most 1 activation sweep. The timestep buffer starts
    /// as the pre-projected `x_t @ kernel` slice. The recurrent product accumulates into it
    /// because `beta = 1`, and the bias add rides the same GEMM epilogue. This removes 2
    /// separate allocating broadcast adds that unfused code would need. `ReLU` also fuses into
    /// the backend's vectorized `Relu` epilogue, and `Linear` needs no pass at all. Every other
    /// activation instead runs as a separate vectorized [`Activation::forward`] pass. It runs
    /// this way because a per-element closure epilogue (`gemm_map`) would need 1 indirect
    /// scalar call per element.
    ///
    /// A fused `f32` epilogue matches the unfused product plus scalar activation bit for bit,
    /// with 1 exception. gemmkit's fused `Relu` maps `NaN` to `0`, while this crate's scalar
    /// closure propagates `NaN` instead. Outside that case, fusion changes no recorded hidden
    /// state.
    ///
    /// Each call uses gemmkit's automatic parallelism. A timestep product is
    /// `batch * units * units` of work. At small layer sizes, this sits below the backend's
    /// work gate and stays on the calling thread, so the tight loop pays no parallel dispatch.
    /// Wider layers cross the gate, and the backend spreads them itself.
    fn run(
        &self,
        x3: &ndarray::ArrayView3<f32>,
        mut hidden_states: Option<&mut Vec<Array2<f32>>>,
    ) -> Result<Tensor, Error> {
        let (batch, timesteps, _) = (x3.shape()[0], x3.shape()[1], x3.shape()[2]);
        let xw = self.project_input(x3);
        let bias = self.bias.as_slice().expect("bias must be contiguous");

        let mut sequence = if self.return_sequences {
            Some(Array3::<f32>::zeros((batch, timesteps, self.units)))
        } else {
            None
        };

        let mut h_prev = Array2::<f32>::zeros((batch, self.units));
        if let Some(hs) = hidden_states.as_deref_mut() {
            hs.push(h_prev.clone());
        }

        // An RNN requires sequential timestep processing
        for k in 0..timesteps {
            let t = input_step(k, timesteps, self.go_backwards);
            // z = x_t @ W + h_{t-1} @ U + b, with `x_t @ W` prefilled as the accumulator
            let mut z = xw.index_axis(Axis(1), t).to_owned();
            let fused_act = match self.activation {
                Activation::ReLU => Some(FusedActivation::Relu),
                _ => None,
            };
            gemmkit_ndarray::gemm_fused(
                1.0,
                &h_prev,
                &self.recurrent_kernel,
                1.0,
                &mut z,
                Some(Bias::PerCol(bias)),
                fused_act,
                Parallelism::Rayon(0),
            );
            let h_t = match self.activation {
                Activation::Linear | Activation::ReLU => z,
                _ => self
                    .activation
                    .forward(&z.into_dyn())?
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap(),
            };
            h_prev = h_t;
            if let Some(seq) = sequence.as_mut() {
                seq.index_axis_mut(Axis(1), k).assign(&h_prev);
            }
            if let Some(hs) = hidden_states.as_deref_mut() {
                hs.push(h_prev.clone());
            }
        }
        Ok(match sequence {
            Some(seq) => seq.into_dyn(),
            None => h_prev.into_dyn(),
        })
    }
}

impl Layer for SimpleRNN {
    /// Reads the feature count from the last axis, and draws both kernels and the bias
    ///
    /// 1 generator threads the input kernel and then the orthogonal recurrent kernel, in that
    /// order. A second generator, or the other order, changes every value of the second draw
    ///
    /// The kernels depend on the feature count and on the unit count, and on no other extent.
    /// The build shape therefore fixes the last axis alone, and the layer takes a batch of any
    /// size and a sequence of any length
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        input.check_rank("SimpleRNN", 3)?;
        let Some(input_dim) = input.axes()[2] else {
            return Err(Error::invalid_input(format!(
                "SimpleRNN needs a fixed feature count on axis 2, and the shape {input} leaves \
                 that axis free"
            )));
        };
        validate_recurrent_dimensions(input_dim, self.units)?;
        let canonical = Shape::new(vec![None, None, Some(input_dim)]);
        let Some(built) = start_build(&self.built, "SimpleRNN", &canonical)? else {
            return Ok(());
        };
        self.input_dim = input_dim;
        self.built = Some(built);
        self.draw_parameters();
        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        validate_input_3d(input)?;
        build_on_forward!(self, input);
        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        self.input_cache = Some(x3.to_owned());

        let mut hs = Vec::with_capacity(x3.shape()[1] + 1);
        let output = self.run(&x3, Some(&mut hs))?;
        self.hidden_state_cache = Some(hs);
        Ok(output)
    }

    /// Inference forward pass. Runs in eval mode and writes no caches. See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("SimpleRNN"));
        }
        validate_input_3d(input)?;
        let x3 = input.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        self.run(&x3, None)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let x3 = take_cache(&mut self.input_cache, "SimpleRNN")?;
        let hs = take_cache(&mut self.hidden_state_cache, "SimpleRNN")?;

        let batch = x3.shape()[0];
        let timesteps = x3.shape()[1];
        let feat = x3.shape()[2];

        // With `return_sequences`, every step also takes a direct contribution from `grad_seq`
        let (mut grad_h, grad_seq) = split_grad_output(
            grad_output,
            "SimpleRNN",
            self.return_sequences,
            batch,
            timesteps,
            self.units,
        )?;

        // Per-timestep d_z, stored so the input-side reductions can batch into single GEMMs
        let mut dz_all = Array3::<f32>::zeros((batch, timesteps, self.units));
        // backpropagation through time (BPTT)
        for k in (0..timesteps).rev() {
            // The direct contribution accumulates onto the carried gradient. It must land before
            // the activation backward, which consumes the total gradient of this step's state
            if let Some(seq) = grad_seq.as_ref() {
                grad_h += &seq.index_axis(Axis(1), k);
            }

            let d_z = {
                let h_t = hs[k + 1].clone().into_dyn();
                let grad_h_dyn = grad_h.clone().into_dyn();
                let grad_z_dyn = self.activation.backward(&h_t, &grad_h_dyn)?;
                grad_z_dyn.into_dimensionality::<ndarray::Ix2>().unwrap()
            };

            // gradient with respect to the previous hidden state, used by the next iteration
            // (sequential)
            grad_h = dot(&d_z, &self.recurrent_kernel.t());
            // The reductions below pair `d_z` with the input row it came from, so the scatter
            // uses the input timestep, not the processing step
            dz_all
                .index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards))
                .assign(&d_z);
        }

        // Batched reductions over all timesteps
        let dz_flat = dz_all
            .to_shape((batch * timesteps, self.units))
            .expect("contiguous DZ reshape");
        let x_flat = x3
            .to_shape((batch * timesteps, feat))
            .expect("contiguous input reshape");

        // `hs[k]` is the state that enters processing step `k`, and it pairs with the `d_z` of
        // that same step, which now sits at the step's input timestep
        let mut h_prev3 = Array3::<f32>::zeros((batch, timesteps, self.units));
        for (k, h) in hs.iter().take(timesteps).enumerate() {
            h_prev3
                .index_axis_mut(Axis(1), input_step(k, timesteps, self.go_backwards))
                .assign(h);
        }
        let h_prev_flat = h_prev3
            .to_shape((batch * timesteps, self.units))
            .expect("contiguous H_prev reshape");

        // Each reduction is the whole per-call gradient, so it becomes the buffer directly rather
        // than being added into a freshly zeroed one
        let grad_k = dot(&x_flat.t(), &dz_flat);
        let grad_rk = dot(&h_prev_flat.t(), &dz_flat);
        let grad_b = dz_flat.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Layout-tolerant reshape
        let grad_x3 = crate::neural_network::layers::recurrent::gate::reshape_2d_to_3d(
            dot(&dz_flat, &self.kernel.t()),
            (batch, timesteps, feat),
        );

        self.grad_kernel = Some(grad_k);
        self.grad_recurrent_kernel = Some(grad_rk);
        self.grad_bias = Some(grad_b);

        Ok(grad_x3.into_dyn())
    }

    fn layer_type(&self) -> &str {
        "SimpleRNN"
    }

    build_config_function!();

    fn known_input_shape(&self) -> Option<Shape> {
        // The layer keeps no input shape. It knows the feature count of 1 timestep, and it
        // serves every batch size and every sequence length, so both of those axes are free
        self.built.clone()
    }

    /// A returned sequence keeps the time axis, and a returned final state drops it
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        input.check_rank("SimpleRNN", 3)?;
        let axes = input.axes();
        // The unit count settles the answer, so an unbuilt layer gives it. A built layer holds
        // a kernel of a fixed width, and it refuses a feature count that the kernel cannot take
        if self.built.is_some()
            && let Some(features) = axes[2]
            && features != self.input_dim
        {
            return Err(Error::invalid_input(format!(
                "SimpleRNN expects {} features per timestep, got {features}",
                self.input_dim
            )));
        }
        Ok(if self.return_sequences {
            Shape::new(vec![axes[0], axes[1], Some(self.units)])
        } else {
            Shape::new(vec![axes[0], Some(self.units)])
        })
    }

    fn param_count(&self) -> ParamCounts {
        // Read the arrays the layer holds rather than the configuration, so a change to
        // the roster corrects the count with no second formula to keep in step
        ParamCounts::trainable(self.kernel.len() + self.recurrent_kernel.len() + self.bias.len())
    }

    fn parameters(&mut self) -> Vec<ParamGrad<'_>> {
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
        // Each tensor is pushed on its own, so a tensor without a gradient holds back no other
        if let Some(grad) = grad_kernel.as_ref() {
            params.push(ParamGrad::weight(
                "kernel",
                kernel.as_slice_mut().expect("kernel must be contiguous"),
                grad.as_slice().expect("kernel gradient must be contiguous"),
            ));
        }
        if let Some(grad) = grad_recurrent_kernel.as_ref() {
            params.push(ParamGrad::weight(
                "recurrent_kernel",
                recurrent_kernel
                    .as_slice_mut()
                    .expect("recurrent kernel must be contiguous"),
                grad.as_slice()
                    .expect("recurrent kernel gradient must be contiguous"),
            ));
        }
        if let Some(grad) = grad_bias.as_ref() {
            params.push(ParamGrad::no_decay(
                "bias",
                bias.as_slice_mut().expect("bias must be contiguous"),
                grad.as_slice().expect("bias gradient must be contiguous"),
            ));
        }
        params
    }

    named_weight_layer_functions!(
        trainable "kernel" => kernel,
        trainable "recurrent_kernel" => recurrent_kernel,
        trainable "bias" => bias,
    );
}
