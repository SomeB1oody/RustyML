use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::layer_weight::LayerWeight;
use crate::neural_network::layers::pooling::layer_functions_2d_pooling;
use crate::neural_network::layers::pooling::pooling_engine::{
    PoolKind, windowed_pool_backward, windowed_pool_forward,
};
use crate::neural_network::layers::pooling::validation::{
    validate_all_dims_positive, validate_input_shape_dims, validate_pool_size_2d,
    validate_strides_2d,
};
use crate::neural_network::layers::shape_helpers::calculate_output_shape_2d_pooling;
use crate::neural_network::traits::Layer;

/// 2D max pooling layer.
///
/// Selects the maximum value within each pooling window across height and width.
/// Input tensor shape: `[batch_size, channels, height, width]`. Output tensor shape:
/// `[batch_size, channels, pooled_height, pooled_width]` where
/// `pooled_height = (height - pool_size_h) / stride_h + 1` and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`.
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window as (height, width)
/// - `strides` - Step size of the pooling operation as (vertical stride, horizontal stride)
/// - `input_shape` - Shape of the input tensor declared at construction time
/// - `forward_input_shape` - Shape of the most recent forward input, cached for backpropagation
/// - `argmax` - Cached flat per-output arg-max indices used for backpropagation
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
///
/// // Create a simple 4D input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 input channels, 6x6 pixels
/// let mut input_data = Array4::zeros((2, 3, 6, 6));
///
/// // Set some specific values so we can predict the max pooling result
/// for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..6 {
///             for j in 0..6 {
///                 // Create input data with an easily observable pattern
///                 input_data[[b, c, i, j]] = (i * j) as f32 + b as f32 * 0.1 + c as f32 * 0.01;
///             }
///         }
///     }
/// }
///
/// let x = input_data.clone().into_dyn();
///
/// // Test using MaxPooling2D in a model
/// let mut model = Sequential::new();
/// model
///     .add(MaxPooling2D::new(
///         (2, 2),           // Pool window size
///         vec![2, 3, 6, 6], // Input shape
///         None,             // Use default stride (2,2)
///     ).unwrap())
///     .compile(RMSprop::new(0.001, 0.9, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Create target tensor - corresponding to the pooled shape
/// let y = Array4::ones((2, 3, 3, 3)).into_dyn();
///
/// // Print model structure
/// model.summary();
///
/// // Train the model (run a few epochs)
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x).unwrap();
/// println!("MaxPooling2D prediction results: {:?}", prediction);
///
/// // Check if output shape is correct
/// assert_eq!(prediction.shape(), &[2, 3, 3, 3]);
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`.
#[derive(Debug)]
pub struct MaxPooling2D {
    pool_size: (usize, usize),
    strides: (usize, usize),
    input_shape: Vec<usize>,
    forward_input_shape: Option<Vec<usize>>,
    argmax: Option<Vec<usize>>,
}

impl MaxPooling2D {
    /// Creates a new 2D max pooling layer.
    ///
    /// If `strides` is None, it defaults to `pool_size`.
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (height, width)
    /// - `input_shape` - Input tensor shape `[batch_size, channels, height, width]`
    /// - `strides` - Optional strides of the pooling operation as (vertical stride, horizontal stride)
    ///
    /// # Returns
    ///
    /// - `Result<MaxPooling2D, Error>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If `input_shape` is not 4D
    /// - `Error::InvalidParameter` - If `pool_size` has a zero dimension or any stride is zero
    pub fn new(
        pool_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize)>,
    ) -> Result<Self, Error> {
        let strides = strides.unwrap_or(pool_size);

        // input validation
        validate_input_shape_dims(&input_shape, 4, "MaxPooling2D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_2d(pool_size, input_shape[2], input_shape[3])?;
        validate_strides_2d(strides)?;

        Ok(MaxPooling2D {
            pool_size,
            strides,
            input_shape,
            forward_input_shape: None,
            argmax: None,
        })
    }
}

impl Layer for MaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Cache the actual input shape and arg-max positions for the backward pass
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, argmax) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Max,
        );
        self.argmax = argmax;
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`].
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Validate input is 4D
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Max,
        );
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let input_shape = self
            .forward_input_shape
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("MaxPooling2D"))?;
        let argmax = self
            .argmax
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("MaxPooling2D"))?;

        Ok(windowed_pool_backward(
            grad_output,
            input_shape,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Max,
            Some(argmax),
        ))
    }

    fn layer_type(&self) -> &str {
        "MaxPooling2D"
    }

    layer_functions_2d_pooling!();
}
