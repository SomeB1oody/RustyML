//! 2D average pooling layer that downsamples spatial dimensions by averaging over each window

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::TrainingParameters;
use crate::neural_network::layers::convolution::PaddingType;
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

/// 2D average pooling layer
///
/// Computes the mean value over each pooling window along the height and width dimensions
/// Input tensor shape: `[batch_size, channels, height, width]`. Output tensor shape:
/// `[batch_size, channels, pooled_height, pooled_width]` where
/// `pooled_height = (height - pool_size_h) / stride_h + 1` and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::sequential::Sequential;
/// use rustyml::neural_network::layers::*;
/// use rustyml::neural_network::optimizers::*;
/// use rustyml::neural_network::losses::*;
/// use ndarray::Array4;
/// use approx::assert_relative_eq;
///
/// // Create a simple input tensor: [batch_size, channels, height, width]
/// // Batch size=2, 3 input channels, each channel is 4x4 pixels
/// let mut input_data = Array4::zeros((2, 3, 4, 4));
///
///  // Set test data to make average pooling results predictable
///  for b in 0..2 {
///     for c in 0..3 {
///         for i in 0..4 {
///             for j in 0..4 {
///                 input_data[[b, c, i, j]] = (i + j) as f32;
///             }
///         }
///     }
///  }
///
///  let x = input_data.clone().into_dyn();
///
///  // Test AveragePooling with Sequential model
///  let mut model = Sequential::new();
///  model
///  .add(AveragePooling2D::new(
///  (2, 2),           // Pooling window size
///  vec![2, 3, 4, 4], // Input shape
///  Some((2, 2)),     // Strides (optional, defaults to pool_size if None)
///  PaddingType::Valid,
///  ).unwrap())
///  .compile(RMSprop::new(0.001, 0.9, 1e-8, None, 0.0).unwrap(), MeanSquaredError::new());
///
///  // Output shape should be [2, 3, 2, 2]
///  let output = model.predict(&x).unwrap();
///  assert_eq!(output.shape(), &[2, 3, 2, 2]);
///
///  // Verify correctness of pooling results
///  // For a 2x2 window with stride 2, we expect the result to be the average of the elements in the window
///  for b in 0..2 {
///     for c in 0..3 {
///         // First window (0,0), (0,1), (1,0), (1,1) -> average should be (0+1+1+2)/4 = 1.0
///         assert_relative_eq!(output[[b, c, 0, 0]], 1.0);
///         // Second window (0,2), (0,3), (1,2), (1,3) -> average should be (2+3+3+4)/4 = 3.0
///         assert_relative_eq!(output[[b, c, 0, 1]], 3.0);
///         // Third window (2,0), (2,1), (3,0), (3,1) -> average should be (2+3+3+4)/4 = 3.0
///         assert_relative_eq!(output[[b, c, 1, 0]], 3.0);
///         // Fourth window (2,2), (2,3), (3,2), (3,3) -> average should be (4+5+5+6)/4 = 5.0
///         assert_relative_eq!(output[[b, c, 1, 1]], 5.0);
///     }
///  }
/// ```
///
/// # Performance
///
/// Parallel execution is used when `batch_size * channels >= 32`
#[derive(Debug)]
pub struct AveragePooling2D {
    /// Size of the pooling window as (height, width)
    pool_size: (usize, usize),
    /// Step size of the pooling operation as (height, width)
    strides: (usize, usize),
    /// Shape of the input tensor declared at construction time
    input_shape: Vec<usize>,
    /// Padding mode applied around the input before pooling
    padding: PaddingType,
    /// Shape of the most recent forward input, cached for backpropagation
    forward_input_shape: Option<Vec<usize>>,
}

impl AveragePooling2D {
    /// Creates a new 2D average pooling layer
    ///
    /// If `strides` is None, it defaults to `pool_size`
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (height, width)
    /// - `input_shape` - Input tensor shape `[batch_size, channels, height, width]`
    /// - `strides` - Optional strides of the pooling operation as (height, width)
    ///
    /// # Returns
    ///
    /// - `Result<AveragePooling2D, Error>` - New layer instance on success
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `input_shape` is not 4D
    /// - [`Error::InvalidInput`] if any `input_shape` dimension is zero
    /// - [`Error::InvalidParameter`] if `pool_size` has a zero dimension or exceeds the input
    ///   spatial size, or any stride is zero
    pub fn new(
        pool_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize)>,
        padding: PaddingType,
    ) -> Result<Self, Error> {
        let strides = strides.unwrap_or(pool_size);

        // input validation
        validate_input_shape_dims(&input_shape, 4, "AveragePooling2D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_2d(pool_size, input_shape[2], input_shape[3])?;
        validate_strides_2d(strides)?;

        Ok(AveragePooling2D {
            pool_size,
            strides,
            input_shape,
            padding,
            forward_input_shape: None,
        })
    }
}

impl Layer for AveragePooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // cache the input shape for backward (averaging only needs the shape)
        self.forward_input_shape = Some(input.shape().to_vec());

        let (output, _) = windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Average,
            self.padding,
        );
        Ok(output)
    }

    /// Inference forward (eval mode, writes no caches). See [`Layer::predict`]
    fn predict(&self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        Ok(windowed_pool_forward(
            input,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Average,
            self.padding,
        )
        .0)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, Error> {
        let input_shape = self
            .forward_input_shape
            .as_ref()
            .ok_or_else(|| Error::forward_pass_not_run("AveragePooling2D"))?;

        Ok(windowed_pool_backward(
            grad_output,
            input_shape,
            &[self.pool_size.0, self.pool_size.1],
            &[self.strides.0, self.strides.1],
            PoolKind::Average,
            None,
            self.padding,
        ))
    }

    fn layer_type(&self) -> &str {
        "AveragePooling2D"
    }

    layer_functions_2d_pooling!();
}
