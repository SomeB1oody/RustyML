//! 2D average pooling layer that computes the mean value within each pooling window across
//! height and width.

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
/// Computes the mean value within each pooling window across height and width.
/// Input tensor shape: `[batch_size, height, width, channels]`. Output tensor shape:
/// `[batch_size, pooled_height, pooled_width, channels]`. With `Valid` padding,
/// `pooled_height = (height - pool_size_h) / stride_h + 1` and
/// `pooled_width = (width - pool_size_w) / stride_w + 1`.
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
/// // Input tensor: [batch_size, height, width, channels]
/// // batch size 2, a 4x4 pixel image, 3 input channels
/// let mut input_data = Array4::zeros((2, 4, 4, 3));
///
///  // Set test data to make the average pooling results predictable
///  for b in 0..2 {
///     for i in 0..4 {
///         for j in 0..4 {
///             for c in 0..3 {
///                 input_data[[b, i, j, c]] = (i + j) as f32;
///             }
///         }
///     }
///  }
///
///  let x = input_data.clone().into_dyn();
///
///  // Test AveragePooling2D with a Sequential model
///  let mut model = Sequential::new();
///  model
///  // strides default to pool_size (2, 2) and padding defaults to Valid
///  .add(AveragePooling2D::new((2, 2), vec![2, 4, 4, 3]).unwrap())
///  .compile(RMSprop::new(0.001, 0.9, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
///
///  // Output shape should be [2, 2, 2, 3]
///  let output = model.predict(&x).unwrap();
///  assert_eq!(output.shape(), &[2, 2, 2, 3]);
///
///  // Verify the pooling results
///  // A 2x2 window with stride 2 holds the average of its 4 elements
///  for b in 0..2 {
///     for c in 0..3 {
///         // First window (0, 0), (0, 1), (1, 0), (1, 1): average (0 + 1 + 1 + 2) / 4 = 1.0
///         assert_relative_eq!(output[[b, 0, 0, c]], 1.0);
///         // Second window (0, 2), (0, 3), (1, 2), (1, 3): average (2 + 3 + 3 + 4) / 4 = 3.0
///         assert_relative_eq!(output[[b, 0, 1, c]], 3.0);
///         // Third window (2, 0), (2, 1), (3, 0), (3, 1): average (2 + 3 + 3 + 4) / 4 = 3.0
///         assert_relative_eq!(output[[b, 1, 0, c]], 3.0);
///         // Fourth window (2, 2), (2, 3), (3, 2), (3, 3): average (4 + 5 + 5 + 6) / 4 = 5.0
///         assert_relative_eq!(output[[b, 1, 1, c]], 5.0);
///     }
///  }
/// ```
///
/// # Performance
///
/// The pass runs in parallel when its estimated element count
/// (`batch * out_positions * channels * window taps`) clears the gate in
/// [`tuning::pool`](crate::tuning::pool). The gate does not depend on any fixed shape.
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
    /// Shape of the most recent forward input, cached for the backward pass
    forward_input_shape: Option<Vec<usize>>,
}

impl AveragePooling2D {
    /// Creates a new 2D average pooling layer
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window as (height, width)
    /// - `input_shape` - Input tensor shape `[batch_size, height, width, channels]`
    ///
    /// # Returns
    ///
    /// - `Result<AveragePooling2D, Error>` - New layer instance on success
    ///
    /// # Notes
    ///
    /// Strides default to `pool_size` and padding defaults to [`PaddingType::Valid`]. Override them
    /// with [`AveragePooling2D::with_strides`] and [`AveragePooling2D::with_padding`].
    ///
    /// # Errors
    ///
    /// - `Error::DimensionMismatch` - If `input_shape` is not 4D
    /// - `Error::InvalidInput` - If any `input_shape` dimension is zero
    /// - `Error::InvalidParameter` - If `pool_size` has a zero dimension or exceeds the
    ///   corresponding input dimension
    pub fn new(pool_size: (usize, usize), input_shape: Vec<usize>) -> Result<Self, Error> {
        validate_input_shape_dims(&input_shape, 4, "AveragePooling2D")?;
        validate_all_dims_positive(&input_shape)?;
        validate_pool_size_2d(pool_size, input_shape[1], input_shape[2])?;

        Ok(AveragePooling2D {
            pool_size,
            strides: pool_size,
            input_shape,
            padding: PaddingType::Valid,
            forward_input_shape: None,
        })
    }

    /// Sets the pooling strides (defaults to `pool_size`)
    ///
    /// # Parameters
    ///
    /// - `strides` - Strides of the pooling operation as (height, width)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error if any stride is zero
    pub fn with_strides(mut self, strides: (usize, usize)) -> Result<Self, Error> {
        validate_strides_2d(strides)?;
        self.strides = strides;
        Ok(self)
    }

    /// Sets the padding mode (defaults to [`PaddingType::Valid`])
    ///
    /// # Parameters
    ///
    /// - `padding` - `Valid` (no padding) or `Same` (pad so the output covers the input, with
    ///   padded cells excluded from each window)
    ///
    /// # Returns
    ///
    /// - `Self` - The updated layer
    pub fn with_padding(mut self, padding: PaddingType) -> Self {
        self.padding = padding;
        self
    }
}

impl Layer for AveragePooling2D {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
        if input.ndim() != 4 {
            return Err(Error::invalid_input("input tensor is not 4D"));
        }

        // Cache the input shape for the backward pass
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

    /// Runs the forward pass for inference. Writes no cache. See [`Layer::predict`].
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
