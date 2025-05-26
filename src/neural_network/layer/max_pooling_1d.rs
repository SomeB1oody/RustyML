use super::super::Tensor;
use crate::ModelError;
use crate::neural_network::layer::LayerWeight;
use crate::traits::Layer;
use ndarray::Array3;
use rayon::prelude::*;

/// 1D Max Pooling layer for neural networks.
///
/// This layer performs max pooling operation on a 3D tensor.
/// Max pooling selects the maximum value from each patch as defined by the pool size.
///
/// # Input Shape
///
/// Input is a 3D tensor with shape \[batch_size, channels, length\]
///
/// # Output Shape
///
/// Output is a 3D tensor with shape \[batch_size, channels, pooled_length\]
/// Where:
/// - pooled_length = (length - pool_size) / stride + 1
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window
/// - `stride` - Stride of the pooling operation
/// - `input_shape` - Shape of the input tensor
/// - `input_cache` - Input tensor cached during forward pass, used for backward pass
/// - `max_positions` - Positions of the maximum values in each pooling window, used for backpropagation
pub struct MaxPooling1D {
    pool_size: usize,
    stride: usize,
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize)>>,
}

impl MaxPooling1D {
    /// Create a new 1D Max Pooling layer
    ///
    /// # Parameters
    ///
    /// * `pool_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling operation
    /// * `input_shape` - Shape of the input tensor \[batch_size, channels, length\]
    pub fn new(pool_size: usize, stride: usize, input_shape: Vec<usize>) -> Self {
        MaxPooling1D {
            pool_size,
            stride,
            input_shape,
            input_cache: None,
            max_positions: None,
        }
    }

    /// Calculate output shape
    ///
    /// Computes the output shape based on input shape, pool size, and stride
    fn compute_output_shape(&self) -> Vec<usize> {
        let batch_size = self.input_shape[0];
        let channels = self.input_shape[1];
        let length = self.input_shape[2];

        let output_length = (length - self.pool_size) / self.stride + 1;

        vec![batch_size, channels, output_length]
    }
}

impl Layer for MaxPooling1D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Cache input for backward pass
        self.input_cache = Some(input.clone());

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let output_length = (length - self.pool_size) / self.stride + 1;
        let mut output = Array3::<f32>::zeros((batch_size, channels, output_length)).into_dyn();

        // Store the positions of maximum values for backpropagation
        let mut max_positions = Vec::new();

        // Copy needed values from self to avoid capturing self in closure
        let pool_size = self.pool_size;
        let stride = self.stride;

        // Use rayon to process batches and channels in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();
                    let mut batch_channel_max_positions = Vec::new();

                    // Perform pooling for each output position
                    for i in 0..output_length {
                        let start_idx = i * stride;
                        let end_idx = start_idx + pool_size;

                        // Find maximum value and its position in the window
                        let mut max_val = f32::MIN;
                        let mut max_idx = start_idx;
                        for j in start_idx..end_idx {
                            if input[[b, c, j]] > max_val {
                                max_val = input[[b, c, j]];
                                max_idx = j;
                            }
                        }

                        batch_channel_output.push((i, max_val));
                        batch_channel_max_positions.push((b, c, max_idx));
                    }

                    ((b, c), (batch_channel_output, batch_channel_max_positions))
                })
            })
            .collect();

        // Merge results into output tensor and collect max positions
        for ((b, c), (outputs, positions)) in results {
            for (i, val) in outputs {
                output[[b, c, i]] = val;
            }
            max_positions.extend(positions);
        }

        self.max_positions = Some(max_positions);
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        // Ensure we have cached input
        let input = match &self.input_cache {
            Some(input) => input,
            None => {
                return Err(ModelError::ProcessingError(
                    "No cached input for MaxPooling1D".to_string(),
                ));
            }
        };

        // Ensure we have stored max positions
        let max_positions = match &self.max_positions {
            Some(positions) => positions,
            None => {
                return Err(ModelError::ProcessingError(
                    "No cached max positions for MaxPooling1D".to_string(),
                ));
            }
        };

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let length = input.shape()[2];

        let mut grad_input = Array3::<f32>::zeros((batch_size, channels, length)).into_dyn();

        // For max pooling, gradients flow only through the maximum value in each pooling window
        for (b, c, idx) in max_positions {
            // Calculate which output position this max value contributed to
            let output_idx = (*idx - (*idx % self.stride)) / self.stride;

            // Ensure we're within bounds of grad_output
            if output_idx < grad_output.shape()[2] {
                // Pass the gradient to the input position that had the maximum value
                grad_input[[*b, *c, *idx]] += grad_output[[*b, *c, output_idx]];
            }
        }

        Ok(grad_input)
    }

    fn layer_type(&self) -> &str {
        "MaxPooling1D"
    }

    fn output_shape(&self) -> String {
        let output_shape = self.compute_output_shape();
        format!("{:?}", output_shape)
    }

    fn update_parameters_sgd(&mut self, _lr: f32) {
        // Pooling layers have no trainable parameters
    }

    fn update_parameters_adam(
        &mut self,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _t: u64,
    ) {
        // Pooling layers have no trainable parameters
    }

    fn update_parameters_rmsprop(&mut self, _lr: f32, _rho: f32, _epsilon: f32) {
        // Pooling layers have no trainable parameters
    }

    fn get_weights(&self) -> LayerWeight {
        LayerWeight::Empty
    }
}
