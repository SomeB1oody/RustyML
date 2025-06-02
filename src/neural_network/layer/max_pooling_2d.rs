use super::*;
use crate::neural_network::{ModelError, Tensor};
use crate::traits::Layer;
use ndarray::ArrayD;
use rayon::prelude::*;

/// Defines a structure for max pooling operation, used to perform max pooling on 2D data.
///
/// Max pooling is a common downsampling technique in CNNs that reduces
/// the spatial dimensions of feature maps by selecting the maximum value in each pooling window,
/// thereby reducing computation and controlling overfitting.
///
/// # Fields
///
/// - `pool_size` - Size of the pooling window, expressed as (height, width).
/// - `strides` - Stride of the pooling operation, expressed as (vertical stride, horizontal stride).
/// - `input_shape` - Shape of the input tensor.
/// - `input_cache` - Cached input data, used for backpropagation.
/// - `max_positions` - Cache of maximum value positions, used for backpropagation.
///
/// # Example
/// ```rust
/// use rustyml::prelude::*;
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
///     ))
///     .compile(RMSprop::new(0.001, 0.9, 1e-8), MeanSquaredError::new());
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
/// let prediction = model.predict(&x);
/// println!("MaxPooling2D prediction results: {:?}", prediction);
///
/// // Check if output shape is correct
/// assert_eq!(prediction.shape(), &[2, 3, 3, 3]);
/// ```
pub struct MaxPooling2D {
    pool_size: (usize, usize),
    strides: (usize, usize),
    input_shape: Vec<usize>,
    input_cache: Option<Tensor>,
    max_positions: Option<Vec<(usize, usize, usize, usize)>>,
}

impl MaxPooling2D {
    /// Creates a new 2D max pooling layer.
    ///
    /// # Parameters
    ///
    /// - `pool_size` - Size of the pooling window, expressed as (height, width).
    /// - `input_shape` - Shape of the input tensor, in format \[batch_size, channels, height, width\].
    /// - `strides` - Stride of the pooling operation, expressed as (vertical stride, horizontal stride). If None, uses the same value as pool_size.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of the MaxPooling2D layer.
    pub fn new(
        pool_size: (usize, usize),
        input_shape: Vec<usize>,
        strides: Option<(usize, usize)>,
    ) -> Self {
        // If stride is not specified, use the same stride as pool size
        let strides = strides.unwrap_or(pool_size);

        MaxPooling2D {
            pool_size,
            strides,
            input_shape,
            input_cache: None,
            max_positions: None,
        }
    }

    /// Calculates the output shape of the max pooling layer.
    ///
    /// # Parameters
    ///
    /// * `input_shape` - Shape of the input tensor, in format \[batch_size, channels, height, width\].
    ///
    /// # Returns
    ///
    /// A vector containing the calculated output shape, in format \[batch_size, channels, output_height, output_width\].
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        calculate_output_shape_2d_pooling(input_shape, self.pool_size, self.strides)
    }

    /// Performs max pooling operation.
    ///
    /// # Parameters
    ///
    /// * `input` - Input tensor with shape \[batch_size, channels, height, width\].
    ///
    /// # Returns
    ///
    /// * `(Tensor, Vec<(usize, usize, usize, usize)>)` - Result of the pooling operation and positions of maximum values.
    fn max_pool(&self, input: &Tensor) -> (Tensor, Vec<(usize, usize, usize, usize)>) {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let output_shape = self.calculate_output_shape(input_shape);

        // Pre-allocate output array
        let mut output = ArrayD::zeros(output_shape.clone());
        // Vector to store positions of maximum values
        let mut max_positions = Vec::new();

        // Process each batch and channel in parallel
        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                // Clone output_shape here to avoid ownership movement issues
                let output_shape_clone = output_shape.clone();
                (0..channels).into_par_iter().map(move |c| {
                    let mut batch_channel_output = Vec::new();
                    let mut batch_channel_positions = Vec::new();

                    // Perform pooling for each output position
                    for i in 0..output_shape_clone[2] {
                        let i_start = i * self.strides.0;

                        for j in 0..output_shape_clone[3] {
                            let j_start = j * self.strides.1;

                            // Find maximum value in pooling window
                            let mut max_val = f32::NEG_INFINITY;
                            let mut max_pos = (0, 0);

                            for di in 0..self.pool_size.0 {
                                let i_pos = i_start + di;
                                if i_pos >= input_shape[2] {
                                    continue;
                                }

                                for dj in 0..self.pool_size.1 {
                                    let j_pos = j_start + dj;
                                    if j_pos >= input_shape[3] {
                                        continue;
                                    }

                                    let val = input[[b, c, i_pos, j_pos]];
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = (i_pos, j_pos);
                                    }
                                }
                            }

                            batch_channel_output.push((i, j, max_val));
                            batch_channel_positions.push((b, c, max_pos.0, max_pos.1));
                        }
                    }

                    ((b, c), (batch_channel_output, batch_channel_positions))
                })
            })
            .collect();

        // Merge results into output tensor
        for ((b, c), (outputs, positions)) in results {
            for (i, j, val) in outputs {
                output[[b, c, i, j]] = val;
            }
            max_positions.extend(positions);
        }

        (output, max_positions)
    }
}

impl Layer for MaxPooling2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Save input for backpropagation
        self.input_cache = Some(input.clone());

        // Perform max pooling operation
        let (output, max_positions) = self.max_pool(input);

        // Store maximum value positions for backpropagation
        self.max_positions = Some(max_positions);

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ModelError> {
        if let (Some(input), Some(max_positions)) = (&self.input_cache, &self.max_positions) {
            let grad_shape = grad_output.shape();

            // Initialize input gradients with same shape as input
            let mut input_gradients = ArrayD::zeros(input.dim());

            // Create a vector containing update positions and values
            let gradient_updates: Vec<_> = max_positions
                .par_iter()
                .filter_map(|&(b, c, i, j)| {
                    // Calculate corresponding output gradient index
                    let out_i = i / self.strides.0;
                    let out_j = j / self.strides.1;

                    // Ensure indices are within valid range
                    if out_i < grad_shape[2] && out_j < grad_shape[3] {
                        // Return index and gradient value
                        Some(((b, c, i, j), grad_output[[b, c, out_i, out_j]]))
                    } else {
                        None
                    }
                })
                .collect();

            // Apply gradient updates sequentially
            for ((b, c, i, j), grad_val) in gradient_updates {
                input_gradients[[b, c, i, j]] = grad_val;
            }

            Ok(input_gradients)
        } else {
            Err(ModelError::ProcessingError(
                "Forward pass has not been run".to_string(),
            ))
        }
    }

    fn layer_type(&self) -> &str {
        "MaxPooling2D"
    }

    layer_functions_2d_pooling!();
}
