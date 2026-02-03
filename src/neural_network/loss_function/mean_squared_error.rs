use crate::neural_network::Tensor;
use crate::neural_network::neural_network_trait::LossFunction;

/// Mean Squared Error loss function
///
/// # Example
///
/// ```rust
/// use rustyml::neural_network::loss_function::*;
/// use ndarray::ArrayD;
/// use rustyml::prelude::neural_network_trait::LossFunction;
///
/// // Create an MSE loss function instance
/// let mse = MeanSquaredError::new();
///
/// // Create sample data - true values and predictions
/// let y_true = ArrayD::from_shape_vec(
///     vec![3, 1],
///     vec![1.0, 2.0, 3.0]
/// ).unwrap();
///
/// let y_pred = ArrayD::from_shape_vec(
///     vec![3, 1],
///     vec![1.1, 2.2, 2.8]
/// ).unwrap();
///
/// // Compute the MSE loss
/// let loss = mse.compute_loss(&y_true, &y_pred);
/// println!("MSE Loss: {:.4}", loss);
///
/// // Compute gradients for backpropagation
/// let gradients = mse.compute_grad(&y_true, &y_pred);
/// println!("Gradients: {:?}", gradients);
/// ```
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Creates a new instance of MeanSquaredError
    ///
    /// # Returns
    ///
    /// - `MeanSquaredError` - Returns a unit-like struct `MeanSquaredError`
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanSquaredError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Calculate the squared difference
        let squared_diff = (y_pred - y_true).mapv(|x| x * x);

        // Calculate the mean (sum divided by number of elements)
        let n = squared_diff.len() as f32;
        squared_diff.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Calculate the difference between predictions and ground truth
        let diff = y_pred - y_true;

        // Gradient is 2 times the difference divided by sample count
        let n = diff.len() as f32;

        let mut result = diff.clone();
        result.par_mapv_inplace(|x| 2.0 * x / n);

        result
    }
}
