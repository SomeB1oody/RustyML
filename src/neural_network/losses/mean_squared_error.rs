//! Mean Squared Error loss function and its gradient

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::validate_same_shape;
use crate::neural_network::traits::Loss;

/// Mean Squared Error loss function
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::losses::*;
/// use ndarray::ArrayD;
/// use rustyml::neural_network::traits::Loss;
///
/// // Create an MSE loss function instance
/// let mse = MeanSquaredError::new();
///
/// // Sample data: true values and predictions
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
/// let loss = mse.compute_loss(&y_true, &y_pred).unwrap();
/// println!("MSE Loss: {:.4}", loss);
///
/// // Compute gradients for backpropagation
/// let gradients = mse.compute_grad(&y_true, &y_pred).unwrap();
/// println!("Gradients: {:?}", gradients);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Creates a new `MeanSquaredError` instance
    ///
    /// # Returns
    ///
    /// - `MeanSquaredError` - A unit-like loss-function struct
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for MeanSquaredError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error> {
        validate_same_shape(y_true, y_pred)?;

        let squared_diff = (y_pred - y_true).mapv(|x| x * x);

        // Mean over all elements
        let n = squared_diff.len() as f32;
        Ok(squared_diff.sum() / n)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        validate_same_shape(y_true, y_pred)?;

        let diff = y_pred - y_true;

        // Gradient is 2 * diff / element count
        let n = diff.len() as f32;

        let mut result = diff.clone();
        result.par_mapv_inplace(|x| 2.0 * x / n);

        Ok(result)
    }
}
