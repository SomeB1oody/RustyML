//! Mean Absolute Error (MAE) loss function and its gradient

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::validate_same_shape;
use crate::neural_network::traits::Loss;

/// Mean Absolute Error loss function
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::losses::*;
/// use rustyml::neural_network::Tensor;
/// use rustyml::neural_network::traits::Loss;
/// use ndarray::ArrayD;
///
/// // Create sample data
/// let y_true: Tensor = ArrayD::from_shape_vec(
///     vec![4],
///     vec![1.0, 2.0, 3.0, 4.0]
/// ).unwrap();
///
/// let y_pred: Tensor = ArrayD::from_shape_vec(
///     vec![4],
///     vec![1.2, 1.8, 3.2, 3.9]
/// ).unwrap();
///
/// // Initialize MAE loss function
/// let mae = MeanAbsoluteError::new();
///
/// // Compute loss
/// let loss = mae.compute_loss(&y_true, &y_pred).unwrap();
/// println!("MAE Loss: {:.4}", loss); // Expected: ~0.175
///
/// // Compute gradients
/// let gradients = mae.compute_grad(&y_true, &y_pred).unwrap();
/// println!("Gradients: {:?}", gradients);
/// ```
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Creates a new `MeanAbsoluteError` instance
    ///
    /// # Returns
    ///
    /// - `MeanAbsoluteError` - A unit-like struct
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for MeanAbsoluteError {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for MeanAbsoluteError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error> {
        validate_same_shape(y_true, y_pred)?;
        let mut diff = y_pred - y_true;
        diff.par_mapv_inplace(|x| x.abs());
        Ok(diff.sum() / (y_true.len() as f32))
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        validate_same_shape(y_true, y_pred)?;
        let n = y_true.len() as f32;
        let mut result = y_pred - y_true;
        result.par_mapv_inplace(|x| {
            if x > 0.0 {
                1.0 / n
            } else if x < 0.0 {
                -1.0 / n
            } else {
                0.0
            }
        });
        Ok(result)
    }
}
