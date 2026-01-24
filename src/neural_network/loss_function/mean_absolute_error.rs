use super::super::{LossFunction, Tensor};

/// Mean Absolute Error loss function
///
/// # Examples
///
/// ```rust
/// use rustyml::prelude::*;
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
/// let loss = mae.compute_loss(&y_true, &y_pred);
/// println!("MAE Loss: {:.4}", loss); // Expected: ~0.175
///
/// // Compute gradients
/// let gradients = mae.compute_grad(&y_true, &y_pred);
/// println!("Gradients: {:?}", gradients);
/// ```
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Creates a new instance of MeanAbsoluteError
    ///
    /// # Returns
    ///
    /// - `MeanAbsoluteError` - Returns a unit-like struct `MeanAbsoluteError`
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanAbsoluteError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        let mut diff = y_pred - y_true;
        diff.par_mapv_inplace(|x| x.abs());
        diff.sum() / (y_true.len() as f32)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
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
        result
    }
}
