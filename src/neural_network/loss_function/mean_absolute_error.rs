use crate::neural_network::{LossFunction, Tensor};

/// Mean Absolute Error loss function
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Creates a new instance of MeanAbsoluteError
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanAbsoluteError {
    /// Computes the Mean Absolute Error between predicted and true values
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values
    /// - `y_pred` - Tensor with predicted values
    ///
    /// # Returns
    ///
    /// * `f32` - Average of absolute differences between predictions and ground truth
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        let diff = y_pred - y_true;
        let abs_diff = diff.mapv(|x| x.abs());
        abs_diff.sum() / (y_true.len() as f32)
    }

    /// Computes the gradient of Mean Absolute Error with respect to predictions
    ///
    /// # Parameters
    ///
    /// - `y_true` - Tensor with ground truth values
    /// - `y_pred` - Tensor with predicted values
    ///
    /// # Returns
    ///
    /// * `Tensor` - Gradient tensor for backpropagation
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let n = y_true.len() as f32;
        (y_pred - y_true).mapv(|x| {
            if x > 0.0 {
                1.0 / n
            } else if x < 0.0 {
                -1.0 / n
            } else {
                0.0
            }
        })
    }
}
