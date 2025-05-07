use crate::neural_network::{LossFunction, Tensor};

/// Mean Squared Error loss function
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Creates a new instance of MeanSquaredError
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
