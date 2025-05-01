use crate::neural_network::{LossFunction, Tensor};

/// Categorical Cross Entropy loss function for multi-class classification
pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    /// Creates a new instance of CategoricalCrossEntropy
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for CategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Ensure predictions are in a numerically stable range to avoid log(0) issues
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Calculate multi-class cross entropy: -Σ[y_true * log(y_pred)]
        // Here y_true must be one-hot encoded
        let losses = y_true * &y_pred_clipped.mapv(|y_p| y_p.ln());

        // Calculate average loss (with negative sign)
        let n = y_true.shape()[0] as f32; // Assume first dimension is sample count
        -losses.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in a numerically stable range
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Multi-class cross entropy gradient is -y_true / y_pred
        let grad = -y_true / &y_pred_clipped;

        // Divide by sample count to get average gradient
        let n = y_true.shape()[0] as f32; // Assume first dimension is sample count
        grad / n
    }
}
