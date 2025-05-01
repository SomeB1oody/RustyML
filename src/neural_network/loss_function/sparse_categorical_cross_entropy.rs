use crate::neural_network::{LossFunction, Tensor};
use ndarray::s;

/// Sparse Categorical Cross Entropy loss function for multi-class classification
/// where true labels are integers instead of one-hot vectors
pub struct SparseCategoricalCrossEntropy;

impl SparseCategoricalCrossEntropy {
    /// Creates a new instance of SparseCategoricalCrossEntropy
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for SparseCategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // Ensure predictions are in a numerically stable range to avoid log(0) issues
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Assume y_true contains class indices (integer values)
        // We need to get the probability of the true class for each sample
        let mut total_loss = 0.0;
        let batch_size = y_true.shape()[0];

        for i in 0..batch_size {
            // Get the true class index for the current sample
            let class_idx = y_true.slice(s![i, ..]).iter().next().unwrap().round() as usize;

            // First save the slice view, then extract value from it
            let slice = y_pred_clipped.slice(s![i, class_idx]);
            let predicted_prob = slice.iter().next().unwrap();

            // Accumulate cross entropy loss: -log(predicted_prob)
            total_loss -= predicted_prob.ln();
        }

        // Return average loss
        total_loss / batch_size as f32
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in a numerically stable range
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Create a gradient tensor with the same shape as y_pred, initialized to 0
        let mut grad = y_pred.clone();
        grad.fill(0.0);

        let batch_size = y_true.shape()[0];

        for i in 0..batch_size {
            // Get the true class index for the current sample
            let class_idx = y_true.slice(s![i, ..]).iter().next().unwrap().round() as usize;

            // First save the slice view, then extract value from it
            let slice = y_pred_clipped.slice(s![i, class_idx]);
            let predicted_prob = slice.iter().next().unwrap();

            // Modify the gradient at the corresponding position
            let mut view = grad.slice_mut(s![i, class_idx]);
            *view.iter_mut().next().unwrap() = -1.0 / predicted_prob;
        }

        // Return average gradient
        grad / batch_size as f32
    }
}
