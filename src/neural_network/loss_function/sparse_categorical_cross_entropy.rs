use crate::neural_network::{LossFunction, Tensor};
use rayon::prelude::*;

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
        let mut y_pred_clipped = y_pred.clone();
        y_pred_clipped.par_mapv_inplace(|x| x.max(1e-7).min(1.0 - 1e-7));

        let batch_size = y_true.shape()[0];

        // Pre-compute class indices for all samples
        let class_indices: Vec<usize> = (0..batch_size)
            .map(|i| y_true[[i, 0]].round() as usize)
            .collect();

        // Compute loss in parallel
        let total_loss: f32 = class_indices
            .par_iter()
            .enumerate()
            .map(|(i, &class_idx)| -y_pred_clipped[[i, class_idx]].ln())
            .sum();

        // Return average loss
        total_loss / batch_size as f32
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // Ensure predictions are in a numerically stable range
        let mut y_pred_clipped = y_pred.clone();
        y_pred_clipped.par_mapv_inplace(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Create a gradient tensor with the same shape as y_pred, initialized to 0
        let mut grad = Tensor::zeros(y_pred.raw_dim());

        let batch_size = y_true.shape()[0];

        // Pre-compute class indices for all samples
        let class_indices: Vec<usize> = (0..batch_size)
            .map(|i| y_true[[i, 0]].round() as usize)
            .collect();

        // Create a thread-safe vector to store update information
        let updates: Vec<((usize, usize), f32)> = class_indices
            .par_iter()
            .enumerate()
            .map(|(i, &class_idx)| {
                let predicted_prob = y_pred_clipped[[i, class_idx]];
                ((i, class_idx), -1.0 / predicted_prob)
            })
            .collect();

        // Apply updates to the gradient tensor
        for ((i, class_idx), value) in updates {
            grad[[i, class_idx]] = value;
        }

        // Return average gradient
        grad / batch_size as f32
    }
}
