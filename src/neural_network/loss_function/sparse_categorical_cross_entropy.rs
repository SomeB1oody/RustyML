use super::super::{LossFunction, Tensor};
use rayon::prelude::*;

/// Sparse Categorical Cross Entropy loss function for multi-class classification
/// where true labels are integers instead of one-hot vectors
///
/// # Example
///
/// ```rust
/// use rustyml::prelude::*;
/// use ndarray::ArrayD;
///
/// // Create a Sparse Categorical Cross Entropy loss function instance
/// let scce = SparseCategoricalCrossEntropy::new();
///
/// // Create sample data - true class labels (as integers) and predicted probabilities
/// let y_true = ArrayD::from_shape_vec(
///     vec![3, 1],
///     vec![0.0, 1.0, 2.0]  // Class indices: 0, 1, 2
/// ).unwrap();
///
/// let y_pred = ArrayD::from_shape_vec(
///     vec![3, 3],  // 3 samples, 3 classes
///     vec![
///         0.7, 0.2, 0.1,  // Sample 1: high confidence for class 0 (correct)
///         0.1, 0.8, 0.1,  // Sample 2: high confidence for class 1 (correct)
///         0.2, 0.3, 0.5   // Sample 3: moderate confidence for class 2 (correct)
///     ]
/// ).unwrap();
///
/// // Compute the Sparse Categorical Cross Entropy loss
/// let loss = scce.compute_loss(&y_true, &y_pred);
/// println!("Sparse Categorical Cross Entropy Loss: {:.4}", loss);
///
/// // Compute gradients for backpropagation
/// let gradients = scce.compute_grad(&y_true, &y_pred);
/// println!("Gradients shape: {:?}", gradients.shape());
/// println!("Gradients: {:?}", gradients);
/// ```
pub struct SparseCategoricalCrossEntropy;

impl SparseCategoricalCrossEntropy {
    /// Creates a new instance of SparseCategoricalCrossEntropy
    ///
    /// # Returns
    ///
    /// - `SparseCategoricalCrossEntropy` - Returns a unit-like struct `SparseCategoricalCrossEntropy`
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
