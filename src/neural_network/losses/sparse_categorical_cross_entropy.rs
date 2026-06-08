use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::clip_probabilities;
use crate::neural_network::traits::Loss;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Sparse Categorical Cross Entropy loss function for multi-class classification
/// where true labels are integers instead of one-hot vectors
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::losses::*;
/// use ndarray::ArrayD;
/// use rustyml::neural_network::traits::Loss;
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
/// let loss = scce.compute_loss(&y_true, &y_pred).unwrap();
/// println!("Sparse Categorical Cross Entropy Loss: {:.4}", loss);
///
/// // Compute gradients for backpropagation
/// let gradients = scce.compute_grad(&y_true, &y_pred).unwrap();
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

impl Default for SparseCategoricalCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates the inputs and extracts the integer class index for every sample.
///
/// Predictions must be 2D `[batch, num_classes]` and labels must be 2D `[batch, 1]` with
/// non-negative integer values strictly less than `num_classes`. Without these checks an
/// out-of-range label would cause an opaque index-out-of-bounds panic inside a Rayon worker.
fn validate_and_extract_labels(
    y_true: &Tensor,
    y_pred: &Tensor,
) -> Result<Vec<usize>, Error> {
    if y_pred.ndim() != 2 {
        return Err(Error::invalid_input(format!(
            "SparseCategoricalCrossEntropy expects 2D predictions [batch, num_classes], got shape {:?}",
            y_pred.shape()
        )));
    }
    if y_true.ndim() != 2 || y_true.shape()[1] != 1 {
        return Err(Error::invalid_input(format!(
            "SparseCategoricalCrossEntropy expects integer labels of shape [batch, 1], got shape {:?}",
            y_true.shape()
        )));
    }
    if y_true.shape()[0] != y_pred.shape()[0] {
        return Err(Error::dimension_mismatch(
            y_true.shape()[0],
            y_pred.shape()[0],
        ));
    }

    let batch_size = y_true.shape()[0];
    let num_classes = y_pred.shape()[1];

    (0..batch_size)
        .map(|i| {
            let label = y_true[[i, 0]];
            if !label.is_finite() || label < 0.0 {
                return Err(Error::invalid_input(format!(
                    "SparseCategoricalCrossEntropy label at sample {i} must be a non-negative integer, got {label}"
                )));
            }
            let class_idx = label.round() as usize;
            if class_idx >= num_classes {
                return Err(Error::invalid_input(format!(
                    "SparseCategoricalCrossEntropy label {class_idx} at sample {i} is out of range for {num_classes} classes"
                )));
            }
            Ok(class_idx)
        })
        .collect()
}

impl Loss for SparseCategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error> {
        // Ensure predictions are in a numerically stable range to avoid log(0) issues
        let y_pred_clipped = clip_probabilities(y_pred);

        let class_indices = validate_and_extract_labels(y_true, y_pred)?;
        let batch_size = class_indices.len();

        // Compute loss in parallel
        let total_loss: f32 = class_indices
            .par_iter()
            .enumerate()
            .map(|(i, &class_idx)| -y_pred_clipped[[i, class_idx]].ln())
            .sum();

        // Return average loss
        Ok(total_loss / batch_size as f32)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        // Ensure predictions are in a numerically stable range
        let y_pred_clipped = clip_probabilities(y_pred);

        let class_indices = validate_and_extract_labels(y_true, y_pred)?;
        let batch_size = class_indices.len();

        // Create a gradient tensor with the same shape as y_pred, initialized to 0.
        // Each sample writes a single distinct entry, so a plain sequential pass is both
        // correct and faster than collecting updates in parallel and re-applying them.
        let mut grad = Tensor::zeros(y_pred.raw_dim());
        for (i, &class_idx) in class_indices.iter().enumerate() {
            grad[[i, class_idx]] = -1.0 / y_pred_clipped[[i, class_idx]];
        }

        // Return average gradient
        Ok(grad / batch_size as f32)
    }
}
