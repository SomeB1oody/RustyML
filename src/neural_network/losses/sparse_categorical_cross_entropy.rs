//! Sparse categorical cross-entropy loss for class labels.
//!
//! Predictions have shape `[batch, classes]`, and labels have shape `[batch, 1]`.
//! The loss accepts logits or probability scores and averages over the batch.

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::{normalize_and_clip_rows, stable_log_softmax_softmax};
use crate::neural_network::traits::Loss;
use ndarray::{Array2, Ix2};

/// Sparse Categorical Cross Entropy loss for multi-class classification where true labels are
/// integers instead of 1-hot vectors.
///
/// # Probability path
///
/// This matches
/// [`CategoricalCrossEntropy`](crate::neural_network::losses::CategoricalCrossEntropy). The
/// integer label implies the 1-hot target. The loss renormalizes each row along the class
/// axis before the clip. Because `compute_grad` also differentiates that division,
/// it returns a dense gradient. Every class picks up a shared `1 / (batch * row_sum)` term,
/// instead of a gradient supported only on the labeled class.
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::losses::*;
/// use ndarray::ArrayD;
/// use rustyml::neural_network::traits::Loss;
///
/// // Pass `true` for `from_logits` to feed raw logits instead of probabilities.
/// let scce = SparseCategoricalCrossEntropy::new(false);
///
/// // True class labels (as integers) and predicted probabilities.
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
/// let loss = scce.compute_loss(&y_true, &y_pred).unwrap();
/// println!("Sparse Categorical Cross Entropy Loss: {:.4}", loss);
///
/// // Gradients for backpropagation.
/// let gradients = scce.compute_grad(&y_true, &y_pred).unwrap();
/// println!("Gradients shape: {:?}", gradients.shape());
/// println!("Gradients: {:?}", gradients);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SparseCategoricalCrossEntropy {
    /// Whether predictions are logits. `false` selects row normalization and probability
    /// clipping.
    from_logits: bool,
}

impl SparseCategoricalCrossEntropy {
    /// Creates a new instance of `SparseCategoricalCrossEntropy`.
    ///
    /// # Parameters
    ///
    /// - `from_logits` - if `true`, the loss treats `y_pred` as raw logits and applies softmax
    ///   internally. This is more numerically stable, and the gradient uses 1 fused step. If
    ///   `false`, the loss normalizes probability scores by row.
    ///
    /// # Returns
    ///
    /// - `SparseCategoricalCrossEntropy` - the configured loss
    pub fn new(from_logits: bool) -> Self {
        Self { from_logits }
    }
}

/// Validates the inputs and extracts the integer class index for every sample.
///
/// Predictions must have shape `[batch, num_classes]`, and labels must have shape `[batch, 1]`.
/// The validator accepts finite nonnegative labels and rounds them to the nearest integer. The
/// rounded index must be less than `num_classes`.
///
/// # Parameters
///
/// - `y_true` - integer class labels of shape `[batch, 1]`
/// - `y_pred` - predicted probabilities of shape `[batch, num_classes]`
///
/// # Returns
///
/// - `Result<Vec<usize>, Error>` - The rounded class index for each sample
///
/// # Errors
///
/// Returns `Error::invalid_input` for malformed shapes or out-of-range labels, and
/// `Error::dimension_mismatch` when the batch sizes of the 2 tensors differ.
fn validate_and_extract_labels(y_true: &Tensor, y_pred: &Tensor) -> Result<Vec<usize>, Error> {
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
        let class_indices = validate_and_extract_labels(y_true, y_pred)?;
        let batch_size = class_indices.len();

        if self.from_logits {
            // Stable log-softmax over logits, then pick the true class's log-probability per sample
            let logits = y_pred.view().into_dimensionality::<Ix2>().map_err(|_| {
                Error::invalid_input("SparseCategoricalCrossEntropy expects 2D logits")
            })?;
            let (log_sm, _) = stable_log_softmax_softmax(&logits);
            let total: f32 = class_indices
                .iter()
                .enumerate()
                .map(|(i, &class_idx)| -log_sm[[i, class_idx]])
                .sum();
            return Ok(total / batch_size as f32);
        }

        // Normalize each row before the clip. Sum in sample order so the reported loss has a
        // fixed reduction order.
        let probs = y_pred.view().into_dimensionality::<Ix2>().map_err(|_| {
            Error::invalid_input("SparseCategoricalCrossEntropy expects 2D predictions")
        })?;
        let (normalized, _) = normalize_and_clip_rows(&probs);
        let total_loss: f32 = class_indices
            .iter()
            .enumerate()
            .map(|(i, &class_idx)| -normalized[[i, class_idx]].ln())
            .sum();
        Ok(total_loss / batch_size as f32)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        let class_indices = validate_and_extract_labels(y_true, y_pred)?;
        let batch_size = class_indices.len();

        if self.from_logits {
            // Fused gradient with respect to the logits: softmax(z) with 1 subtracted at each
            // true class.
            let logits = y_pred.view().into_dimensionality::<Ix2>().map_err(|_| {
                Error::invalid_input("SparseCategoricalCrossEntropy expects 2D logits")
            })?;
            let (_, mut grad) = stable_log_softmax_softmax(&logits);
            for (i, &class_idx) in class_indices.iter().enumerate() {
                grad[[i, class_idx]] -= 1.0;
            }
            grad /= batch_size as f32;
            return Ok(grad.into_dyn());
        }

        // The row normalization adds `1 / s` to every class before the batch average. A softmax
        // backward pass cancels this constant row term.
        let probs = y_pred.view().into_dimensionality::<Ix2>().map_err(|_| {
            Error::invalid_input("SparseCategoricalCrossEntropy expects 2D predictions")
        })?;
        let (normalized, row_sums) = normalize_and_clip_rows(&probs);

        let mut grad = Array2::<f32>::zeros(probs.raw_dim());
        for (i, &class_idx) in class_indices.iter().enumerate() {
            let scale = 1.0 / (batch_size as f32 * row_sums[i]);
            grad.row_mut(i).fill(scale);
            grad[[i, class_idx]] = (1.0 - 1.0 / normalized[[i, class_idx]]) * scale;
        }
        Ok(grad.into_dyn())
    }
}
