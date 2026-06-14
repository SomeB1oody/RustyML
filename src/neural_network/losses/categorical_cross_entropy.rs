//! Categorical cross entropy loss for multi-class classification with one-hot targets

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::{
    clip_probabilities, stable_log_softmax_softmax, validate_same_shape,
};
use crate::neural_network::traits::Loss;

/// Categorical cross entropy loss function for multi-class classification
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::losses::*;
/// use rustyml::neural_network::traits::Loss;
/// use ndarray::array;
///
/// // Create the loss function (pass `true` for `from_logits` to feed raw logits instead)
/// let loss_fn = CategoricalCrossEntropy::new(false);
///
/// // Ground truth labels (one-hot encoded)
/// // Class 0: [1, 0, 0], Class 1: [0, 1, 0], Class 2: [0, 0, 1]
/// let y_true = array![
///     [1.0f32, 0.0f32, 0.0f32],  // Sample 1: belongs to class 0
///     [0.0f32, 1.0f32, 0.0f32],  // Sample 2: belongs to class 1
///     [0.0f32, 0.0f32, 1.0f32]   // Sample 3: belongs to class 2
/// ].into_dyn();
///
/// // Predicted probabilities (softmax output)
/// let y_pred = array![
///     [0.8f32, 0.1f32, 0.1f32],  // Sample 1: high confidence for class 0
///     [0.2f32, 0.7f32, 0.1f32],  // Sample 2: good confidence for class 1
///     [0.1f32, 0.2f32, 0.7f32]   // Sample 3: good confidence for class 2
/// ].into_dyn();
///
/// // Compute loss
/// let loss = loss_fn.compute_loss(&y_true, &y_pred).unwrap();
/// println!("Categorical Cross Entropy Loss: {:.4}", loss);
///
/// // Compute gradient for backpropagation
/// let gradient = loss_fn.compute_grad(&y_true, &y_pred).unwrap();
/// println!("Gradient shape: {:?}", gradient.shape());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CategoricalCrossEntropy {
    /// When `true`, `y_pred` is treated as raw logits: the loss applies a numerically stable
    /// log-softmax internally and `compute_grad` returns the fused `(softmax(z) - y) / batch`
    /// gradient. When `false` (default), `y_pred` must already be a probability distribution
    from_logits: bool,
}

impl CategoricalCrossEntropy {
    /// Creates a new `CategoricalCrossEntropy` instance
    ///
    /// # Parameters
    ///
    /// - `from_logits` - If `true`, `y_pred` is interpreted as raw logits and softmax is applied
    ///   internally (more numerically stable, and the gradient is computed in one fused
    ///   `softmax(z) - y` step, so the model's last layer should output logits, not probabilities).
    ///   If `false`, `y_pred` must already be a normalized probability distribution per row
    ///
    /// # Returns
    ///
    /// - `CategoricalCrossEntropy` - the configured loss
    pub fn new(from_logits: bool) -> Self {
        Self { from_logits }
    }
}

/// Flattens a `[batch, ...]` tensor's trailing axes into `[batch, classes]` for the logits path
fn batch_and_classes(t: &Tensor) -> (usize, usize) {
    let batch = t.shape()[0];
    let classes: usize = t.shape()[1..].iter().product();
    (batch, classes)
}

/// Validates that one-hot targets and predictions are non-empty, at least 2D, and shape-compatible
///
/// Categorical cross entropy is an elementwise product reduced over the batch, so the two
/// tensors must share the same shape, and the batch axis must be non-empty (it is the divisor).
/// The input must be at least 2D `[batch, classes]`: with a 1D tensor, `shape()[0]` would be the
/// total element count rather than the batch size, silently rescaling the loss and its gradient
fn validate_shapes(y_true: &Tensor, y_pred: &Tensor) -> Result<(), Error> {
    if y_true.is_empty() {
        return Err(Error::empty_input(
            "CategoricalCrossEntropy expects non-empty y_true",
        ));
    }
    if y_true.ndim() < 2 {
        return Err(Error::invalid_input(format!(
            "CategoricalCrossEntropy expects at least 2D tensors [batch, classes], got {}D",
            y_true.ndim()
        )));
    }
    validate_same_shape(y_true, y_pred)
}

impl Loss for CategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32, Error> {
        validate_shapes(y_true, y_pred)?;
        let n = y_true.shape()[0] as f32;

        if self.from_logits {
            // Fused softmax-cross-entropy
            let (b, c) = batch_and_classes(y_pred);
            let logits = y_pred
                .to_shape((b, c))
                .map_err(|e| Error::computation(format!("CCE logits reshape failed: {e}")))?;
            let labels = y_true
                .to_shape((b, c))
                .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
            let (log_sm, _) = stable_log_softmax_softmax(&logits.view());
            return Ok(-(&labels * &log_sm).sum() / n);
        }

        // Probability path
        let y_pred_clipped = clip_probabilities(y_pred);
        let losses = y_true * &y_pred_clipped.mapv(|y_p| y_p.ln());
        Ok(-losses.sum() / n)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        validate_shapes(y_true, y_pred)?;
        let n = y_true.shape()[0] as f32;

        if self.from_logits {
            // Fused gradient w.r.t. the logits
            let (b, c) = batch_and_classes(y_pred);
            let logits = y_pred
                .to_shape((b, c))
                .map_err(|e| Error::computation(format!("CCE logits reshape failed: {e}")))?;
            let labels = y_true
                .to_shape((b, c))
                .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
            let (_, sm) = stable_log_softmax_softmax(&logits.view());
            let grad2d = (&sm - &labels) / n;
            let grad = grad2d
                .into_shape_with_order(y_pred.raw_dim())
                .map_err(|e| Error::computation(format!("CCE gradient reshape failed: {e}")))?;
            return Ok(grad);
        }

        // Probability path: gradient is -y_true / y_pred
        let y_pred_clipped = clip_probabilities(y_pred);
        let grad = -y_true / &y_pred_clipped;
        Ok(grad / n)
    }
}
