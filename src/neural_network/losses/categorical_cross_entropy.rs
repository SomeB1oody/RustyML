//! Categorical cross entropy loss for multi-class classification with one-hot targets.

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::{
    normalize_and_clip_rows, stable_log_softmax_softmax, validate_same_shape,
};
use crate::neural_network::traits::Loss;
use ndarray::{Array2, Zip};

/// Categorical cross entropy loss function for multi-class classification.
///
/// # Input rank
///
/// The last axis is always the class axis. Every axis before it indexes an independent
/// prediction site. So `[batch, classes]` is 1 prediction per sample. The `[batch, height,
/// width, classes]` output of a channels-last `Conv2D` softmax head is 1 prediction per pixel.
/// The loss sums over the class axis and divides by the total number of sites (`batch * height *
/// width`). This is what Keras' default `sum_over_batch_size` reduction computes. Under
/// `from_logits`, the softmax also normalizes within a site only, never across sites.
///
/// # Probability path
///
/// With `from_logits == false`, the loss renormalizes each row along the class axis, using
/// `y_pred / sum(y_pred, axis=-1)`. It then clips the row away from 0 and 1. This order matches
/// Keras.
///
/// A genuine softmax row already sums to 1, so the division does not change the loss value. The
/// division is still part of what `compute_grad` differentiates, so it adds a row-constant term
/// to the gradient. A softmax backward pass cancels any row-constant vector, so a normal softmax
/// head trains the same either way. The term only matters when the loss reads an unnormalized
/// head directly.
///
/// The clip guards against `log(0)`. It does not gate the gradient. Like the loss, `compute_grad`
/// evaluates at the clipped probability. It does not set clipped positions to zero the way
/// Keras' autodiff does. Loss and gradient agree wherever `y_pred` is inside `[1e-7, 1 - 1e-7]`.
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
    /// When `true`, the loss treats `y_pred` as raw logits. It applies a numerically stable
    /// log-softmax over the class axis, and `compute_grad` returns the fused
    /// `(softmax(z) - y) / prediction_sites` gradient. When `false` (default), `y_pred` must
    /// already be a probability distribution along its last axis.
    from_logits: bool,
}

impl CategoricalCrossEntropy {
    /// Creates a new `CategoricalCrossEntropy` instance.
    ///
    /// # Parameters
    ///
    /// - `from_logits` - if `true`, the loss treats `y_pred` as raw logits and applies softmax
    ///   internally. This is more numerically stable, and `compute_grad` fuses the gradient into
    ///   1 `softmax(z) - y` step, so the last layer should output logits, not probabilities.
    ///   If `false`, `y_pred` must already be a normalized probability distribution per row.
    ///
    /// # Returns
    ///
    /// - `CategoricalCrossEntropy` - the configured loss
    pub fn new(from_logits: bool) -> Self {
        Self { from_logits }
    }
}

/// Splits a `[..., classes]` tensor into `(prediction_sites, classes)`: the product of every
/// leading axis, and the length of the trailing class axis.
///
/// The class axis is always the last one. Every leading axis (batch, plus any spatial or
/// temporal positions) indexes an independent prediction site. It contributes to the site
/// count, not to the class count. This matches Keras' `sum_over_batch_size` reduction: it
/// reduces the last axis, then divides by the element count of what remains.
fn leading_and_classes(t: &Tensor) -> (usize, usize) {
    let ndim = t.ndim();
    let classes = t.shape()[ndim - 1];
    let sites: usize = t.shape()[..ndim - 1].iter().product();
    (sites, classes)
}

/// Validates that one-hot targets and predictions are non-empty, at least 2D, and
/// shape-compatible.
///
/// The 2 tensors must share the same shape. The input must also be at least 2D, so a class axis
/// and at least 1 prediction-site axis both exist. With a 1D tensor there would be no leading
/// axis at all, leaving a divisor of 1 and a softmax over the only axis present. That treats a
/// single prediction as a batch of 1.
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
        let (sites, classes) = leading_and_classes(y_pred);
        let n = sites as f32;

        if self.from_logits {
            // Fused softmax-cross-entropy. The reshape merges only the leading axes, so the
            // softmax below normalizes within 1 prediction site and never across sites.
            let logits = y_pred
                .to_shape((sites, classes))
                .map_err(|e| Error::computation(format!("CCE logits reshape failed: {e}")))?;
            let labels = y_true
                .to_shape((sites, classes))
                .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
            let (log_sm, _) = stable_log_softmax_softmax(&logits.view());
            return Ok(-(&labels * &log_sm).sum() / n);
        }

        // Probability path. The reshape is the same as above, because Keras' renormalizer runs
        // over the class axis alone. A `[batch, h, w, classes]` head, for example, is 1
        // distribution per pixel.
        let probs = y_pred
            .to_shape((sites, classes))
            .map_err(|e| Error::computation(format!("CCE probability reshape failed: {e}")))?;
        let labels = y_true
            .to_shape((sites, classes))
            .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
        let (normalized, _) = normalize_and_clip_rows(&probs.view());
        Ok(-(&labels * &normalized.mapv(f32::ln)).sum() / n)
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<Tensor, Error> {
        validate_shapes(y_true, y_pred)?;
        let (sites, classes) = leading_and_classes(y_pred);
        let n = sites as f32;

        if self.from_logits {
            // Fused gradient with respect to the logits, per prediction site (see `compute_loss`).
            let logits = y_pred
                .to_shape((sites, classes))
                .map_err(|e| Error::computation(format!("CCE logits reshape failed: {e}")))?;
            let labels = y_true
                .to_shape((sites, classes))
                .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
            let (_, sm) = stable_log_softmax_softmax(&logits.view());
            let grad2d = (&sm - &labels) / n;
            let grad = grad2d
                .into_shape_with_order(y_pred.raw_dim())
                .map_err(|e| Error::computation(format!("CCE gradient reshape failed: {e}")))?;
            return Ok(grad);
        }

        // Probability path. The row-normalizer is part of the function that `compute_grad`
        // differentiates. The gradient of `-sum_c y_c * ln(p_c / s)` with respect to `p_k` is
        // `(sum_c y_c) / s - y_k / p_k`. That is `-y_k / p_k` plus the row-constant term the
        // type-level doc above describes.
        let probs = y_pred
            .to_shape((sites, classes))
            .map_err(|e| Error::computation(format!("CCE probability reshape failed: {e}")))?;
        let labels = y_true
            .to_shape((sites, classes))
            .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
        let (normalized, row_sums) = normalize_and_clip_rows(&probs.view());

        let mut grad2d = Array2::<f32>::zeros((sites, classes));
        Zip::from(grad2d.rows_mut())
            .and(normalized.rows())
            .and(labels.rows())
            .and(&row_sums)
            .for_each(|mut grad_row, normalized_row, label_row, &sum| {
                // One-hot targets make this 1, but the loss also works for any target mass.
                let target_mass = label_row.sum();
                Zip::from(&mut grad_row)
                    .and(normalized_row)
                    .and(label_row)
                    .for_each(|grad, &q, &y| *grad = (target_mass - y / q) / (n * sum));
            });

        grad2d
            .into_shape_with_order(y_pred.raw_dim())
            .map_err(|e| Error::computation(format!("CCE gradient reshape failed: {e}")))
    }
}
