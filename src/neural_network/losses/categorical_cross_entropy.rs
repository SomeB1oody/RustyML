//! Categorical cross entropy loss for multi-class classification with one-hot targets

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::losses::{
    normalize_and_clip_rows, stable_log_softmax_softmax, validate_same_shape,
};
use crate::neural_network::traits::Loss;
use ndarray::{Array2, Zip};

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
///
/// # Input rank
///
/// The **last** axis is the class axis; every axis before it indexes an independent prediction
/// site. So `[batch, classes]` is one prediction per sample, while the `[batch, height, width,
/// classes]` output of a channels-last `Conv2D` softmax head is one prediction per pixel. The
/// loss sums over the class axis and divides by the total number of sites (`batch * height *
/// width`), which is what Keras' default `sum_over_batch_size` reduction computes; under
/// `from_logits` the softmax likewise normalizes within a site and never across sites
///
/// # Probability path
///
/// With `from_logits == false`, each row is renormalized along the class axis
/// (`y_pred / sum(y_pred, axis=-1)`) *before* being clipped away from 0 and 1, matching Keras'
/// order of operations. A genuine softmax row already sums to 1, so the division does not move
/// the loss value - but it is still differentiated, which adds a row-constant term to the
/// gradient. A softmax backward annihilates row-constant vectors, so this changes nothing for the
/// usual softmax head; it makes the loss well-defined for an unnormalized one
///
/// The clip is a `log(0)` guard, not a gradient gate: like the loss itself, `compute_grad`
/// evaluates at the clipped probability rather than zeroing out clipped positions the way Keras'
/// autodiff does. The two agree wherever `y_pred` is inside `[1e-7, 1 - 1e-7]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CategoricalCrossEntropy {
    /// When `true`, `y_pred` is treated as raw logits: the loss applies a numerically stable
    /// log-softmax over the class axis and `compute_grad` returns the fused
    /// `(softmax(z) - y) / prediction_sites` gradient. When `false` (default), `y_pred` must
    /// already be a probability distribution along its last axis
    from_logits: bool,
}

impl CategoricalCrossEntropy {
    /// Creates a new `CategoricalCrossEntropy` instance
    ///
    /// # Parameters
    ///
    /// - `from_logits` - If `true`, `y_pred` is interpreted as raw logits and softmax is applied
    ///   internally. This is more numerically stable, and the gradient is computed in one fused
    ///   `softmax(z) - y` step, so the last layer should output logits, not probabilities. If
    ///   `false`, `y_pred` must already be a normalized probability distribution per row
    ///
    /// # Returns
    ///
    /// - `CategoricalCrossEntropy` - the configured loss
    pub fn new(from_logits: bool) -> Self {
        Self { from_logits }
    }
}

/// Splits a `[..., classes]` tensor into `(prediction_sites, classes)`: the product of every
/// leading axis, and the length of the trailing class axis
///
/// The class axis is the **last** one, and it is the only axis the softmax normalizer and the
/// one-hot sum may run over. Every leading axis - the batch, plus any spatial or temporal
/// positions - indexes an *independent* prediction site, so each contributes to the divisor
/// rather than to the class dimension. This is Keras' rule: reduce the last axis, then divide by
/// the element count of what remains (`sum_over_batch_size`, whose divisor is
/// `prod(shape(values))` after the class axis is gone)
fn leading_and_classes(t: &Tensor) -> (usize, usize) {
    let ndim = t.ndim();
    let classes = t.shape()[ndim - 1];
    let sites: usize = t.shape()[..ndim - 1].iter().product();
    (sites, classes)
}

/// Validates that one-hot targets and predictions are non-empty, at least 2D, and shape-compatible
///
/// The two tensors must share the same shape, and the input must be at least 2D so that a class
/// axis and at least one prediction-site axis both exist. With a 1D tensor there would be no
/// leading axis at all, leaving a divisor of 1 and a softmax over the only axis present - a
/// single prediction dressed up as a batch
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
            // Fused softmax-cross-entropy. The reshape merges only the *leading* axes, so the
            // softmax below normalizes within one prediction site and never across sites
            let logits = y_pred
                .to_shape((sites, classes))
                .map_err(|e| Error::computation(format!("CCE logits reshape failed: {e}")))?;
            let labels = y_true
                .to_shape((sites, classes))
                .map_err(|e| Error::computation(format!("CCE labels reshape failed: {e}")))?;
            let (log_sm, _) = stable_log_softmax_softmax(&logits.view());
            return Ok(-(&labels * &log_sm).sum() / n);
        }

        // Probability path. The same reshape as above, because Keras' renormalizer runs over the
        // class axis alone - a `[batch, h, w, classes]` head is one distribution per pixel
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
            // Fused gradient w.r.t. the logits, per prediction site (see `compute_loss`)
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

        // Probability path. The row-normalizer is part of the function being differentiated, so
        // the gradient of `-sum_c y_c * ln(p_c / s)` w.r.t. `p_k` is `(sum_c y_c) / s - y_k / p_k`
        // - the old `-y_k / p_k` plus a term that is *constant across the row*. A softmax backward
        // annihilates any row-constant vector, so a softmax head trains identically either way;
        // the term matters only when the loss is wired to something else
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
                // One-hot targets make this 1, but the loss is defined for any target mass
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
