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
///
/// # Input rank
///
/// The **last** axis is the class axis; every axis before it indexes an independent prediction
/// site. So `[batch, classes]` is one prediction per sample, while the `[batch, height, width,
/// classes]` output of a channels-last `Conv2D` softmax head is one prediction per pixel. The
/// loss sums over the class axis and divides by the total number of sites (`batch * height *
/// width`), which is what Keras' default `sum_over_batch_size` reduction computes; under
/// `from_logits` the softmax likewise normalizes within a site and never across sites
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

        // Probability path
        let y_pred_clipped = clip_probabilities(y_pred);
        let losses = y_true * &y_pred_clipped.mapv(|y_p| y_p.ln());
        Ok(-losses.sum() / n)
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

        // Probability path: gradient is -y_true / y_pred
        let y_pred_clipped = clip_probabilities(y_pred);
        let grad = -y_true / &y_pred_clipped;
        Ok(grad / n)
    }
}
