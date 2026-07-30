//! Loss functions for neural network training.
//!
//! Provides regression losses (MSE, MAE) and classification losses (binary, categorical, and
//! sparse-categorical cross-entropy), along with shared helpers for shape validation and
//! probability clipping.

use crate::error::Error;
use crate::neural_network::Tensor;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

/// Epsilon used to clip predicted probabilities away from 0 and 1, so the cross-entropy losses
/// avoid `log(0)` and division by zero.
const PROB_CLIP_EPS: f32 = 1e-7;

/// Row-wise numerically stable log-softmax and softmax over the last axis of a `[batch, classes]`
/// view, returned as `(log_softmax, softmax)`.
///
/// The `from_logits` cross-entropy paths use this function. It subtracts each row's max before
/// it exponentiates, so `exp` does not overflow. It computes `log_softmax` as `z - logsumexp(z)`
/// directly, instead of `ln(softmax)`, to avoid the precision loss of taking the log of a
/// clipped probability.
fn stable_log_softmax_softmax(logits: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
    let mut log_sm = logits.to_owned();
    for mut row in log_sm.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        let log_sum_exp = max + sum_exp.ln();
        row.mapv_inplace(|x| x - log_sum_exp);
    }
    let sm = log_sm.mapv(|v| v.exp());
    (log_sm, sm)
}

/// Validates that `y_true` and `y_pred` have identical shapes.
///
/// The losses that require element-wise correspondence between targets and predictions share
/// this function. It catches mismatches before a downstream ndarray broadcast panics.
///
/// # Errors
///
/// Returns [`Error::shape_mismatch`] when the 2 shapes differ.
fn validate_same_shape(y_true: &Tensor, y_pred: &Tensor) -> Result<(), Error> {
    if y_true.shape() != y_pred.shape() {
        return Err(Error::shape_mismatch(y_true.shape(), y_pred.shape()));
    }
    Ok(())
}

/// Clips every element of a probability tensor into `[PROB_CLIP_EPS, 1 - PROB_CLIP_EPS]`.
///
/// `BinaryCrossEntropy` uses this function to guard its `log` calls. The categorical and
/// sparse-categorical losses clip through `normalize_and_clip_rows` instead, so all 3 losses
/// still share the same `PROB_CLIP_EPS` constant.
fn clip_probabilities(probs: &Tensor) -> Tensor {
    let mut clipped = probs.clone();
    clipped.par_mapv_inplace(|x| x.clamp(PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS));
    clipped
}

/// Row-normalizes a `[sites, classes]` probability matrix, then clips it away from 0 and 1.
///
/// Matches Keras' probability path. It divides each row by its sum before the clip. The loss
/// then scores a row that does not sum to 1 as the distribution it implies. A softmax row
/// already sums to 1, so the division does not change the loss value, but
/// `CategoricalCrossEntropy`'s gradient still accounts for it.
///
/// # Returns
///
/// - `(Array2<f32>, Array1<f32>)` - the normalized and clipped probabilities, and the row sums
///   used to normalize them
///
/// # Notes
///
/// A row that sums to zero produces a non-finite result, as in Keras. The clip guards against
/// `log(0)`, not a zero divisor.
fn normalize_and_clip_rows(probs: &ArrayView2<f32>) -> (Array2<f32>, Array1<f32>) {
    let row_sums = probs.sum_axis(Axis(1));
    let mut normalized = probs.to_owned();
    Zip::from(normalized.rows_mut())
        .and(&row_sums)
        .for_each(|mut row, &sum| {
            row.mapv_inplace(|p| (p / sum).clamp(PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS));
        });
    (normalized, row_sums)
}

/// Binary Cross Entropy loss function for binary classification.
pub mod binary_cross_entropy;
/// Categorical Cross Entropy loss function for multi-class classification.
pub mod categorical_cross_entropy;
/// Mean Absolute Error loss function.
pub mod mean_absolute_error;
/// Mean Squared Error loss function.
pub mod mean_squared_error;
/// Sparse Categorical Cross Entropy loss function for multi-class classification.
pub mod sparse_categorical_cross_entropy;

pub use binary_cross_entropy::BinaryCrossEntropy;
pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use mean_absolute_error::MeanAbsoluteError;
pub use mean_squared_error::MeanSquaredError;
pub use sparse_categorical_cross_entropy::SparseCategoricalCrossEntropy;
