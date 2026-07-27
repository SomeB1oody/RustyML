//! Loss functions for neural network training
//!
//! Provides regression losses (MSE, MAE) and classification losses
//! (binary, categorical, and sparse-categorical cross-entropy), along with
//! shared helpers for shape validation and probability clipping

use crate::error::Error;
use crate::neural_network::Tensor;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

/// Epsilon used to clip predicted probabilities into the open interval `(0, 1)`,
/// preventing `log(0)` and division-by-zero in the cross-entropy losses
const PROB_CLIP_EPS: f32 = 1e-7;

/// Row-wise numerically stable log-softmax and softmax over the last axis of a `[batch, classes]`
/// view, returned as `(log_softmax, softmax)`
///
/// Used by the `from_logits` cross-entropy paths. Subtracting each row's max before exponentiating
/// keeps `exp` from overflowing. Computing `log_softmax = z - logsumexp(z)` directly, rather than
/// `ln(softmax)`, avoids the precision loss of logging a clipped probability
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

/// Validates that `y_true` and `y_pred` have identical shapes
///
/// Shared by the losses that require element-wise correspondence between targets
/// and predictions, catching mismatches before a downstream ndarray broadcast panics
///
/// # Errors
///
/// Returns [`Error::shape_mismatch`] when the two shapes differ
fn validate_same_shape(y_true: &Tensor, y_pred: &Tensor) -> Result<(), Error> {
    if y_true.shape() != y_pred.shape() {
        return Err(Error::shape_mismatch(y_true.shape(), y_pred.shape()));
    }
    Ok(())
}

/// Clips every element of a probability tensor into `[PROB_CLIP_EPS, 1 - PROB_CLIP_EPS]`
///
/// Shared by the cross-entropy losses (binary/categorical/sparse-categorical) so the
/// numerical-stability constant lives in exactly one place
fn clip_probabilities(probs: &Tensor) -> Tensor {
    let mut clipped = probs.clone();
    clipped.par_mapv_inplace(|x| x.clamp(PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS));
    clipped
}

/// Row-normalizes a `[sites, classes]` probability matrix, then clips it away from 0 and 1
///
/// This is Keras' probability path for the two categorical cross-entropies, in Keras' order:
/// `output / sum(output, axis=-1, keepdims=True)` runs **before** the clip, so a head whose rows
/// do not sum to 1 is scored as the distribution it implies rather than taken at face value. For
/// a genuine softmax row the division is a numerical no-op, but it is still part of the function
/// being differentiated - see the gradient derivation in
/// [`CategoricalCrossEntropy`](crate::neural_network::losses::CategoricalCrossEntropy)
///
/// # Returns
///
/// - `(Array2<f32>, Array1<f32>)` - the normalized and clipped probabilities, and the raw row sums
///   that normalized them. The gradient needs those sums, so they are returned rather than recomputed
///
/// A row summing to zero yields a non-finite result, exactly as it does in Keras; the clip guards
/// `log(0)`, not a degenerate divisor
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

/// Binary Cross Entropy loss function for binary classification
pub mod binary_cross_entropy;
/// Categorical Cross Entropy loss function for multi-class classification
pub mod categorical_cross_entropy;
/// Mean Absolute Error loss function
pub mod mean_absolute_error;
/// Mean Squared Error loss function
pub mod mean_squared_error;
/// Sparse Categorical Cross Entropy loss function for multi-class classification
pub mod sparse_categorical_cross_entropy;

pub use binary_cross_entropy::BinaryCrossEntropy;
pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use mean_absolute_error::MeanAbsoluteError;
pub use mean_squared_error::MeanSquaredError;
pub use sparse_categorical_cross_entropy::SparseCategoricalCrossEntropy;
