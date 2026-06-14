//! Loss functions for neural network training
//!
//! Provides regression losses (MSE, MAE) and classification losses
//! (binary, categorical, and sparse-categorical cross-entropy), along with
//! shared helpers for shape validation and probability clipping

use crate::error::Error;
use crate::neural_network::Tensor;
use ndarray::{Array2, ArrayView2};

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
