use crate::error::ModelError;
use crate::neural_network::Tensor;

/// Epsilon used to clip predicted probabilities into the open interval `(0, 1)`,
/// preventing `log(0)` and division-by-zero in the cross-entropy losses.
pub(crate) const PROB_CLIP_EPS: f32 = 1e-7;

/// Validates that `y_true` and `y_pred` have identical shapes, returning a descriptive error
/// instead of letting a downstream ndarray broadcast mismatch panic. Shared by the losses that
/// require element-wise correspondence between targets and predictions.
pub(crate) fn validate_same_shape(y_true: &Tensor, y_pred: &Tensor) -> Result<(), ModelError> {
    if y_true.shape() != y_pred.shape() {
        return Err(ModelError::InputValidationError(format!(
            "Loss inputs must have the same shape, got y_true {:?} and y_pred {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }
    Ok(())
}

/// Clips every element of a probability tensor into `[PROB_CLIP_EPS, 1 - PROB_CLIP_EPS]`.
///
/// Shared by the cross-entropy losses (binary/categorical/sparse-categorical) so the
/// numerical-stability constant lives in exactly one place.
pub(crate) fn clip_probabilities(probs: &Tensor) -> Tensor {
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
