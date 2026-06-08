// Each metric lives in a category submodule and is also re-exported at the module root, so it is
// reachable both by category (`metric::regression::mean_squared_error`) and flat
// (`metric::mean_squared_error`). The flat paths are the ones the crate and its prelude use.

/// Classification metrics: confusion matrix, accuracy, and ROC AUC.
pub mod classification;
/// Clustering metrics: mutual information (NMI/AMI), adjusted Rand index, and silhouette.
pub mod clustering;
/// Regression metrics: MSE, RMSE, MAE, and R2.
pub mod regression;

pub use classification::{ConfusionMatrix, accuracy, roc_auc};
pub use clustering::{
    adjusted_mutual_info, adjusted_rand_index, normalized_mutual_info, silhouette_score,
};
pub use regression::{mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error};

/// Validates a `(y_true, y_pred)`-style pair of inputs: equal length and non-empty.
///
/// The length check runs first, so a length mismatch is reported even when one input is empty.
/// Panics with messages that mirror [`crate::error::Error::DimensionMismatch`] and
/// [`crate::error::Error::EmptyInput`], keeping the lightweight `metric` tier consistent with the
/// rest of the crate without depending on the `error` module. `what` names the inputs for the
/// empty-input message (e.g. `"y_true and y_pred"`).
#[inline]
fn validate_pair(expected: usize, found: usize, what: &str) {
    if expected != found {
        panic!("dimension mismatch: expected {expected}, found {found}");
    }
    if expected == 0 {
        panic!("input is empty: {what}");
    }
}
