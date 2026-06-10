//! Model-evaluation metrics for classification, clustering, and regression
//!
//! Each metric lives in a category submodule and is re-exported at the module root, so it is
//! reachable both by category (`metrics::regression::mean_squared_error`) and flat
//! (`metrics::mean_squared_error`); the crate and its prelude use the flat paths

/// Classification metrics: confusion matrices, accuracy, ROC/PR curves, log loss, and Cohen's kappa
pub mod classification;
/// Clustering metrics: mutual information (NMI/AMI), adjusted Rand index, and silhouette
pub mod clustering;
/// Regression metrics: MSE, RMSE, MAE, R-squared, explained variance, and MAPE
pub mod regression;

pub use classification::{
    Average, ConfusionMatrix, MulticlassConfusionMatrix, accuracy, average_precision, cohen_kappa,
    log_loss, precision_recall_curve, roc_auc, roc_curve, top_k_accuracy,
};
pub use clustering::{
    adjusted_mutual_info, adjusted_rand_index, calinski_harabasz_score, completeness_score,
    davies_bouldin_score, fowlkes_mallows_score, homogeneity_score, normalized_mutual_info,
    silhouette_score, v_measure_score,
};
pub use regression::{
    explained_variance_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, r2_score, root_mean_squared_error,
};

/// Validates a `(y_true, y_pred)`-style pair of inputs: equal length and non-empty
///
/// The length check runs first, so a length mismatch is reported even when one input is empty
/// Panic messages mirror [`crate::error::Error::DimensionMismatch`] and
/// [`crate::error::Error::EmptyInput`], keeping this lightweight tier consistent with the rest of
/// the crate without depending on the `error` module
///
/// # Parameters
///
/// - `expected` - expected length, taken from the first input
/// - `found` - actual length of the second input
/// - `what` - names the inputs for the empty-input message (e.g. `"y_true and y_pred"`)
///
/// # Panics
///
/// - when `expected != found`
/// - when `expected == 0` (empty input)
#[inline]
fn validate_pair(expected: usize, found: usize, what: &str) {
    if expected != found {
        panic!("dimension mismatch: expected {expected}, found {found}");
    }
    if expected == 0 {
        panic!("input is empty: {what}");
    }
}
