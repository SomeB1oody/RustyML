//! Model-evaluation metrics for classification, clustering, and regression
//!
//! Each metric lives in a category submodule and is re-exported at the module root, so it is
//! reachable both by category (`metrics::regression::mean_squared_error`) and flat
//! (`metrics::mean_squared_error`); the crate and its prelude use the flat paths
//!
//! # Regression metrics
//! - **mean_squared_error** / **root_mean_squared_error**: average squared error and its root
//! - **mean_absolute_error** / **median_absolute_error**: mean and (outlier-robust) median absolute error
//! - **mean_absolute_percentage_error**: mean error relative to the true values
//! - **r2_score**: coefficient of determination (explained variance)
//! - **explained_variance_score**: variance of the residuals relative to the data (ignores constant bias)
//!
//! # Classification metrics
//! - [`ConfusionMatrix`](crate::metrics::ConfusionMatrix): binary TP/FP/TN/FN with derived accuracy,
//!   precision, recall, specificity, F1, error rate, balanced accuracy, and MCC
//! - [`MulticlassConfusionMatrix`](crate::metrics::MulticlassConfusionMatrix): KxK matrix with
//!   per-class precision/recall/F1/support and macro/micro/weighted aggregation via
//!   [`Average`](crate::metrics::Average)
//! - **accuracy**, **roc_auc**, **log_loss**, **cohen_kappa**, **top_k_accuracy**,
//!   **average_precision**, and the **roc_curve** / **precision_recall_curve** threshold sweeps
//!
//! # Clustering metrics
//! - **adjusted_rand_index**, **normalized_mutual_info**, **adjusted_mutual_info**
//! - **homogeneity_score** / **completeness_score** / **v_measure_score**, **fowlkes_mallows_score**
//! - **silhouette_score**, **davies_bouldin_score**, **calinski_harabasz_score** (internal indices, no ground truth needed)
//!
//! # Conventions
//!
//! - **Panics instead of returning `Result`**. `metrics` is a lightweight leaf module - pure
//!   `array -> scalar` functions pulling only `ndarray` and `ahash` - so, like `ndarray` on a
//!   dimension mismatch, the metrics panic on precondition violations (mismatched lengths, empty
//!   input) rather than returning the crate's `Error`. The panic messages mirror that crate's
//!   wording (`dimension mismatch: ...`, `input is empty: ...`) for consistency
//! - **Arguments are `(y_true, y_pred)`** - ground truth first, mirroring the clustering metrics'
//!   `(labels_true, labels_pred)`. The order is irrelevant for the symmetric metrics (MSE, MAE,
//!   accuracy) but significant for `r2_score`, `ConfusionMatrix::new`, and `roc_auc`
//!
//! # Examples
//!
//! ```rust
//! use rustyml::metrics::*;
//! use ndarray::array;
//!
//! // Regression evaluation - arguments are (y_true, y_pred)
//! let y_true = array![2.8, 2.1, 3.3, 4.2];
//! let y_pred = array![3.0, 2.0, 3.5, 4.1];
//! let mse = mean_squared_error(&y_true.view(), &y_pred.view());
//! let r2 = r2_score(&y_true.view(), &y_pred.view());
//!
//! // Classification evaluation with confusion matrix
//! let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
//! let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];
//! let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
//! println!("F1 Score: {:.3}", cm.f1_score());
//!
//! // ROC AUC for binary classification
//! let labels = array![false, true, false, true];
//! let scores = array![0.1, 0.4, 0.35, 0.8];
//! let auc = roc_auc(&labels.view(), &scores.view());
//! ```

/// Classification metrics: confusion matrices, accuracy, ROC/PR curves, log loss, and Cohen's kappa
pub mod classification;
/// Clustering metrics: mutual information (NMI/AMI), adjusted Rand index, and silhouette
pub mod clustering;
/// Regression metrics: MSE, RMSE, MAE, R^2, explained variance, and MAPE
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
    if expected != found || expected == 0 {
        fail_pair(expected, found, what);
    }
}

/// Cold, out-of-line panic path for [`validate_pair`], keeping its inlined hot path down to the two
/// comparisons. Re-checks the length condition only to select the right message
#[cold]
#[inline(never)]
fn fail_pair(expected: usize, found: usize, what: &str) -> ! {
    if expected != found {
        panic!("dimension mismatch: expected {expected}, found {found}");
    }
    panic!("input is empty: {what}");
}
