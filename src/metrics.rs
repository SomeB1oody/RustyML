//! Model-evaluation metrics for classification, clustering, and regression
//!
//! Each metric lives in a category submodule and the module root re-exports it. You can reach a
//! metric by category (`metrics::regression::mean_squared_error`) or by the flat path
//! (`metrics::mean_squared_error`). The crate and its prelude use the flat paths.
//!
//! # Regression metrics
//! - **mean_squared_error** / **root_mean_squared_error**: average squared error and its root
//! - **mean_absolute_error** / **median_absolute_error**: mean and (outlier-robust) median
//!   absolute error
//! - **mean_absolute_percentage_error**: mean error relative to the true values
//! - **r2_score**: coefficient of determination (R^2)
//! - **explained_variance_score**: residual variance relative to the data variance (ignores
//!   constant bias)
//!
//! # Classification metrics
//! - [`ConfusionMatrix`](crate::metrics::ConfusionMatrix): binary TP/FP/TN/FN with derived
//!   accuracy, precision, recall, specificity, F1, error rate, balanced accuracy, and MCC
//! - [`MulticlassConfusionMatrix`](crate::metrics::MulticlassConfusionMatrix): KxK matrix with
//!   per-class precision/recall/F1/support and macro/micro/weighted aggregation via
//!   [`Average`](crate::metrics::Average)
//! - **accuracy**, **roc_auc**, **log_loss**, **cohen_kappa**, **top_k_accuracy**,
//!   **average_precision**, and the **roc_curve** / **precision_recall_curve** threshold sweeps
//!
//! # Clustering metrics
//! - **adjusted_rand_index**, **normalized_mutual_info**, **adjusted_mutual_info**
//! - **homogeneity_score** / **completeness_score** / **v_measure_score**,
//!   **fowlkes_mallows_score**
//! - **silhouette_score**, **davies_bouldin_score**, **calinski_harabasz_score** (internal
//!   indices, no ground truth needed)
//!
//! # Conventions
//!
//! - **Panics instead of returning `Result`.** The functions in `metrics` panic on a precondition
//!   violation, such as mismatched lengths or empty input. This follows `ndarray` on a dimension
//!   mismatch, rather than returning the crate's `Error`. The panic messages mirror that type's
//!   wording (`dimension mismatch: ...`, `input is empty: ...`).
//! - **2 argument conventions.** A hard-label function takes `(y_true, y_pred)`, ground truth
//!   first. This mirrors the clustering metrics' `(labels_true, labels_pred)`. Order does not
//!   affect the symmetric metrics (MSE, MAE, `accuracy`), but it does affect `r2_score` and the
//!   confusion matrix types. A threshold-based function instead takes `(labels, scores)`: `bool`
//!   ground truth first, then `f64` scores. `roc_auc`, `roc_curve`, `precision_recall_curve`, and
//!   `average_precision` use this second convention.
//!
//! # Examples
//!
//! ```rust
//! use rustyml::metrics::*;
//! use ndarray::array;
//!
//! // Regression evaluation. Arguments are (y_true, y_pred).
//! let y_true = array![2.8, 2.1, 3.3, 4.2];
//! let y_pred = array![3.0, 2.0, 3.5, 4.1];
//! let mse = mean_squared_error(&y_true.view(), &y_pred.view());
//! let r2 = r2_score(&y_true.view(), &y_pred.view());
//!
//! // Classification evaluation with a confusion matrix.
//! let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
//! let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];
//! let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
//! println!("F1 Score: {:.3}", cm.f1_score());
//!
//! // ROC AUC for binary classification.
//! let labels = array![false, true, false, true];
//! let scores = array![0.1, 0.4, 0.35, 0.8];
//! let auc = roc_auc(&labels.view(), &scores.view());
//! ```

/// Classification metrics: confusion matrices, accuracy, ROC/PR curves, log loss,
/// and Cohen's kappa.
pub mod classification;
/// Clustering metrics: mutual information (NMI/AMI), adjusted Rand index, and silhouette.
pub mod clustering;
/// Regression metrics: MSE, RMSE, MAE, R^2, explained variance, and MAPE.
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

/// Checks a `(y_true, y_pred)`-style pair of inputs for equal length and non-empty content.
///
/// The length check runs first, so a length mismatch is reported even when one input is empty.
/// The panic messages mirror [`crate::error::Error::DimensionMismatch`] and
/// [`crate::error::Error::EmptyInput`]. This keeps the lightweight `metrics` tier consistent with
/// the rest of the crate, without depending on the `error` module.
///
/// # Parameters
///
/// - `expected` - the expected length, taken from the first input
/// - `found` - the actual length of the second input
/// - `what` - names the inputs for the empty-input message, for example `"y_true and y_pred"`
///
/// # Panics
///
/// - Panics if `expected` does not equal `found`.
/// - Panics if `expected` is 0 (empty input).
#[inline]
fn validate_pair(expected: usize, found: usize, what: &str) {
    if expected != found || expected == 0 {
        fail_pair(expected, found, what);
    }
}

/// Out-of-line panic path for [`validate_pair`], marked `#[cold]` so the caller's hot path stays
/// down to 2 comparisons. Re-checks the length condition once, only to pick the right message.
#[cold]
#[inline(never)]
fn fail_pair(expected: usize, found: usize, what: &str) -> ! {
    if expected != found {
        panic!("dimension mismatch: expected {expected}, found {found}");
    }
    panic!("input is empty: {what}");
}
