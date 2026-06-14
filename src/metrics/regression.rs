//! Regression metrics comparing ground-truth and predicted values
//!
//! Provides MSE, RMSE, MAE, R^2, explained variance, median absolute error, and MAPE

use ndarray::{Array1, ArrayBase, Data, Ix1};

use super::validate_pair;
use crate::math::{sum_of_square_total, sum_of_squared_errors, variance};

/// Variance below which the total sum of squares is treated as zero (all `y_true` identical)
const SST_EPSILON: f64 = 1e-10;

/// Calculates the Mean Squared Error (MSE) between ground-truth and predicted values
///
/// MSE is the average of the squared differences between predictions and ground truth. Because the
/// per-sample error is squared, the order of the two arguments does not affect the result
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - Mean squared error
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::mean_squared_error;
///
/// let y_true = array![3.0, -0.5, 2.0, 7.0];
/// let y_pred = array![2.5, 0.0, 2.1, 7.8];
/// let mse = mean_squared_error(&y_true, &y_pred);
/// // MSE = ((3.0 - 2.5)^2 + (-0.5 - 0.0)^2 + (2.0 - 2.1)^2 + (7.0 - 7.8)^2) / 4
/// //     = (0.25 + 0.25 + 0.01 + 0.64) / 4 = 0.2875
/// assert!((mse - 0.2875).abs() < 1e-10);
/// ```
pub fn mean_squared_error<S1, S2>(y_true: &ArrayBase<S1, Ix1>, y_pred: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    let sum_squared_error = y_true.iter().zip(y_pred.iter()).fold(0.0, |acc, (&t, &p)| {
        let error = t - p;
        acc + error * error
    });

    sum_squared_error / y_true.len() as f64
}

/// Calculates the Root Mean Squared Error (RMSE) between ground-truth and predicted values
///
/// RMSE is the square root of the [`mean_squared_error`], giving an error in the same units as the
/// original data. As MSE is non-negative, the square root is always well-defined
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - Root mean squared error
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::root_mean_squared_error;
///
/// let y_true = array![1.0, 2.0, 3.0];
/// let y_pred = array![2.0, 3.0, 4.0];
/// let rmse = root_mean_squared_error(&y_true, &y_pred);
/// // RMSE = sqrt(((2 - 1)^2 + (3 - 2)^2 + (4 - 3)^2) / 3) = sqrt(3/3) = 1.0
/// assert!((rmse - 1.0).abs() < 1e-6);
/// ```
pub fn root_mean_squared_error<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Calculates the Mean Absolute Error (MAE) between ground-truth and predicted values
///
/// MAE is the average absolute difference between predictions and ground truth, ignoring the
/// direction of the error. The order of the two arguments does not affect the result
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - Mean absolute error
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::mean_absolute_error;
///
/// let y_true = array![1.0, 2.0, 3.0];
/// let y_pred = array![2.0, 3.0, 4.0];
/// let mae = mean_absolute_error(&y_true, &y_pred);
/// // MAE = (|2 - 1| + |3 - 2| + |4 - 3|) / 3 = (1 + 1 + 1) / 3 = 1.0
/// assert!((mae - 1.0).abs() < 1e-6);
/// ```
pub fn mean_absolute_error<S1, S2>(y_true: &ArrayBase<S1, Ix1>, y_pred: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    let sum_absolute_error = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(0.0, |acc, (&t, &p)| acc + (t - p).abs());

    sum_absolute_error / y_true.len() as f64
}

/// Calculates the R-squared (coefficient of determination) score
///
/// R^2 measures how well predictions explain the variance in the ground truth, using
/// `R^2 = 1 - SSE / SST` where `SSE = sum(y_pred - y_true)^2` and `SST = sum(y_true - mean(y_true))^2`,
/// and since SST is computed from `y_true` alone, the argument order is significant
///
/// When `y_true` has (near-)zero variance the score is undefined; following scikit-learn, this
/// returns `1.0` for a perfect fit (`SSE ~= 0`) and `0.0` otherwise
///
/// # NaN handling
///
/// Unlike [`explained_variance_score`], this does **not** skip non-finite samples: `SSE` and `SST`
/// are plain sums, so a single `NaN`/`inf` in `y_true` or `y_pred` propagates and makes the result
/// `NaN`. That surfaces bad data rather than hiding it. Prefer it when you want corruption to be
/// visible, and clean the inputs beforehand if you do not
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - R-squared value (typically in `(-inf, 1.0]`)
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::r2_score;
///
/// let y_true = array![1.0, 3.0, 5.0];
/// let y_pred = array![2.0, 3.0, 4.0];
/// let r2 = r2_score(&y_true, &y_pred);
/// // mean(y_true) = 3, SST = 4 + 0 + 4 = 8, SSE = 1 + 0 + 1 = 2, so R^2 = 1 - 2/8 = 0.75
/// assert!((r2 - 0.75).abs() < 1e-6);
/// ```
pub fn r2_score<S1, S2>(y_true: &ArrayBase<S1, Ix1>, y_pred: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    // SSE is symmetric; SST depends only on y_true
    let sse = sum_of_squared_errors(y_pred, y_true);
    let sst = sum_of_square_total(y_true);

    // When all y_true are identical SST is zero and R^2 is undefined: perfect fit scores 1.0, else 0.0
    if sst < SST_EPSILON {
        return if sse < SST_EPSILON { 1.0 } else { 0.0 };
    }

    1.0 - sse / sst
}

/// Calculates the explained variance score
///
/// `1 - Var(y_true - y_pred) / Var(y_true)`. Unlike [`r2_score`], the numerator is the variance of
/// the residuals rather than their mean square, so a constant prediction bias does not lower the
/// score. The best possible value is 1.0; when `y_true` has zero variance the score is undefined
/// and returns 1.0 for residuals of zero variance, otherwise 0.0
///
/// # NaN handling
///
/// This goes through [`crate::math::variance`], which **silently skips** non-finite samples and
/// averages over the finite subset. So unlike [`r2_score`] (where a `NaN` propagates to a `NaN`
/// result), a few `NaN`/`inf` entries here leave a normal-looking score computed from the rest,
/// which is convenient but can mask corrupt data. Validate the inputs if a silently dropped sample
/// would be a problem
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - Explained variance score (typically in `(-inf, 1.0]`)
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::explained_variance_score;
///
/// let y_true = array![1.0, 2.0, 3.0];
/// let y_pred = array![2.0, 3.0, 4.0]; // a constant +1 bias
/// // The residuals are all -1, so their variance is 0 and the score is 1.0, even though the
/// // predictions are biased (r2_score would be lower here)
/// assert!((explained_variance_score(&y_true, &y_pred) - 1.0).abs() < 1e-12);
/// ```
pub fn explained_variance_score<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    let residuals: Array1<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t - p)
        .collect();
    let residual_variance = variance(&residuals);
    let true_variance = variance(y_true);

    if true_variance < SST_EPSILON {
        return if residual_variance < SST_EPSILON {
            1.0
        } else {
            0.0
        };
    }

    1.0 - residual_variance / true_variance
}

/// Calculates the median absolute error between ground-truth and predicted values
///
/// The median of the absolute errors is robust to outliers, so a few large mistakes do not
/// dominate it the way they do [`mean_absolute_error`]
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - Median absolute error (>= 0.0)
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::median_absolute_error;
///
/// let y_true = array![1.0, 2.0, 3.0, 4.0];
/// let y_pred = array![1.0, 2.0, 3.0, 10.0]; // one large outlier error of 6
/// // Sorted absolute errors are [0, 0, 0, 6]; the median is 0.0, unmoved by the outlier
/// assert!((median_absolute_error(&y_true, &y_pred) - 0.0).abs() < 1e-12);
/// ```
pub fn median_absolute_error<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    let mut errors: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .collect();
    errors.sort_by(|a, b| a.total_cmp(b));

    let n = errors.len();
    if n % 2 == 1 {
        errors[n / 2]
    } else {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    }
}

/// Calculates the mean absolute percentage error (MAPE) between ground-truth and predicted values
///
/// `mean(|y_true - y_pred| / max(|y_true|, eps))`. The result is a fraction (multiply by 100 for a
/// percentage). Each denominator is floored at a tiny epsilon, so samples whose true value is zero
/// do not cause a division by zero but can still inflate the score
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Returns
///
/// - `f64` - Mean absolute percentage error as a fraction (>= 0.0)
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::mean_absolute_percentage_error;
///
/// let y_true = array![2.0, 4.0, 5.0];
/// let y_pred = array![1.0, 4.0, 5.0];
/// // Per-sample ratios are 0.5, 0, 0, so MAPE = 0.5 / 3 = 0.1666...
/// assert!((mean_absolute_percentage_error(&y_true, &y_pred) - 0.166666667).abs() < 1e-6);
/// ```
pub fn mean_absolute_percentage_error<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    // Floor on |y_true| to keep the division finite when a true value is zero
    const EPS: f64 = 1e-15;
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs() / t.abs().max(EPS))
        .sum();

    sum / y_true.len() as f64
}
