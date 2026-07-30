//! Regression metrics comparing ground-truth and predicted values.
//!
//! Provides MSE, RMSE, MAE, R^2, explained variance, median absolute error, and MAPE.

use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix1};

use super::validate_pair;

/// Calculates the Mean Squared Error (MSE) between ground-truth and predicted values.
///
/// MSE is the average of the squared differences between predictions and ground truth. Because
/// the per-sample error is squared, the order of the 2 arguments does not affect the result.
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
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
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

/// Calculates the Root Mean Squared Error (RMSE) between ground-truth and predicted values.
///
/// RMSE is the square root of the [`mean_squared_error`], giving an error in the same units as
/// the original data. Because MSE is non-negative, the square root is always well-defined.
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
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
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

/// Calculates the Mean Absolute Error (MAE) between ground-truth and predicted values.
///
/// MAE is the average absolute difference between predictions and ground truth, ignoring the
/// direction of the error. The order of the 2 arguments does not affect the result.
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
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
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

/// Calculates the R-squared (coefficient of determination) score.
///
/// R^2 measures how well predictions explain the variance in the ground truth. The formula is
/// `R^2 = 1 - SSE / SST`. Here `SSE = sum(y_pred - y_true)^2` and
/// `SST = sum(y_true - mean(y_true))^2`. SST comes from `y_true` alone, so the argument order
/// changes the result.
///
/// When every entry of `y_true` is identical, the score is undefined. By convention, this returns
/// `1.0` for an exact fit (`SSE == 0`) and `0.0` otherwise, matching scikit-learn's `r2_score`.
/// The function recognizes the constant case from the values themselves. A target that varies
/// only slightly (for example `[1e-6, 2e-6, 3e-6]`) is scored normally, not mistaken for a
/// constant.
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
/// # NaN Handling
///
/// Unlike [`explained_variance_score`], this does **not** skip non-finite samples. `SSE` and
/// `SST` are plain sums, so a single `NaN`/`inf` in `y_true` or `y_pred` propagates and makes the
/// result `NaN`. That surfaces bad data rather than hiding it. Prefer it when you want corruption
/// to be visible. Clean the inputs beforehand if you do not.
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
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
pub fn r2_score<S1, S2>(y_true: &ArrayBase<S1, Ix1>, y_pred: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    let mean = y_true.mean().unwrap();
    let sse: f64 = y_pred
        .iter()
        .zip(y_true.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum();

    // A constant `y_true` leaves R^2 undefined. By convention, a perfect fit scores 1.0 and
    // anything else scores 0.0.
    //
    // The check compares the raw values instead of testing `sst == 0.0`. Floating-point sums do
    // not round-trip every constant. 3 copies of `0.1` leave `sst` at 5.8e-34, not zero.
    // A threshold on `sst` could miss real constants or flag a low-variance target as constant.
    let mut values = y_true.iter();
    let first = *values.next().expect("validate_pair rejects an empty input");
    if values.all(|&v| v == first) {
        return if sse == 0.0 { 1.0 } else { 0.0 };
    }

    let sst: f64 = y_true.mapv(|x| (x - mean).powi(2)).sum();

    1.0 - sse / sst
}

/// Calculates the explained variance score.
///
/// Uses `1 - Var(y_true - y_pred) / Var(y_true)`. Unlike [`r2_score`], the numerator is the
/// variance of the residuals rather than their mean square, so a constant prediction bias does
/// not lower the score. The best possible value is 1.0. When `y_true` has zero variance, the
/// score is undefined: it returns 1.0 for residuals of zero variance, and 0.0 otherwise.
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
/// # NaN Handling
///
/// This uses a NaN-skipping variance. It **silently skips** non-finite samples and averages over
/// the finite subset. Unlike [`r2_score`], where a `NaN` propagates to a `NaN` result, a few
/// `NaN`/`inf` entries here leave a normal-looking score computed from the rest. This is
/// convenient but can mask corrupt data. Validate the inputs first if a silently dropped sample
/// would be a problem.
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
/// // predictions are biased (r2_score would be lower here).
/// assert!((explained_variance_score(&y_true, &y_pred) - 1.0).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
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

    // Computes population variance over the finite subset. A constant (or empty) subset has zero
    // spread by definition. The check compares the raw values instead of testing the computed
    // sum of squares for zero. Floating-point sums do not round-trip every constant (8 copies
    // of `0.1` do not), so a near-zero threshold could misjudge real data.
    let variance = |v: ArrayView1<f64>| -> f64 {
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        let mut first: Option<f64> = None;
        let mut constant = true;
        for &val in v.iter() {
            if !val.is_finite() {
                continue;
            }
            sum += val;
            count += 1;
            match first {
                None => first = Some(val),
                Some(f) => constant &= val == f,
            }
        }
        if count == 0 || constant {
            return 0.0;
        }
        let mean = sum / count as f64;
        let ss = v.iter().fold(0.0, |acc, &val| {
            if val.is_finite() {
                let d = val - mean;
                acc + d * d
            } else {
                acc
            }
        });
        ss / count as f64
    };
    let residual_variance = variance(residuals.view());
    let true_variance = variance(y_true.view());

    // Same convention as [`r2_score`]. A constant `y_true` leaves the ratio undefined: residuals
    // of zero variance score 1.0, anything else scores 0.0. Both checks use exact comparisons.
    if true_variance == 0.0 {
        return if residual_variance == 0.0 { 1.0 } else { 0.0 };
    }

    1.0 - residual_variance / true_variance
}

/// Calculates the median absolute error between ground-truth and predicted values.
///
/// The median of the absolute errors is robust to outliers, so a few large mistakes do not
/// dominate it the way they do in [`mean_absolute_error`].
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
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::median_absolute_error;
///
/// let y_true = array![1.0, 2.0, 3.0, 4.0];
/// let y_pred = array![1.0, 2.0, 3.0, 10.0]; // one large outlier error of 6
/// // Sorted absolute errors are [0, 0, 0, 6]. The median is 0.0, unmoved by the outlier.
/// assert!((median_absolute_error(&y_true, &y_pred) - 0.0).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
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

/// Calculates the mean absolute percentage error (MAPE) between ground-truth and predicted
/// values.
///
/// Uses `mean(|y_true - y_pred| / max(|y_true|, eps))`. The result is a fraction. Multiply by
/// 100 for a percentage. Each denominator is floored at a tiny epsilon. This keeps a zero true
/// value from causing a division by zero. The score can still be inflated when the true value
/// is near zero.
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
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths.
/// - Panics if the inputs are empty.
pub fn mean_absolute_percentage_error<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    const EPS: f64 = f64::EPSILON;
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs() / t.abs().max(EPS))
        .sum();

    sum / y_true.len() as f64
}
