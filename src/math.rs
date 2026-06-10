//! Free math and statistics functions: variances, distances, losses, and impurity measures
//!
//! Each function returns a plain value and may panic on malformed input rather than returning a
//! `Result`. The module covers regression statistics (SST, SSE, variance, standard deviation),
//! distance metrics, classification losses, decision-tree impurity (gini, entropy), and the
//! isolation-forest path-length correction

use ahash::AHashMap;
use ndarray::{ArrayBase, Data, Ix1};

const EULER_GAMMA: f64 = 0.57721566490153286060651209008240243104215933593992;

/// Calculates the total sum of squares (SST)
///
/// SST measures the total variability in the data as the sum of squared
/// differences between each value and the mean of all values
///
/// # Parameters
///
/// - `values` - Observed values stored in a 1D array
///
/// # Returns
///
/// - `f64` - Total sum of squares for the provided values
///
/// # Examples
///
/// ```rust
/// use rustyml::math::sum_of_square_total;
/// use ndarray::array;
///
/// let values = array![1.0, 2.0, 3.0];
/// let sst = sum_of_square_total(&values);
/// // Mean is 2.0, so SST = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2.0
/// assert!((sst - 2.0).abs() < 1e-5);
/// ```
#[inline]
pub fn sum_of_square_total<S>(values: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.mean().unwrap();
    values.mapv(|x| (x - mean).powi(2)).sum()
}

/// Calculates the sum of squared errors (SSE)
///
/// SSE measures the total squared difference between predicted values and actual labels
///
/// # Parameters
///
/// - `predicted` - Predicted values vector
/// - `actual` - Actual values vector
///
/// # Returns
///
/// - `f64` - Sum of squared errors computed as sum((predicted_i - actual_i)^2)
///
/// # Examples
///
/// ```rust
/// use rustyml::math::sum_of_squared_errors;
/// use ndarray::array;
///
/// let predicted = array![2.0, 3.0];
/// let actual = array![1.0, 3.0];
/// let sse = sum_of_squared_errors(&predicted, &actual);
/// // (2-1)^2 + (3-3)^2 = 1 + 0 = 1
/// assert!((sse - 1.0).abs() < 1e-6);
/// ```
#[inline]
pub fn sum_of_squared_errors<S1, S2>(
    predicted: &ArrayBase<S1, Ix1>,
    actual: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let sum: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum();

    sum
}

/// Calculates the mean squared error (variance) of a set of values
///
/// The variance is the average of the squared differences between each value
/// and the mean
///
/// Non-finite entries (`NaN`/`+/-inf`) are **skipped**: both the mean and the variance are
/// computed over the finite subset, dividing by the count of finite values. A few bad
/// entries therefore do not poison the statistic
///
/// # Parameters
///
/// - `y` - Values for which to calculate the variance
///
/// # Returns
///
/// - `f64` - Population variance over the finite values (0.0 when the array is empty or has
///   no finite values)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::variance;
///
/// let values = array![1.0, 2.0, 3.0];
/// let mse = variance(&values);
/// // Mean is 2.0, so variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 ~= 0.66667
/// assert!((mse - 0.6666667).abs() < 1e-6);
/// ```
#[inline]
pub fn variance<S>(y: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    if y.is_empty() {
        return 0.0;
    }

    // Mean over the finite values only, in one pass: NaN/infinite entries are skipped so a
    // single bad element does not poison the statistic
    let (finite_sum, finite_count) = y.fold((0.0_f64, 0_usize), |(sum, count), &val| {
        if val.is_finite() {
            (sum + val, count + 1)
        } else {
            (sum, count)
        }
    });

    // All entries were NaN/infinite, so there is nothing to summarize
    if finite_count == 0 {
        return 0.0;
    }
    let mean = finite_sum / finite_count as f64;

    // Sum of squared differences over the same finite subset
    let sum_squared_diff = y.fold(0.0, |acc, &val| {
        if val.is_finite() {
            let diff = val - mean;
            acc + diff * diff
        } else {
            acc
        }
    });

    // Population variance: divide by the finite count, not the full length
    sum_squared_diff / finite_count as f64
}

/// Computes the logistic sigmoid for a scalar input
///
/// The sigmoid maps any real number into the open interval (0, 1) with clipping
/// for extreme values to preserve numerical stability
///
/// # Parameters
///
/// - `z` - Input value to transform
///
/// # Returns
///
/// - `f64` - Sigmoid output in the range (0, 1)
///
/// # Examples
///
/// ```rust
/// use rustyml::math::sigmoid;
///
/// let value = sigmoid(0.0);
/// // sigmoid(0) = 0.5
/// assert!((value - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    // Clamp extreme inputs to keep exp() from overflowing
    const MAX_SIGMOID_INPUT: f64 = 500.0;
    const MIN_SIGMOID_INPUT: f64 = -500.0;

    if z > MAX_SIGMOID_INPUT {
        return 1.0;
    } else if z < MIN_SIGMOID_INPUT {
        return 0.0;
    }

    1.0 / (1.0 + (-z).exp())
}

/// Calculates the logistic regression loss (log loss)
///
/// This computes the average cross-entropy loss by applying the sigmoid
/// to raw logits before evaluating the log-likelihood
///
/// # Parameters
///
/// - `logits` - Raw model outputs (logits before sigmoid)
/// - `actual_labels` - Binary labels (0 or 1)
///
/// # Returns
///
/// - `f64` - Average logistic regression loss
///
/// # Examples
///
/// ```rust
/// use rustyml::math::logistic_loss;
/// use ndarray::array;
///
/// let logits = array![0.0, 2.0, -1.0];
/// let actual_labels = array![0.0, 1.0, 0.0];
/// let loss = logistic_loss(&logits, &actual_labels);
/// // Expected average loss is approximately 0.37778
/// assert!((loss - 0.37778).abs() < 1e-5);
/// ```
#[inline]
pub fn logistic_loss<S1, S2>(logits: &ArrayBase<S1, Ix1>, actual_labels: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let n = logits.len() as f64;

    let total_loss = logits
        .iter()
        .zip(actual_labels.iter())
        .map(|(&x, &y)| {
            // Numerically stable log loss: max(0, x) - x*y + log(1 + exp(-|x|))
            x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln()
        })
        .sum::<f64>();

    total_loss / n
}

/// Calculates the mean hinge loss for margin-based classifiers (e.g. linear SVM)
///
/// For each sample the loss is `max(0, 1 - y_i * margin_i)`, averaged over all samples,
/// where the margin is the decision-function value `w*x_i + b`. Labels are expected in
/// `{-1, +1}`
///
/// # Parameters
///
/// - `margins` - Decision-function values for each sample
/// - `labels` - True labels in `{-1, +1}`
///
/// # Returns
///
/// - `f64` - Mean hinge loss (0.0 when the input is empty)
///
/// # Examples
///
/// ```rust
/// use rustyml::math::hinge_loss;
/// use ndarray::array;
///
/// let margins = array![0.8, -0.5, 2.0];
/// let labels = array![1.0, -1.0, 1.0];
/// // max(0,1-0.8)=0.2, max(0,1-0.5)=0.5, max(0,1-2)=0 -> mean = 0.7/3
/// let loss = hinge_loss(&margins, &labels);
/// assert!((loss - 0.2333333).abs() < 1e-5);
/// ```
#[inline]
pub fn hinge_loss<S1, S2>(margins: &ArrayBase<S1, Ix1>, labels: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let n = margins.len();
    if n == 0 {
        return 0.0;
    }

    margins
        .iter()
        .zip(labels.iter())
        .map(|(&m, &y)| (1.0 - y * m).max(0.0))
        .sum::<f64>()
        / n as f64
}

/// Calculates the squared Euclidean distance between two vectors
///
/// # Parameters
///
/// - `x1` - First vector
/// - `x2` - Second vector
///
/// # Returns
///
/// - `f64` - Squared Euclidean distance between the two vectors
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::squared_euclidean_distance_row;
///
/// let v1 = array![1.0, 2.0, 3.0];
/// let v2 = array![4.0, 5.0, 6.0];
/// let dist = squared_euclidean_distance_row(&v1, &v2);
/// // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
/// assert!((dist - 27.0).abs() < 1e-10);
/// ```
#[inline]
pub fn squared_euclidean_distance_row<S1, S2>(
    x1: &ArrayBase<S1, Ix1>,
    x2: &ArrayBase<S2, Ix1>,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let diff = x1 - x2;
    diff.mapv(|x| x * x).sum()
}

/// Calculates the Manhattan (L1) distance between two vectors
///
/// # Parameters
///
/// - `x1` - First vector
/// - `x2` - Second vector
///
/// # Returns
///
/// - `f64` - Manhattan distance between the two vectors
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::manhattan_distance_row;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let distance = manhattan_distance_row(&v1, &v2);
/// // |1-4| + |2-6| = 3 + 4 = 7
/// assert!((distance - 7.0).abs() < 1e-6);
/// ```
#[inline]
pub fn manhattan_distance_row<S1, S2>(x1: &ArrayBase<S1, Ix1>, x2: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let diff = x1 - x2;
    diff.mapv(|x| x.abs()).sum()
}

/// Calculates the Minkowski distance between two vectors
///
/// Computes the p-norm of the difference between two 1D arrays
///
/// # Parameters
///
/// - `x1` - First vector
/// - `x2` - Second vector
/// - `p` - Order of the norm (must be at least 1.0)
///
/// # Returns
///
/// - `f64` - Minkowski distance between the two vectors
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::minkowski_distance_row;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let distance = minkowski_distance_row(&v1, &v2, 3.0);
/// // Expected distance is approximately 4.497
/// assert!((distance - 4.497).abs() < 1e-3);
/// ```
#[inline]
pub fn minkowski_distance_row<S1, S2>(
    x1: &ArrayBase<S1, Ix1>,
    x2: &ArrayBase<S2, Ix1>,
    p: f64,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    // Sum the absolute differences raised to power p, then take the p-th root
    let diff = x1 - x2;
    let sum: f64 = diff.mapv(|x| x.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Calculates the Gini impurity of a label set
///
/// Gini impurity measures how frequently a randomly chosen element would be
/// mislabeled if it were randomly labeled according to the distribution of labels
///
/// # Parameters
///
/// - `y` - Class labels stored in a 1D array
///
/// # Returns
///
/// - `f64` - Gini impurity in the range \[0.0, 1.0\]
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::gini;
///
/// let labels = array![0.0, 0.0, 1.0, 1.0];
/// let gini_val = gini(&labels);
/// // For two classes with equal frequency, Gini = 1 - (0.5^2 + 0.5^2) = 0.5
/// assert!((gini_val - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub fn gini<S>(y: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    let total_samples = y.len() as f64;
    if total_samples == 0.0 {
        return 0.0;
    }

    // Capacity 10 covers most classification problems without reallocating
    let mut class_counts = AHashMap::with_capacity(10);

    y.fold((), |_, &value| {
        if value.is_nan() {
            return; // skip NaN as invalid input
        }
        // Round to 3 decimal places to form an integer class key
        let key = (value * 1000.0).round() as i64;
        *class_counts.entry(key).or_insert(0) += 1;
    });

    // All values were NaN, so treat as an empty dataset
    if class_counts.is_empty() {
        return 0.0;
    }

    // Normalize by the valid (non-NaN) sample count, not `y.len()`
    let valid_samples: f64 = class_counts.values().map(|&c| c as f64).sum();

    let mut sum_squared_proportions = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / valid_samples;
        sum_squared_proportions += p * p;
    }

    1.0 - sum_squared_proportions
}

/// Calculates the entropy of a label set
///
/// Entropy quantifies the impurity or randomness in a dataset and is used
/// by decision tree algorithms to evaluate split quality
///
/// # Parameters
///
/// - `y` - Class labels stored in a 1D array
///
/// # Returns
///
/// - `f64` - Entropy value of the dataset (0.0 for homogeneous data)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::entropy;
///
/// let labels = array![0.0, 1.0, 1.0, 0.0];
/// let ent = entropy(&labels);
/// // For two classes with equal frequency, entropy = 1.0
/// assert!((ent - 1.0).abs() < 1e-6);
/// ```
#[inline]
pub fn entropy<S>(y: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    let total_samples = y.len() as f64;
    if total_samples == 0.0 {
        return 0.0;
    }

    // Capacity 10 covers most classification problems without reallocating
    let mut class_counts = AHashMap::with_capacity(10);

    y.fold((), |_, &value| {
        if value.is_nan() {
            return; // skip NaN as invalid input
        }
        // Round to 3 decimal places to form an integer class key
        let key = (value * 1000.0).round() as i64;
        *class_counts.entry(key).or_insert(0) += 1;
    });

    // All values were NaN, so treat as an empty dataset
    if class_counts.is_empty() {
        return 0.0;
    }

    // Normalize by the valid (non-NaN) sample count, not `y.len()`: dividing by the full length
    // would let probabilities sum to less than 1 and understate impurity when NaN is present
    let valid_samples: f64 = class_counts.values().map(|&c| c as f64).sum();

    let mut entropy = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / valid_samples;
        // Guard against log2(0), though p is always positive here
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Calculates the population standard deviation of a set of values
///
/// This is `sqrt(`[`variance`]`)`, and inherits its NaN/infinite-skipping contract: the
/// statistic is computed over the finite subset of `values`
///
/// # Parameters
///
/// - `values` - Values to measure dispersion
///
/// # Returns
///
/// - `f64` - Population standard deviation over the finite values (0.0 when the array is
///   empty or has no finite values)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::math::standard_deviation;
///
/// let values = array![1.0, 2.0, 3.0];
/// let std_dev = standard_deviation(&values);
/// // Population standard deviation for [1,2,3] is approximately 0.8165
/// assert!((std_dev - 0.8165).abs() < 1e-4);
/// ```
#[inline]
pub fn standard_deviation<S>(values: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    // Delegate to `variance` so both statistics share the finite-subset / NaN-skipping
    // contract and divisor, keeping `standard_deviation == variance.sqrt()` consistent
    variance(values).sqrt()
}

/// Calculates the average path length adjustment factor for isolation trees
///
/// This is the correction factor `c(n)` used in isolation forests to normalize
/// path lengths based on the expected height of a binary search tree
///
/// # Parameters
///
/// - `n` - Number of samples in the isolation tree node (must be greater than 0)
///
/// # Returns
///
/// - `f64` - Adjustment factor for path length normalization:
///   - 0.0 for `n <= 1`
///   - 1.0 for `n == 2`
///   - Computed correction factor for larger `n`
///
/// # Examples
///
/// ```rust
/// use rustyml::math::average_path_length_factor;
///
/// let factor_small = average_path_length_factor(10);
/// let factor_large = average_path_length_factor(1000);
/// assert_eq!(average_path_length_factor(0), 0.0);
/// assert_eq!(average_path_length_factor(1), 0.0);
/// assert_eq!(average_path_length_factor(2), 1.0);
/// assert!(factor_small > 0.0);
/// assert!(factor_large > factor_small);
/// ```
#[inline]
pub fn average_path_length_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }

    let h_n_minus_1 = if n > 50 {
        ((n - 1) as f64).ln() + EULER_GAMMA
    } else {
        (1..n).map(|i| 1.0 / i as f64).sum::<f64>()
    };

    2.0 * h_n_minus_1 - 2.0 * (n - 1) as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // hinge_loss

    /// Mean hinge loss over a mix of correct and margin-violating samples
    #[test]
    fn test_hinge_loss_basic() {
        let margins = array![0.8, -0.5, 2.0];
        let labels = array![1.0, -1.0, 1.0];
        let expected = 0.7_f64 / 3.0;
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), expected, epsilon = 1e-6);
    }

    /// Loss is 0 when every sample has margin >= 1
    #[test]
    fn test_hinge_loss_all_correct() {
        let margins = array![2.0, -3.0];
        let labels = array![1.0, -1.0];
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-10);
    }

    /// Empty input returns 0.0
    #[test]
    fn test_hinge_loss_empty() {
        let margins: ndarray::Array1<f64> = array![];
        let labels: ndarray::Array1<f64> = array![];
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-10);
    }

    /// Single-sample hinge loss
    #[test]
    fn test_hinge_loss_single_sample() {
        let margins = array![0.5];
        let labels = array![1.0];
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.5, epsilon = 1e-10);
    }

    // average_path_length_factor

    #[test]
    fn test_average_path_length_factor_n0() {
        assert_abs_diff_eq!(average_path_length_factor(0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_path_length_factor_n1() {
        assert_abs_diff_eq!(average_path_length_factor(1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_path_length_factor_n2() {
        assert_abs_diff_eq!(average_path_length_factor(2), 1.0, epsilon = 1e-10);
    }

    /// c(3) uses the exact harmonic branch and equals 5/3
    #[test]
    fn test_average_path_length_factor_n3() {
        let expected = 5.0_f64 / 3.0;
        assert_abs_diff_eq!(average_path_length_factor(3), expected, epsilon = 1e-9);
    }

    /// Factor increases with n: c(10) < c(100) < c(1000)
    #[test]
    fn test_average_path_length_factor_monotone() {
        let f10 = average_path_length_factor(10);
        let f100 = average_path_length_factor(100);
        let f1000 = average_path_length_factor(1000);
        assert!(
            f10 < f100,
            "expected factor(10) < factor(100), got {f10} vs {f100}"
        );
        assert!(
            f100 < f1000,
            "expected factor(100) < factor(1000), got {f100} vs {f1000}"
        );
    }

    /// Factor stays strictly increasing across the n=50/51 switch from the exact harmonic sum
    /// to the ln(n-1)+gamma approximation
    #[test]
    fn test_average_path_length_factor_monotone_across_branch_boundary() {
        let f49 = average_path_length_factor(49); // exact-harmonic branch
        let f50 = average_path_length_factor(50); // exact-harmonic branch (last)
        let f51 = average_path_length_factor(51); // ln+gamma branch (first)
        let f52 = average_path_length_factor(52); // ln+gamma branch
        assert!(f49 < f50, "factor(49)={f49} should be < factor(50)={f50}");
        assert!(f50 < f51, "factor(50)={f50} should be < factor(51)={f51}");
        assert!(f51 < f52, "factor(51)={f51} should be < factor(52)={f52}");
    }

    /// For n>50 the ln(n-1)+gamma approximation stays within the documented error bound of the
    /// exact value c(n) = 2*H_{n-1} - 2(n-1)/n
    #[test]
    fn test_average_path_length_factor_matches_exact_harmonic_within_tolerance() {
        for &n in &[51usize, 100, 1000] {
            let exact_h: f64 = (1..n).map(|i| 1.0 / i as f64).sum();
            let theoretical = 2.0 * exact_h - 2.0 * (n - 1) as f64 / n as f64;
            let got = average_path_length_factor(n);
            assert!(
                (got - theoretical).abs() < 0.025,
                "n={n}: factor={got} deviates from theoretical {theoretical} by more than the documented approximation bound",
            );
        }
    }

    // logistic_loss

    /// Single sample with logit 0 and label 1 yields ln(2)
    #[test]
    fn test_logistic_loss_single_logit_zero_label_one() {
        let logits = array![0.0];
        let labels = array![1.0];
        let expected = 2.0_f64.ln();
        assert_abs_diff_eq!(logistic_loss(&logits, &labels), expected, epsilon = 1e-6);
    }

    /// Mean log loss matches the documented example value
    #[test]
    fn test_logistic_loss_docstring_example() {
        let logits = array![0.0, 2.0, -1.0];
        let labels = array![0.0, 1.0, 0.0];
        assert_abs_diff_eq!(logistic_loss(&logits, &labels), 0.377779, epsilon = 1e-5);
    }

    // gini / entropy float-key rounding behavior

    // Rounding (value * 1000.0) to i64 collapses two values to one class iff |v1 - v2| < 0.0005

    /// Values within one rounded key collapse to a single class, giving gini 0.0
    #[test]
    fn test_gini_float_key_collapse_same_class() {
        let labels = array![1.0000_f64, 1.0003_f64];
        assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
    }

    /// Values 0.001 apart map to two distinct classes, giving gini 0.5
    #[test]
    fn test_gini_float_key_two_distinct_classes() {
        let labels = array![1.000_f64, 1.001_f64];
        assert_abs_diff_eq!(gini(&labels), 0.5, epsilon = 1e-10);
    }

    /// NaN-only input returns 0.0
    #[test]
    fn test_gini_all_nan() {
        let labels = array![f64::NAN];
        assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
    }

    /// Values within one rounded key collapse to a single class, giving entropy 0.0
    #[test]
    fn test_entropy_float_key_collapse_same_class() {
        let labels = array![1.0000_f64, 1.0003_f64];
        assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
    }

    /// Two equal-frequency distinct classes give entropy 1.0 bit
    #[test]
    fn test_entropy_float_key_two_distinct_classes() {
        let labels = array![1.000_f64, 1.001_f64];
        assert_abs_diff_eq!(entropy(&labels), 1.0, epsilon = 1e-10);
    }

    /// NaN-only input returns 0.0
    #[test]
    fn test_entropy_all_nan() {
        let labels = array![f64::NAN];
        assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
    }

    /// NaN is excluded from both class counts and the denominator, so `[0.0, NaN, 1.0]` gives
    /// entropy 1.0 bit rather than the value produced by keeping NaN in the denominator
    #[test]
    fn test_entropy_nan_excluded_from_denominator() {
        let labels = array![0.0_f64, f64::NAN, 1.0];
        assert_abs_diff_eq!(entropy(&labels), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gini_nan_excluded_from_denominator() {
        let labels = array![0.0_f64, f64::NAN, 1.0];
        assert_abs_diff_eq!(gini(&labels), 0.5, epsilon = 1e-10);
    }
}
