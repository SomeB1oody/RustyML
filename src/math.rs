use ahash::AHashMap;
use ndarray::{ArrayBase, Data, Ix1};

const EULER_GAMMA: f64 = 0.57721566490153286060651209008240243104215933593992;

/// Calculates the total sum of squares (SST).
///
/// SST measures the total variability in the data as the sum of squared
/// differences between each value and the mean of all values.
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
    // Handle empty array case
    if values.is_empty() {
        return 0.0;
    }
    // Calculate the mean
    let mean = values.mean().unwrap();
    // Fully vectorized computation
    values.mapv(|x| (x - mean).powi(2)).sum()
}

/// Calculates the sum of squared errors (SSE).
///
/// SSE measures the total squared difference between predicted values and actual labels.
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

/// Calculates the mean squared error (variance) of a set of values.
///
/// The variance is the average of the squared differences between each value
/// and the mean.
///
/// Non-finite entries (`NaN`/`±∞`) are **skipped**: both the mean and the variance are
/// computed over the finite subset, dividing by the count of finite values. A few bad
/// entries therefore do not poison the statistic.
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
    // Return 0.0 for empty arrays
    if y.is_empty() {
        return 0.0;
    }

    // Mean over the FINITE values only, in one pass: NaN/infinite entries are skipped rather
    // than poisoning the whole statistic (e.g. `y.mean()` would be NaN for any NaN element).
    let (finite_sum, finite_count) = y.fold((0.0_f64, 0_usize), |(sum, count), &val| {
        if val.is_finite() {
            (sum + val, count + 1)
        } else {
            (sum, count)
        }
    });

    // No finite values to summarize (all entries were NaN/infinite).
    if finite_count == 0 {
        return 0.0;
    }
    let mean = finite_sum / finite_count as f64;

    // Sum of squared differences over the same finite subset.
    let sum_squared_diff = y.fold(0.0, |acc, &val| {
        if val.is_finite() {
            let diff = val - mean;
            acc + diff * diff
        } else {
            acc // Skip NaN / infinite values.
        }
    });

    // Population variance of the finite subset: divide by the finite count, not the full length.
    sum_squared_diff / finite_count as f64
}

/// Computes the logistic sigmoid for a scalar input.
///
/// The sigmoid maps any real number into the open interval (0, 1) with clipping
/// for extreme values to preserve numerical stability.
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
/// ```rust
/// use rustyml::math::sigmoid;
///
/// let value = sigmoid(0.0);
/// // sigmoid(0) = 0.5
/// assert!((value - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    // Use numerically stable computation for extreme values
    const MAX_SIGMOID_INPUT: f64 = 500.0;
    const MIN_SIGMOID_INPUT: f64 = -500.0;

    if z > MAX_SIGMOID_INPUT {
        // For very large positive values, sigmoid(z) approaches 1
        return 1.0;
    } else if z < MIN_SIGMOID_INPUT {
        // For very large negative values, sigmoid(z) approaches 0
        return 0.0;
    }

    // Standard computation for normal range
    1.0 / (1.0 + (-z).exp())
}

/// Calculates the logistic regression loss (log loss).
///
/// This computes the average cross-entropy loss by applying the sigmoid
/// to raw logits before evaluating the log-likelihood.
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
    // Using a vectorized approach to calculate log loss
    let n = logits.len() as f64;

    // Calculate total loss using zip to iterate through both arrays simultaneously
    let total_loss = logits
        .iter()
        .zip(actual_labels.iter())
        .map(|(&x, &y)| {
            // Numerically stable way to calculate log loss:
            // max(0, x) - x*y + log(1 + exp(-|x|))
            x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln()
        })
        .sum::<f64>();

    total_loss / n
}

/// Calculates the mean hinge loss for margin-based classifiers (e.g. linear SVM).
///
/// For each sample the loss is `max(0, 1 - yᵢ · marginᵢ)`, averaged over all samples,
/// where the margin is the decision-function value `w·xᵢ + b`. Labels are expected in
/// `{-1, +1}`.
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

/// Calculates the squared Euclidean distance between two vectors.
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
    // Calculate the difference between the two vectors
    let diff = x1 - x2;

    // Calculate the sum of squares (fully vectorized)
    diff.mapv(|x| x * x).sum()
}

/// Calculates the Manhattan (L1) distance between two vectors.
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
    // Calculate the difference between the two vectors
    let diff = x1 - x2;

    // Calculate the sum of absolute differences (fully vectorized)
    diff.mapv(|x| x.abs()).sum()
}

/// Calculates the Minkowski distance between two vectors.
///
/// Computes the p-norm of the difference between two 1D arrays.
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
    // Calculate the difference between the two vectors
    let diff = x1 - x2;

    // Calculate the sum of absolute differences raised to power p,
    // then take the p-th root of the sum
    let sum: f64 = diff.mapv(|x| x.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Calculates the Gini impurity of a label set.
///
/// Gini impurity measures how frequently a randomly chosen element would be
/// mislabeled if it were randomly labeled according to the distribution of labels.
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

    // Pre-allocate capacity for the HashMap to avoid frequent reallocations
    // A capacity of 10 is reasonable for most classification problems
    let mut class_counts = AHashMap::with_capacity(10);

    // Process all elements in the array with fold operation
    y.fold((), |_, &value| {
        // Handle NaN values - they should be treated as invalid input
        if value.is_nan() {
            return; // Skip NaN values
        }

        // Convert float to integer representation with 3 decimal places precision
        let key = (value * 1000.0).round() as i64;
        *class_counts.entry(key).or_insert(0) += 1;
    });

    // If all values were NaN, treat as empty dataset
    if class_counts.is_empty() {
        return 0.0;
    }

    // Normalize by the number of VALID (non-NaN) samples, not `y.len()` (see `entropy` above).
    let valid_samples: f64 = class_counts.values().map(|&c| c as f64).sum();

    // Calculate Gini impurity more efficiently
    let mut sum_squared_proportions = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / valid_samples;
        sum_squared_proportions += p * p;
    }

    1.0 - sum_squared_proportions
}

/// Calculates the entropy of a label set.
///
/// Entropy quantifies the impurity or randomness in a dataset and is used
/// by decision tree algorithms to evaluate split quality.
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

    // Pre-allocate capacity for the HashMap to avoid frequent reallocations
    // A capacity of 10 is reasonable for most classification problems
    let mut class_counts = AHashMap::with_capacity(10);

    // Use fold operation instead of manual iteration for potential compiler optimizations
    y.fold((), |_, &value| {
        // Handle NaN values - they should be treated as invalid input
        if value.is_nan() {
            return; // Skip NaN values
        }

        // Convert float to integer representation with 3 decimal places precision
        let key = (value * 1000.0).round() as i64;
        *class_counts.entry(key).or_insert(0) += 1;
    });

    // If all values were NaN, treat as empty dataset
    if class_counts.is_empty() {
        return 0.0;
    }

    // Normalize by the number of VALID (non-NaN) samples, not `y.len()`. NaN entries were
    // skipped above, so dividing by the full length would make the class probabilities sum to
    // less than 1 and understate the impurity whenever the input contains NaN.
    let valid_samples: f64 = class_counts.values().map(|&c| c as f64).sum();

    // Calculate entropy more efficiently with direct loop
    let mut entropy = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / valid_samples;
        // Safeguard against log2(0), although this shouldn't happen in this context
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Calculates the population standard deviation of a set of values.
///
/// This is `sqrt(`[`variance`]`)`, and inherits its NaN/infinite-skipping contract: the
/// statistic is computed over the finite subset of `values`.
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
    // The standard deviation is the square root of the population variance. Delegating to
    // `variance` keeps the two statistics consistent — same finite-subset / NaN-skipping
    // contract, same divisor — so `standard_deviation == variance.sqrt()` always holds.
    variance(values).sqrt()
}

/// Calculates the average path length adjustment factor for isolation trees.
///
/// This is the correction factor `c(n)` used in isolation forests to normalize
/// path lengths based on the expected height of a binary search tree.
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

    // ─── hinge_loss ───────────────────────────────────────────────────────────

    /// Docstring example: scores=[0.8,-0.5,2.0], labels=[1.0,-1.0,1.0]
    /// Per-sample: max(0,1-0.8)=0.2, max(0,1-0.5)=0.5, max(0,1-2)=0.0
    /// mean = 0.7/3 = 0.23333...
    #[test]
    fn test_hinge_loss_basic() {
        let margins = array![0.8, -0.5, 2.0];
        let labels = array![1.0, -1.0, 1.0];
        let expected = 0.7_f64 / 3.0;
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), expected, epsilon = 1e-6);
    }

    /// When all margins are correctly classified with margin >= 1 the loss is 0.
    /// scores=[2.0, 3.0], labels=[1.0, -1.0]:
    ///   max(0,1-2.0)=0, max(0,1-(-1.0)*(-3.0))=max(0,1-3)=0 → mean=0.0
    #[test]
    fn test_hinge_loss_all_correct() {
        let margins = array![2.0, -3.0];
        let labels = array![1.0, -1.0];
        // item 0: max(0, 1 - 1.0*2.0) = max(0,-1)=0
        // item 1: max(0, 1 - (-1.0)*(-3.0)) = max(0,1-3)=0
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-10);
    }

    /// Empty input → 0.0 (code early-returns 0.0 when n==0)
    #[test]
    fn test_hinge_loss_empty() {
        let margins: ndarray::Array1<f64> = array![];
        let labels: ndarray::Array1<f64> = array![];
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.0, epsilon = 1e-10);
    }

    /// Single sample: score=0.5, label=1.0
    /// loss = max(0, 1 - 1.0*0.5) = 0.5
    #[test]
    fn test_hinge_loss_single_sample() {
        let margins = array![0.5];
        let labels = array![1.0];
        assert_abs_diff_eq!(hinge_loss(&margins, &labels), 0.5, epsilon = 1e-10);
    }

    // ─── average_path_length_factor ───────────────────────────────────────────

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

    /// n=3: harmonic branch (n<=50), H_2 = 1 + 0.5 = 1.5
    /// result = 2*1.5 - 2*(3-1)/3 = 3.0 - 4/3 = 5/3 ≈ 1.666667
    #[test]
    fn test_average_path_length_factor_n3() {
        // Independent derivation:
        //   h_{n-1} = H_2 = 1/1 + 1/2 = 1.5
        //   c(3) = 2*1.5 - 2*2/3 = 3.0 - 1.3333... = 1.6666...
        let expected = 5.0_f64 / 3.0;
        assert_abs_diff_eq!(average_path_length_factor(3), expected, epsilon = 1e-9);
    }

    /// Monotonically increasing: c(10) < c(100) < c(1000)
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

    /// The factor is strictly increasing in `n`, INCLUDING across the n=50/51 switch
    /// from the exact harmonic sum to the `ln(n-1)+γ` approximation.
    ///
    /// Ground truth: c(n) = 2·H_{n-1} − 2(n-1)/n is strictly increasing (H_{n-1} strictly
    /// increases; 2(n-1)/n increases toward 2 but more slowly). The approximation preserves
    /// this — the boundary values are c(49)≈6.958 < c(50)≈6.998 < c(51)≈7.018 < c(52)≈7.057.
    #[test]
    fn test_average_path_length_factor_monotone_across_branch_boundary() {
        let f49 = average_path_length_factor(49); // exact-harmonic branch
        let f50 = average_path_length_factor(50); // exact-harmonic branch (last)
        let f51 = average_path_length_factor(51); // ln+γ branch (first)
        let f52 = average_path_length_factor(52); // ln+γ branch
        assert!(f49 < f50, "factor(49)={f49} should be < factor(50)={f50}");
        assert!(f50 < f51, "factor(50)={f50} should be < factor(51)={f51}");
        assert!(f51 < f52, "factor(51)={f51} should be < factor(52)={f52}");
    }

    /// For n>50 the implementation uses the standard `ln(n-1)+γ` approximation of the
    /// harmonic number (the same form scikit-learn's isolation forest uses). It must stay
    /// within the known approximation error of the exact theoretical value
    /// c(n) = 2·H_{n-1} − 2(n-1)/n, computed here independently from the exact harmonic sum.
    /// The error is ≈ 1/(n-1) ≤ ~0.02 for n≥51 and shrinks as n grows.
    #[test]
    fn test_average_path_length_factor_matches_exact_harmonic_within_tolerance() {
        for &n in &[51usize, 100, 1000] {
            // Independent reference: the exact (n-1)-th harmonic number.
            let exact_h: f64 = (1..n).map(|i| 1.0 / i as f64).sum();
            let theoretical = 2.0 * exact_h - 2.0 * (n - 1) as f64 / n as f64;
            let got = average_path_length_factor(n);
            assert!(
                (got - theoretical).abs() < 0.025,
                "n={n}: factor={got} deviates from theoretical {theoretical} by more than the documented approximation bound",
            );
        }
    }

    // ─── logistic_loss ────────────────────────────────────────────────────────

    /// Single sample, logit=0.0, label=1.0
    /// Formula (numerically stable): max(0,x) - x*y + ln(1+exp(-|x|))
    ///   = 0 - 0 + ln(1+1) = ln(2) ≈ 0.693147
    /// Matches cross-entropy: -ln(sigmoid(0)) = -ln(0.5) = ln(2)
    #[test]
    fn test_logistic_loss_single_logit_zero_label_one() {
        let logits = array![0.0];
        let labels = array![1.0];
        let expected = 2.0_f64.ln(); // ln(2) ≈ 0.693147
        assert_abs_diff_eq!(logistic_loss(&logits, &labels), expected, epsilon = 1e-6);
    }

    /// Docstring example: logits=[0.0,2.0,-1.0], labels=[0.0,1.0,0.0]
    /// Per-sample (stable formula max(0,x) - x*y + ln(1+exp(-|x|))):
    ///   x=0, y=0: 0 - 0 + ln(2) ≈ 0.693147
    ///   x=2, y=1: 2 - 2 + ln(1+e^{-2}) = ln(1+0.135335) ≈ 0.126928
    ///   x=-1,y=0: 0 - 0 + ln(1+e^{-1}) = ln(1+0.367879) ≈ 0.313261
    ///   mean ≈ 1.133336/3 ≈ 0.377779
    #[test]
    fn test_logistic_loss_docstring_example() {
        let logits = array![0.0, 2.0, -1.0];
        let labels = array![0.0, 1.0, 0.0];
        // Independent: 0.693147 + 0.126928 + 0.313261 = 1.133336; /3 = 0.377779
        assert_abs_diff_eq!(logistic_loss(&logits, &labels), 0.377779, epsilon = 1e-5);
    }

    // ─── gini / entropy — float-key rounding behavior ─────────────────────────

    // The code rounds (value * 1000.0) to an i64, so two values collapse to
    // the same class iff |v1*1000 - v2*1000| < 0.5, i.e. |v1 - v2| < 0.0005.

    /// Two values differing by 0.0003 → same rounded key → ONE class → gini=0.0
    /// 1.0000 * 1000 = 1000.0 → key 1000
    /// 1.0003 * 1000 = 1000.3 → round → key 1000  (same)
    #[test]
    fn test_gini_float_key_collapse_same_class() {
        let labels = array![1.0000_f64, 1.0003_f64];
        assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
    }

    /// Two values differing by 0.001 → different rounded keys → TWO classes → gini=0.5
    /// 1.000 * 1000 = 1000.0 → key 1000
    /// 1.001 * 1000 = 1001.0 → key 1001  (different)
    /// p = 0.5 each; gini = 1 - (0.25 + 0.25) = 0.5
    #[test]
    fn test_gini_float_key_two_distinct_classes() {
        let labels = array![1.000_f64, 1.001_f64];
        assert_abs_diff_eq!(gini(&labels), 0.5, epsilon = 1e-10);
    }

    /// NaN labels are skipped. An array of only NaNs → 0.0.
    /// (NaN values do not count toward total_samples denominator check either,
    ///  but total_samples = y.len() = 1 and class_counts is empty → return 0.0)
    #[test]
    fn test_gini_all_nan() {
        let labels = array![f64::NAN];
        assert_abs_diff_eq!(gini(&labels), 0.0, epsilon = 1e-10);
    }

    /// Two values differing by 0.0003 → same key → ONE class → entropy=0.0
    #[test]
    fn test_entropy_float_key_collapse_same_class() {
        let labels = array![1.0000_f64, 1.0003_f64];
        assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
    }

    /// Two equal-frequency distinct classes → entropy = 1.0 bit
    /// 1.000 and 1.001 → keys 1000 and 1001, p=0.5 each
    /// entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
    #[test]
    fn test_entropy_float_key_two_distinct_classes() {
        let labels = array![1.000_f64, 1.001_f64];
        assert_abs_diff_eq!(entropy(&labels), 1.0, epsilon = 1e-10);
    }

    /// NaN-only input → 0.0
    #[test]
    fn test_entropy_all_nan() {
        let labels = array![f64::NAN];
        assert_abs_diff_eq!(entropy(&labels), 0.0, epsilon = 1e-10);
    }

    /// Regression guard for the NaN-denominator fix: with a MIX of NaN and valid labels, NaN is
    /// skipped from the class counts AND from the denominator. `[0.0, NaN, 1.0]` is two balanced
    /// valid classes → entropy = 1.0 bit, gini = 0.5 — NOT the ~1.057 / ~0.556 produced if NaN is
    /// wrongly kept in the denominator (`y.len()` = 3 instead of the 2 valid samples).
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
