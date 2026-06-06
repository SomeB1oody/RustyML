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
/// and the mean of all values.
///
/// # Parameters
///
/// - `y` - Values for which to calculate the variance
///
/// # Returns
///
/// - `f64` - Variance of the input values (0.0 when the array is empty)
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
    let n = y.len();

    // Return 0.0 for empty arrays
    if n == 0 {
        return 0.0;
    }

    // Calculate mean using ndarray's mean method
    // Handle potential NaN case when all values are NaN
    let mean = match y.mean() {
        Some(m) if m.is_finite() => m,
        _ => return 0.0, // Return 0.0 if mean is NaN or can't be calculated
    };

    // Use fold for potentially better performance than map/sum
    // This computes the sum of squared differences in one pass
    let sum_squared_diff = y.fold(0.0, |acc, &val| {
        if val.is_finite() {
            let diff = val - mean;
            acc + diff * diff
        } else {
            acc // Skip NaN or infinite values
        }
    });

    // Compute variance (MSE)
    sum_squared_diff / n as f64
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

    // Calculate Gini impurity more efficiently
    let mut sum_squared_proportions = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / total_samples;
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

    // Calculate entropy more efficiently with direct loop
    let mut entropy = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / total_samples;
        // Safeguard against log2(0), although this shouldn't happen in this context
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Calculates the population standard deviation of a set of values.
///
/// # Parameters
///
/// - `values` - Values to measure dispersion
///
/// # Returns
///
/// - `f64` - Population standard deviation (0.0 when the array is empty)
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
    let n = values.len();

    // Return 0.0 for empty arrays
    if n == 0 {
        return 0.0;
    }

    // Use built-in methods when available for better performance
    // calculate variance and then take the square root

    // First calculate the mean efficiently
    let mean = values.mean().unwrap(); // Safe since we've validated input

    // Calculate variance in one pass
    let variance = values.fold(0.0, |acc, &x| {
        let diff = x - mean;
        acc + diff * diff
    }) / n as f64;

    // Take the square root for standard deviation
    variance.sqrt()
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
