use crate::ModelError;
use ahash::AHashMap;
use ndarray::ArrayView1;

/// Calculates the Sum of Square Total
///
/// SST measures the total variability in the data, computed as the sum of squared
/// differences between each actual value and the mean of all values.
///
/// # Parameters
///
/// * `values` - An ndarray `ArrayView1<f64>` of observed values
///
/// # Returns
///
/// - `Ok(f64)` - The Sum of Square Total (SST)
/// - `Err(ModelError::InputValidationError)` - If input does not match expectation
///
/// # Examples
/// ```rust
/// use rustyml::math::sum_of_square_total;
/// use ndarray::array;
///
/// let values = array![1.0, 2.0, 3.0];
/// let sst = sum_of_square_total(values.view()).unwrap();
/// // Mean is 2.0, so SST = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2.0
/// assert!((sst - 2.0).abs() < 1e-5);
/// ```
pub fn sum_of_square_total(values: ArrayView1<f64>) -> Result<f64, ModelError> {
    if values.is_empty() {
        return Err(ModelError::InputValidationError(
            "Cannot calculate sum of square total with empty inputs".to_string(),
        ));
    }

    // Calculate the mean
    let mean = values.mean().unwrap();

    // Fully vectorized computation
    Ok(values.mapv(|x| (x - mean).powi(2)).sum())
}

/// Calculate the sum of squared errors
///
/// # Parameters
///
/// * `predicted` - Predicted values vector (y') as ndarray `ArrayView1<f64>`
/// * `actual` - Actual values vector (y) as ndarray `ArrayView1<f64>`
///
/// # Returns
///
/// - `Ok(f64)` - Sum of squared errors sum((predicted_i - actual_i)^2)
/// - `Err(ModelError::InputValidationError)` if input does not match expectation
///
/// # Examples
/// ```rust
/// use rustyml::math::sum_of_squared_errors;
/// use ndarray::array;
///
/// let predicted = array![2.0, 3.0];
/// let actual = array![1.0, 3.0];
/// let sse = sum_of_squared_errors(predicted.view(), actual.view()).unwrap();
/// // (2-1)^2 + (3-3)^2 = 1 + 0 = 1
/// assert!((sse - 1.0).abs() < 1e-6);
/// ```
pub fn sum_of_squared_errors(
    predicted: ArrayView1<f64>,
    actual: ArrayView1<f64>,
) -> Result<f64, ModelError> {
    // Ensure both arrays have the same length
    if predicted.len() != actual.len() {
        return Err(ModelError::InputValidationError(format!(
            "Predicted and actual arrays must have the same length, predicted length: {}, actual length: {}",
            predicted.len(),
            actual.len()
        )));
    }

    if predicted.is_empty() {
        return Err(ModelError::InputValidationError(
            "Cannot calculate sum of squared errors with empty inputs".to_string(),
        ));
    }

    let sum = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum();

    Ok(sum)
}

/// Calculate the sigmoid function value for a given input
///
/// The sigmoid function transforms any real-valued number into the range (0, 1).
/// It is defined as: σ(z) = 1 / (1 + e^(-z))
///
/// # Parameters
///
/// * `z` - The input value (can be any real number)
///
/// # Returns
///
/// * A value between 0 and 1, representing the sigmoid of the input
///
/// # Mathematical properties
///
/// * sigmoid(0) = 0.5
/// * As z approaches positive infinity, sigmoid(z) approaches 1
/// * As z approaches negative infinity, sigmoid(z) approaches 0
///
/// # Examples
/// ```rust
/// use rustyml::math::sigmoid;
///
/// let value = sigmoid(0.0);
/// // sigmoid(0) = 0.5
/// assert!((value - 0.5).abs() < 1e-6);
/// ```
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Calculate logistic regression loss (log loss)
///
/// This function computes the average cross-entropy loss for logistic regression,
/// applying sigmoid to raw predictions before calculating loss.
///
/// # Parameters
///
/// * `logits` - `ArrayView1<f64>` of raw model outputs (logits, before sigmoid)
/// * `actual_labels` - `ArrayView1<f64>` of actual binary labels (0 or 1)
///
/// # Returns
///
/// - `Ok(f64)` - Average logistic regression loss
/// - `Err(ModelError::InputValidationError)` - If input does not match expectation
///
/// # Examples
/// ```rust
/// use rustyml::math::logistic_loss;
/// use ndarray::array;
///
/// let logits = array![0.0, 2.0, -1.0];
/// let actual_labels = array![0.0, 1.0, 0.0];
/// let loss = logistic_loss(logits.view(), actual_labels.view()).unwrap();
/// // Expected average loss is approximately 0.37778
/// assert!((loss - 0.37778).abs() < 1e-5);
/// ```
pub fn logistic_loss(
    logits: ArrayView1<f64>,
    actual_labels: ArrayView1<f64>,
) -> Result<f64, ModelError> {
    if logits.len() != actual_labels.len() {
        return Err(ModelError::InputValidationError(format!(
            "Predicted and actual vectors must have the same length, predicted length: {}, actual length: {}",
            logits.len(),
            actual_labels.len()
        )));
    }

    // Validate that all labels are either 0 or 1
    if actual_labels
        .iter()
        .any(|&label| label != 0.0 && label != 1.0)
    {
        return Err(ModelError::InputValidationError(
            "All labels must be either 0 or 1".to_string(),
        ));
    }

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

    Ok(total_loss / n)
}

/// Calculate the squared Euclidean distance between two vectors
///
/// # Parameters
///
/// - `x1` - First vector as ndarray `ArrayView1<f64>`
/// - `x2` - Second vector as ndarray `ArrayView1<f64>`
///
/// # Returns
/// * `f64` - The squared Euclidean distance between x1 and x2
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::squared_euclidean_distance_row;
///
/// let v1 = array![1.0, 2.0, 3.0];
/// let v2 = array![4.0, 5.0, 6.0];
/// let dist = squared_euclidean_distance_row(v1.view(), v2.view());
/// // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
/// assert!((dist - 27.0).abs() < 1e-10);
/// ```
pub fn squared_euclidean_distance_row(x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
    // Calculate the difference between the two vectors
    let diff = &x1 - &x2;

    // Calculate the sum of squares (fully vectorized)
    diff.mapv(|x| x * x).sum()
}

/// Calculate the Manhattan distance between two vectors
///
/// Computes the sum of absolute differences between corresponding elements
/// of two 1D arrays, representing the Manhattan (L1) distance.
///
/// # Parameters
/// - `x1` - First vector as ndarray `ArrayView1<f64>`
/// - `x2` - Second vector as ndarray `ArrayView1<f64>`
///
/// # Returns
/// * `f64` - The Manhattan distance between x1 and x2
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::manhattan_distance_row;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let distance = manhattan_distance_row(v1.view(), v2.view());
/// // |1-4| + |2-6| = 3 + 4 = 7
/// assert!((distance - 7.0).abs() < 1e-6);
/// ```
pub fn manhattan_distance_row(x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
    // Calculate the difference between the two vectors
    let diff = &x1 - &x2;

    // Calculate the sum of absolute differences (fully vectorized)
    diff.mapv(|x| x.abs()).sum()
}

/// Calculate the Minkowski distance between two vectors
///
/// Computes the p-norm between corresponding elements of two 1D arrays,
/// representing the Minkowski distance with parameter p.
///
/// # Parameters
/// - `x1` - First vector as ndarray `ArrayView1<f64>`
/// - `x2` - Second vector as ndarray `ArrayView1<f64>`
/// - `p` - The order of the norm (p ≥ 1.0)
///
/// # Returns
/// * `f64` - The Minkowski distance between x1 and x2
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::minkowski_distance_row;
///
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let distance = minkowski_distance_row(v1.view(), v2.view(), 3.0);
/// // Expected distance is approximately 4.497
/// assert!((distance - 4.497).abs() < 1e-3);
/// ```
pub fn minkowski_distance_row(x1: ArrayView1<f64>, x2: ArrayView1<f64>, p: f64) -> f64 {
    assert!(p >= 1.0, "p must be greater than or equal to 1.0");

    // Calculate the difference between the two vectors
    let diff = &x1 - &x2;

    // Calculate the sum of absolute differences raised to power p,
    // then take the p-th root of the sum
    let sum: f64 = diff.mapv(|x| x.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Calculates the entropy of a label set.
///
/// Entropy is a measure of impurity in a dataset, commonly used in decision tree algorithms
/// like ID3 and C4.5. It quantifies the uncertainty or randomness in the data distribution.
///
/// # Parameters
///
/// * `y` - An `ArrayView1<f64>` of values representing class labels
///
/// # Returns
///
/// * The entropy value of the given dataset. Lower values indicate more homogeneous data.
///
/// # Notes
/// - The function handles floating point labels by rounding to 3 decimal places for counting.
/// - Returns 0.0 for empty datasets.
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::entropy;
///
/// let labels = array![0.0, 1.0, 1.0, 0.0];
/// let ent = entropy(labels.view());
/// // For two classes with equal frequency, entropy = 1.0
/// assert!((ent - 1.0).abs() < 1e-6);
/// ```
pub fn entropy(y: ArrayView1<f64>) -> f64 {
    let total_samples = y.len() as f64;
    if total_samples == 0.0 {
        return 0.0;
    }

    // Pre-allocate capacity for the HashMap to avoid frequent reallocations
    // A capacity of 10 is reasonable for most classification problems
    let mut class_counts = AHashMap::with_capacity(10);

    // Use fold operation instead of manual iteration for potential compiler optimizations
    y.fold((), |_, &value| {
        // Convert float to integer representation with 3 decimal places precision
        let key = (value * 1000.0).round() as i64;
        *class_counts.entry(key).or_insert(0) += 1;
    });

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

/// Calculates the Gini impurity of a label set.
///
/// Gini impurity is a measure of how often a randomly chosen element from the set would be
/// incorrectly labeled if it was randomly labeled according to the distribution of labels
/// in the subset. It is commonly used in decision tree algorithms like CART.
///
/// # Parameters
///
/// * `y` - An `ArrayView1<f64>` of values representing class labels
///
/// # Returns
///
/// * `f64` - The Gini impurity value of the given dataset. The value ranges from 0.0 (pure) to 1.0 (impure).
///
/// # Notes
///
/// - The function handles floating point labels by rounding to 3 decimal places for counting.
/// - Returns 0.0 for empty datasets.
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::gini;
///
/// let labels = array![0.0, 0.0, 1.0, 1.0];
/// let gini_val = gini(labels.view());
/// // For two classes with equal frequency, Gini = 1 - (0.5^2 + 0.5^2) = 0.5
/// assert!((gini_val - 0.5).abs() < 1e-6);
/// ```
pub fn gini(y: ArrayView1<f64>) -> f64 {
    let total_samples = y.len() as f64;
    if total_samples == 0.0 {
        return 0.0;
    }

    // Pre-allocate capacity for the HashMap to avoid frequent reallocations
    // A capacity of 10 is reasonable for most classification problems
    let mut class_counts = AHashMap::with_capacity(10);

    // Process all elements in the array with fold operation
    y.fold((), |_, &value| {
        // Convert float to integer representation with 3 decimal places precision
        let key = (value * 1000.0).round() as i64;
        *class_counts.entry(key).or_insert(0) += 1;
    });

    // Calculate Gini impurity more efficiently
    let mut sum_squared_proportions = 0.0;
    for &count in class_counts.values() {
        let p = count as f64 / total_samples;
        sum_squared_proportions += p * p;
    }

    1.0 - sum_squared_proportions
}

/// Calculates the information gain when splitting a dataset.
///
/// Information gain measures the reduction in entropy (or impurity) achieved by
/// splitting the dataset into two parts. It is commonly used in decision tree algorithms
/// like ID3 and C4.5 to select the best feature for splitting.
///
/// # Parameters
///
/// - `y` - An `ArrayView1<f64>` representing class labels in the parent node
/// - `left_y` - An `ArrayView1<f64>` representing class labels in the left child node
/// - `right_y` - An `ArrayView1<f64>` representing class labels in the right child node
///
/// # Returns
///
/// * The information gain value. Higher values indicate a more useful split.
///
/// # Notes
/// This function uses the entropy function to calculate impurity at each node.
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::information_gain;
///
/// let parent = array![0.0, 0.0, 1.0, 1.0];
/// let left = array![0.0, 0.0];
/// let right = array![1.0, 1.0];
/// let ig = information_gain(parent.view(), left.view(), right.view());
/// // Entropy(parent)=1.0, Entropy(left)=Entropy(right)=0, so IG = 1.0
/// assert!((ig - 1.0).abs() < 1e-6);
/// ```
pub fn information_gain(
    y: ArrayView1<f64>,
    left_y: ArrayView1<f64>,
    right_y: ArrayView1<f64>,
) -> f64 {
    // Calculate sample counts once
    let n = y.len() as f64;

    // Early return for edge cases
    if n == 0.0 {
        return 0.0;
    }

    let n_left = left_y.len() as f64;
    let n_right = right_y.len() as f64;

    // Calculate entropy values
    let e = entropy(y);

    // If parent node is already pure, no information gain is possible
    if e.abs() < f64::EPSILON {
        return 0.0;
    }

    let e_left = entropy(left_y);
    let e_right = entropy(right_y);

    // Calculate the weighted average entropy of children
    let weighted_child_entropy = (n_left / n) * e_left + (n_right / n) * e_right;

    // Information gain = parent entropy - weighted sum of child entropies
    e - weighted_child_entropy
}

/// Calculates the gain ratio for a dataset split.
///
/// Gain ratio is an extension of information gain that reduces the bias towards
/// features with a large number of values. It is used in the C4.5 algorithm to
/// normalize information gain by the entropy of the split itself.
///
/// # Parameters
///
/// - `y` - An `ArrayView1<f64>` representing class labels in the parent node
/// - `left_y` - An `ArrayView1<f64>` representing class labels in the left child node
/// - `right_y` - An `ArrayView1<f64>` representing class labels in the right child node
///
/// # Returns
///
/// * The gain ratio value. Higher values indicate a more useful split.
///
/// # Notes
///
/// - Returns 0.0 if the split information is zero to avoid division by zero.
/// - This function uses the information_gain function as part of its calculation.
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::gain_ratio;
///
/// let parent = array![0.0, 0.0, 1.0, 1.0];
/// let left = array![0.0, 0.0];
/// let right = array![1.0, 1.0];
/// let gr = gain_ratio(parent.view(), left.view(), right.view());
/// // With equal splits, gain ratio should be 1.0
/// assert!((gr - 1.0).abs() < 1e-6);
/// ```
pub fn gain_ratio(y: ArrayView1<f64>, left_y: ArrayView1<f64>, right_y: ArrayView1<f64>) -> f64 {
    // Early return if either split is empty
    if left_y.is_empty() || right_y.is_empty() {
        return 0.0;
    }

    // Calculate sample counts once to avoid redundant computations
    let n = y.len() as f64;
    let n_left = left_y.len() as f64;
    let n_right = right_y.len() as f64;

    // Verify that left and right subsets contain all samples from parent
    debug_assert!(
        (n_left + n_right - n).abs() < f64::EPSILON,
        "The sum of left and right samples should equal parent samples"
    );

    // Calculate information gain
    let info_gain = information_gain(y, left_y, right_y);

    // If information gain is negligible, return 0 to avoid unnecessary calculations
    if info_gain < f64::EPSILON {
        return 0.0;
    }

    // Calculate the proportions for split information
    let p_left = n_left / n;
    let p_right = n_right / n;

    // Calculate split information, which measures the potential information of the split
    // Handle edge cases where one of the proportions is 0
    let mut split_info = 0.0;
    if p_left > 0.0 {
        split_info -= p_left * p_left.log2();
    }
    if p_right > 0.0 {
        split_info -= p_right * p_right.log2();
    }

    // Avoid division by zero
    if split_info < f64::EPSILON {
        0.0
    } else {
        info_gain / split_info
    }
}

/// Calculates the Mean Squared Error (MSE) or variance of a set of values.
///
/// The MSE is calculated as the average of the squared differences between each value
/// and the mean of all values. It represents the variance of the dataset.
///
/// # Arguments
///
/// * `y` - An `ArrayView1<f64>` of values for which to calculate the MSE
///
/// # Returns
///
/// * `f64` - The mean squared error as a f64 value, returns 0.0 if the input array is empty
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::variance;
///
/// let values = array![1.0, 2.0, 3.0];
/// let mse = variance(values.view());
/// // Mean is 2.0, so variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 ≈ 0.66667
/// assert!((mse - 0.6666667).abs() < 1e-6);
/// ```
pub fn variance(y: ArrayView1<f64>) -> f64 {
    let n = y.len();

    // Return 0.0 for empty arrays
    if n == 0 {
        return 0.0;
    }

    // Calculate mean using ndarray's mean method
    // This is more efficient than manual calculation
    let mean = y.mean().unwrap_or(0.0);

    // Use fold for potentially better performance than map/sum
    // This computes the sum of squared differences in one pass
    let sum_squared_diff = y.fold(0.0, |acc, &val| {
        let diff = val - mean;
        acc + diff * diff
    });

    // Compute variance (MSE)
    sum_squared_diff / n as f64
}

/// Calculates the leaf node adjustment factor c(n)
///
/// # Parameters
///
/// * `n` - Number of samples
///
/// # Returns
///
/// * `f64` - The adjustment factor
///
/// # Notes
///
/// Formula: c(n) = 2 * (H(n-1)) - (2*(n-1)/n)
/// where H(n-1) can be approximated by ln(n-1) + gamma, gamma is Euler's constant
///
/// # Examples
/// ```rust
/// use rustyml::math::average_path_length_factor;
///
/// let factor = average_path_length_factor(10.0);
/// // For n = 10, the factor is approximately 3.748
/// assert!((factor - 3.748).abs() < 0.01);
/// ```
pub fn average_path_length_factor(n: f64) -> f64 {
    // Exit early for n <= 1
    if n <= 1.0 {
        return 0.0;
    }

    // Use constant for Euler's gamma to improve readability
    const EULER_GAMMA: f64 = 0.57721566490153286060651209008240243104215933593992;

    // Calculate n-1 once as it's used multiple times
    let n_minus_1 = n - 1.0;

    // Calculate first term: 2 * (ln(n-1) + gamma)
    let harmonic_term = 2.0 * (n_minus_1.ln() + EULER_GAMMA);

    // Calculate second term: 2 * (n-1)/n
    let correction_term = 2.0 * n_minus_1 / n;

    // Combine terms
    harmonic_term - correction_term
}

/// Calculates the standard deviation of a set of values
///
/// # Parameters
///
/// * `values` - An `ArrayView1<f64>` of values
///
/// # Returns
///
/// * `f64` - The population standard deviation. Returns 0.0 if the array is empty or contains only one element.
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::math::standard_deviation;
///
/// let values = array![1.0, 2.0, 3.0];
/// let std_dev = standard_deviation(values.view());
/// // Population standard deviation for [1,2,3] is approximately 0.8165
/// assert!((std_dev - 0.8165).abs() < 1e-4);
/// ```
pub fn standard_deviation(values: ArrayView1<f64>) -> f64 {
    let n = values.len();

    // Return 0.0 for empty arrays or single-element arrays (no variation)
    if n <= 1 {
        return 0.0;
    }

    // Use built-in methods when available for better performance
    // We can calculate variance and then take the square root

    // First calculate the mean efficiently
    let mean = values.mean().unwrap();

    // Calculate variance in one pass
    let variance = values.fold(0.0, |acc, &x| {
        let diff = x - mean;
        acc + diff * diff
    }) / n as f64;

    // Take the square root for standard deviation
    variance.sqrt()
}
