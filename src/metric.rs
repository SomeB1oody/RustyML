use ahash::AHashMap;
use ndarray::Data;
use ndarray::prelude::*;

/// Calculates the Mean Squared Error between predicted and actual values.
///
/// Mean Squared Error measures the average of the squared differences between predicted values and ground truth values.
///
/// # Parameters
///
/// - `y_true` - Ground-truth values for each sample
/// - `y_pred` - Predicted values for each sample
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::mean_squared_error;
///
/// let actual = array![3.0, -0.5, 2.0, 7.0];
/// let predicted = array![2.5, 0.0, 2.1, 7.8];
/// let mse = mean_squared_error(&actual, &predicted);
/// // MSE = ((3.0 - 2.5)^2 + (-0.5 - 0.0)^2 + (2.0 - 2.1)^2 + (7.0 - 7.8)^2) / 4
/// //     = (0.25 + 0.25 + 0.01 + 0.64) / 4 = 0.2875
/// println!("{}", mse);
/// assert!((mse - 0.2875).abs() < 1e-10);
/// ```
///
/// # Returns
///
/// - `f64` - Mean squared error (returns 0.0 when the input arrays are empty)
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
pub fn mean_squared_error<S>(y_true: &ArrayBase<S, Ix1>, y_pred: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    if y_true.len() != y_pred.len() {
        panic!(
            "Input arrays must have the same length. Predicted: {}, Actual: {}",
            y_true.len(),
            y_pred.len()
        );
    }

    let n = y_true.len();

    // Handle edge case
    if n == 0 {
        return 0.0;
    }

    // Calculate the sum of squared differences efficiently using zip
    // This performs a single pass through both arrays
    let sum_squared_diff = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(0.0, |acc, (&actual, &pred)| {
            let error = actual - pred;
            acc + error * error
        });

    // Return the mean
    sum_squared_diff / (n as f64)
}

/// Calculates the Root Mean Squared Error (RMSE) between predicted and actual values.
///
/// RMSE is the square root of the Mean Squared Error, giving a metric in the same units as the original data.
///
/// # Parameters
///
/// - `predictions` - Predicted values for each sample
/// - `targets` - Actual target values for each sample
///
/// # Examples
/// ```rust
/// use rustyml::metric::root_mean_squared_error;
/// use ndarray::array;
///
/// let predictions = array![2.0, 3.0, 4.0];
/// let targets = array![1.0, 2.0, 3.0];
/// let rmse = root_mean_squared_error(&predictions, &targets);
/// // RMSE = sqrt(((2 - 1)^2 + (3 - 2)^2 + (4 - 3)^2) / 3) = sqrt(3/3) = 1.0
/// assert!((rmse - 1.0).abs() < 1e-6);
/// ```
///
/// # Returns
///
/// - `f64` - Root mean squared error (returns 0.0 when the input arrays are empty)
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
pub fn root_mean_squared_error<S>(
    predictions: &ArrayBase<S, Ix1>,
    targets: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = f64>,
{
    // Check if inputs are empty
    if predictions.is_empty() || targets.is_empty() {
        return 0.0;
    }

    // Check if arrays have matching lengths
    if predictions.len() != targets.len() {
        panic!(
            "Prediction and target arrays must have the same length. Predicted: {}, Actual: {}",
            predictions.len(),
            targets.len()
        );
    }

    // Use zip_fold_while for efficient calculation with early error detection
    let sum_squared_errors =
        predictions
            .iter()
            .zip(targets.iter())
            .fold(0.0, |acc, (&pred, &target)| {
                let error = pred - target;
                acc + error * error
            });

    // Calculate mean squared error
    let mse = sum_squared_errors / predictions.len() as f64;

    // Take square root for RMSE
    // Handle potential numerical issues that might cause slightly negative values
    if mse < 0.0 && mse > -f64::EPSILON {
        0.0
    } else {
        mse.sqrt()
    }
}

/// Calculates the Mean Absolute Error (MAE) between predicted and actual values.
///
/// MAE measures the average absolute difference between predicted values and target values without considering error direction.
///
/// # Parameters
///
/// - `predictions` - Predicted values for each sample
/// - `targets` - Actual target values for each sample
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::mean_absolute_error;
///
/// let predictions = array![2.0, 3.0, 4.0];
/// let targets = array![1.0, 2.0, 3.0];
/// let mae = mean_absolute_error(&predictions, &targets);
/// // MAE = (|2 - 1| + |3 - 2| + |4 - 3|) / 3 = (1 + 1 + 1) / 3 = 1.0
/// assert!((mae - 1.0).abs() < 1e-6);
/// ```
///
/// # Returns
///
/// - `f64` - Mean absolute error (returns 0.0 when the input arrays are empty)
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
pub fn mean_absolute_error<S>(predictions: &ArrayBase<S, Ix1>, targets: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    // Check if inputs are empty
    if predictions.is_empty() || targets.is_empty() {
        return 0.0;
    }

    // Check if arrays have matching lengths
    if predictions.len() != targets.len() {
        panic!(
            "Prediction and target arrays must have the same length. Predicted: {}, Actual: {}",
            predictions.len(),
            targets.len()
        );
    }

    // Calculate sum of absolute errors in a single pass
    // Using fold instead of map+sum for potentially better performance
    let sum_absolute_errors = predictions
        .iter()
        .zip(targets.iter())
        .fold(0.0, |acc, (&pred, &target)| acc + (pred - target).abs());

    // Calculate mean absolute error
    let mae = sum_absolute_errors / predictions.len() as f64;

    mae
}

/// Calculate the R-squared (coefficient of determination) score.
///
/// R^2 measures how well predictions explain the variance in the target values using the formula R^2 = 1 - (SSE / SST).
///
/// # Parameters
///
/// - `predicted` - Predicted values for each sample
/// - `actual` - Actual target values for each sample
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::r2_score;
///
/// let predicted = array![2.0, 3.0, 4.0];
/// let actual = array![1.0, 3.0, 5.0];
/// let r2 = r2_score(&predicted, &actual);
/// // For actual values [1, 3, 5], mean = 3, SSE = 1 + 0 + 1 = 2, SST = 4 + 0 + 4 = 8, so R^2 = 1 - (2/8) = 0.75
/// assert!((r2 - 0.75).abs() < 1e-6);
/// ```
///
/// # Returns
///
/// - `f64` - R-squared value (returns 0.0 when the input arrays are empty)
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
pub fn r2_score<S>(predicted: &ArrayBase<S, Ix1>, actual: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    // Validate inputs first
    if predicted.is_empty() || actual.is_empty() {
        return 0.0;
    }

    if predicted.len() != actual.len() {
        panic!(
            "Prediction and target arrays must have the same length. Predicted: {}, Actual: {}",
            predicted.len(),
            actual.len()
        );
    }

    // Calculate mean of actual values
    let actual_mean = actual.mean().unwrap();

    // Calculate SSE (Sum of Squared Errors) and SST (Sum of Squares Total) in one pass
    let (sse, sst) = actual.iter().zip(predicted.iter()).fold(
        (0.0, 0.0),
        |(sse_acc, sst_acc), (&act, &pred)| {
            let error = pred - act;
            let deviation = act - actual_mean;
            (
                sse_acc + error * error,         // Sum of squared errors
                sst_acc + deviation * deviation, // Sum of squared deviations from mean
            )
        },
    );

    // Prevent division by zero (when all actual values are identical)
    if sst < 1e-10 {
        // Using small epsilon for numerical stability
        return 0.0;
    }

    1.0 - (sse / sst)
}

/// Confusion Matrix for binary classification evaluation.
///
/// Stores counts of true positives, false positives, true negatives, and false negatives for binary classifiers.
///
/// # Fields
///
/// - `tp` - True positive count
/// - `fp` - False positive count
/// - `tn` - True negative count
/// - `fn_` - False negative count
///
/// # Examples
/// ```rust
/// use ndarray::arr1;
/// use rustyml::metric::ConfusionMatrix;
///
/// let predicted = arr1(&[0.9, 0.2, 0.8, 0.1, 0.7]);
/// let actual = arr1(&[1.0, 0.0, 1.0, 0.0, 1.0]);
/// let cm = ConfusionMatrix::new(&predicted, &actual);
///
/// println!("Accuracy: {:.2}", cm.accuracy());
/// println!("Precision: {:.2}", cm.precision());
/// println!("Recall: {:.2}", cm.recall());
/// println!("F1 Score: {:.2}", cm.f1_score());
///
/// let (tp, fp, tn, fn_) = cm.get_counts();
/// println!("TP: {}, FP: {}, TN: {}, FN: {}", tp, fp, tn, fn_);
/// println!("{}", cm.summary());
/// ```
pub struct ConfusionMatrix {
    tp: usize,
    fp: usize,
    tn: usize,
    fn_: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix.
    ///
    /// Converts predicted probabilities and actual labels into binary outcomes using a 0.5 threshold and tallies the resulting counts.
    ///
    /// # Parameters
    ///
    /// - `predicted` - Predicted labels or probabilities (>= 0.5 treated as positive)
    /// - `actual` - Ground-truth labels or probabilities (>= 0.5 treated as positive)
    ///
    /// # Returns
    ///
    /// - `Self` - Confusion matrix with populated counts
    ///
    /// # Panics
    ///
    /// - Panics if the two arrays have different lengths
    /// - Panics if input arrays are empty
    pub fn new<S>(predicted: &ArrayBase<S, Ix1>, actual: &ArrayBase<S, Ix1>) -> Self
    where
        S: Data<Elem = f64>,
    {
        if predicted.len() != actual.len() {
            panic!(
                "Input arrays must have the same length. Predicted: {}, Actual: {}",
                predicted.len(),
                actual.len()
            );
        }

        if predicted.is_empty() {
            panic!("Input arrays must not be empty");
        }

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (p, a) in predicted.iter().zip(actual.iter()) {
            let p_binary = if *p >= 0.5 { 1.0 } else { 0.0 };
            let a_binary = if *a >= 0.5 { 1.0 } else { 0.0 };

            match (p_binary, a_binary) {
                (1.0, 1.0) => tp += 1,
                (1.0, 0.0) => fp += 1,
                (0.0, 1.0) => fn_ += 1,
                (0.0, 0.0) => tn += 1,
                _ => unreachable!(), // Should not happen as we explicitly convert to binary values
            }
        }

        Self { tp, fp, tn, fn_ }
    }

    /// Get the components of the confusion matrix.
    ///
    /// # Returns
    ///
    /// - `(usize, usize, usize, usize)` - Tuple of (tp, fp, tn, fn) counts
    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.tp, self.fp, self.tn, self.fn_)
    }

    /// Calculate accuracy: (TP + TN) / (TP + TN + FP + FN).
    ///
    /// Accuracy measures the proportion of correct predictions among all predictions.
    ///
    /// # Returns
    ///
    /// - `f64` - Accuracy in the range \[0.0, 1.0\] (returns 0.0 when there are no predictions)
    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.tn + self.fp + self.fn_;
        if total == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / total as f64
    }

    /// Calculate error rate: (FP + FN) / (TP + TN + FP + FN).
    ///
    /// Error rate is the complement of accuracy.
    ///
    /// # Returns
    ///
    /// - `f64` - Error rate in the range \[0.0, 1.0\]
    pub fn error_rate(&self) -> f64 {
        1.0 - self.accuracy()
    }

    /// Calculate precision: TP / (TP + FP).
    ///
    /// Precision measures how many predicted positives are correct.
    ///
    /// # Returns
    ///
    /// - `f64` - Precision in the range \[0.0, 1.0\] (returns 0.0 when there are no positive predictions)
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            return 0.0;
        }
        self.tp as f64 / (self.tp + self.fp) as f64
    }

    /// Calculate recall (sensitivity): TP / (TP + FN).
    ///
    /// Recall measures how many actual positives are correctly identified.
    ///
    /// # Returns
    ///
    /// - `f64` - Recall in the range \[0.0, 1.0\] (returns 1.0 when there are no actual positives)
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 {
            return 1.0;
        }
        self.tp as f64 / (self.tp + self.fn_) as f64
    }

    /// Calculate specificity: TN / (TN + FP).
    ///
    /// Specificity measures how many actual negatives are correctly identified.
    ///
    /// # Returns
    ///
    /// - `f64` - Specificity in the range \[0.0, 1.0\] (returns 1.0 when there are no actual negatives)
    pub fn specificity(&self) -> f64 {
        if self.tn + self.fp == 0 {
            return 1.0;
        }
        self.tn as f64 / (self.tn + self.fp) as f64
    }

    /// Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall).
    ///
    /// F1 score is the harmonic mean of precision and recall, balancing both metrics.
    ///
    /// # Returns
    ///
    /// - `f64` - F1 score in the range \[0.0, 1.0\] (returns 0.0 when both precision and recall are 0.0)
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            return 0.0;
        }

        2.0 * (precision * recall) / (precision + recall)
    }

    /// Generate a formatted summary of the confusion matrix and derived metrics.
    ///
    /// # Returns
    ///
    /// - `String` - Text summary containing matrix counts and metrics with four decimal places
    pub fn summary(&self) -> String {
        format!(
            "Confusion Matrix:\n\
        |                | Predicted Positive | Predicted Negative |\n\
        |----------------|-------------------|--------------------|\n\
        | Actual Positive | TP: {}           | FN: {}             |\n\
        | Actual Negative | FP: {}           | TN: {}             |\n\
        \n\
        Performance Metrics:\n\
        - Accuracy: {:.4}\n\
        - Error Rate: {:.4}\n\
        - Precision: {:.4}\n\
        - Recall: {:.4}\n\
        - Specificity: {:.4}\n\
        - F1 Score: {:.4}",
            self.tp,
            self.fn_,
            self.fp,
            self.tn,
            self.accuracy(),
            self.error_rate(),
            self.precision(),
            self.recall(),
            self.specificity(),
            self.f1_score()
        )
    }
}

/// Calculate the accuracy of a classification model.
///
/// Accuracy is the proportion of correctly predicted samples over all samples.
///
/// # Parameters
///
/// - `predicted` - Predicted class labels
/// - `actual` - Ground-truth class labels
///
/// # Examples
/// ```rust
/// use rustyml::metric::accuracy;
/// use ndarray::array;
///
/// let predicted = array![0.0, 1.0, 1.0];
/// let actual = array![0.0, 0.0, 1.0];
/// let acc = accuracy(&predicted, &actual);
/// assert!((acc - 0.6666666666666667).abs() < 1e-6);
/// ```
///
/// # Returns
///
/// - `f64` - Accuracy in the range \[0.0, 1.0\]
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
/// - Panics if input arrays are empty
pub fn accuracy<S>(predicted: &ArrayBase<S, Ix1>, actual: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    if predicted.len() != actual.len() {
        panic!(
            "Input arrays must have the same length. Predicted: {}, Actual: {}",
            predicted.len(),
            actual.len()
        );
    }

    if predicted.is_empty() || actual.is_empty() {
        panic!("Input arrays must not be empty");
    }

    let correct_predictions = predicted
        .iter()
        .zip(actual.iter())
        .filter(|&(p, a)| (p - a).abs() < f64::EPSILON)
        .count();

    correct_predictions as f64 / predicted.len() as f64
}

/// Constructs a contingency matrix from two label arrays and returns:
/// - The contingency matrix
/// - Row sums (sizes of clusters in the ground truth)
/// - Column sums (sizes of clusters in the predicted labels)
fn contingency_matrix(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
    let mut label_to_index_true = AHashMap::new();
    let mut label_to_index_pred = AHashMap::new();
    let mut current_index_true = 0;
    let mut current_index_pred = 0;

    for &label in labels_true {
        label_to_index_true.entry(label).or_insert_with(|| {
            let idx = current_index_true;
            current_index_true += 1;
            idx
        });
    }
    for &label in labels_pred {
        label_to_index_pred.entry(label).or_insert_with(|| {
            let idx = current_index_pred;
            current_index_pred += 1;
            idx
        });
    }

    let n_rows = label_to_index_true.len();
    let n_cols = label_to_index_pred.len();
    let mut matrix = vec![vec![0usize; n_cols]; n_rows];

    for (&l_true, &l_pred) in labels_true.iter().zip(labels_pred.iter()) {
        let i = label_to_index_true[&l_true];
        let j = label_to_index_pred[&l_pred];
        matrix[i][j] += 1;
    }

    let row_sums = matrix
        .iter()
        .map(|row| row.iter().sum())
        .collect::<Vec<usize>>();
    let col_sums = (0..n_cols)
        .map(|j| matrix.iter().map(|row| row[j]).sum())
        .collect::<Vec<usize>>();

    (matrix, row_sums, col_sums)
}

/// Computes the mutual information (MI) using the formula:
/// MI = sum_{i,j} (n_ij/n) * ln((n * n_ij) / (a_i * b_j))
fn mutual_information(
    contingency: &Vec<Vec<usize>>,
    n: usize,
    row_sums: &Vec<usize>,
    col_sums: &Vec<usize>,
) -> f64 {
    let mut mi = 0.0;
    for (i, row) in contingency.iter().enumerate() {
        for (j, &n_ij) in row.iter().enumerate() {
            if n_ij > 0 {
                let n_ij_f = n_ij as f64;
                let a = row_sums[i] as f64;
                let b = col_sums[j] as f64;
                mi += (n_ij_f / n as f64) * ((n as f64 * n_ij_f) / (a * b)).ln();
            }
        }
    }
    mi
}

/// Computes the entropy H = - sum_i (p_i * ln(p_i))
fn entropy_nats(counts: &Vec<usize>, n: usize) -> f64 {
    let mut h = 0.0;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / n as f64;
            h -= p * p.ln();
        }
    }
    h
}

/// Computes the logarithm of the binomial coefficient C(n, k) using the log-gamma function.
/// This approach avoids overflow for large values by working in log space.
///
/// Returns ln(C(n, k)) = ln(n! / (k! * (n-k)!))
fn log_binomial_coefficient(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY; // C(n, k) = 0 when k > n
    }
    if k == 0 || k == n {
        return 0.0; // C(n, 0) = C(n, n) = 1, ln(1) = 0
    }

    // Use the more efficient k if k > n/2
    let k = if k > n - k { n - k } else { k };

    // Compute ln(n!) - ln(k!) - ln((n-k)!)
    // Using Stirling's approximation would be faster but less accurate
    // Instead, we compute the sum directly
    let mut result = 0.0;
    for i in 0..k {
        result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    result
}

/// Computes the probability mass function (PMF) of the hypergeometric distribution.
///
/// The hypergeometric distribution models the probability of k successes in n draws,
/// without replacement, from a finite population of size N that contains exactly K successes.
///
/// PMF: P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
///
/// # Parameters
/// - `n_population`: Total population size (N)
/// - `n_successes`: Number of success states in the population (K)
/// - `n_draws`: Number of draws (n)
/// - `k`: Number of observed successes
///
/// # Returns
/// The probability of observing exactly k successes
fn hypergeometric_pmf(n_population: u64, n_successes: u64, n_draws: u64, k: u64) -> f64 {
    // Check validity of parameters
    if n_successes > n_population || n_draws > n_population {
        return 0.0;
    }

    let n_failures = n_population - n_successes;

    // Check if k is in the valid range
    if k > n_successes || k > n_draws {
        return 0.0;
    }
    if n_draws - k > n_failures {
        return 0.0;
    }

    // Compute in log space to avoid overflow
    // ln(P) = ln(C(K,k)) + ln(C(N-K, n-k)) - ln(C(N, n))
    let log_prob = log_binomial_coefficient(n_successes, k)
        + log_binomial_coefficient(n_failures, n_draws - k)
        - log_binomial_coefficient(n_population, n_draws);

    log_prob.exp()
}

/// Computes the expected mutual information (EMI).
/// For each pair (a_i, b_j), assume that n_ij follows a Hypergeometric distribution with parameters:
/// - Total population size: n
/// - Number of successes: a_i
/// - Number of draws: b_j
///
/// EMI = sum_{i,j} sum_{k=max(0, a_i+b_j-n)}^{min(a_i, b_j)}
///       P(k) * (k/n) * ln((n * k) / (a_i * b_j))
fn expected_mutual_information(row_sums: &Vec<usize>, col_sums: &Vec<usize>, n: usize) -> f64 {
    let mut emi = 0.0;
    // For each pair of clusters (ground truth and predicted)
    for &a_i in row_sums {
        for &b_j in col_sums {
            // Skip invalid parameter combinations
            if a_i > n || b_j > n {
                continue;
            }

            // Valid range for k: from max(0, a_i+b_j-n) to min(a_i, b_j)
            let lower_bound = if a_i + b_j > n { a_i + b_j - n } else { 0 };
            // Skip k=0 since it contributes 0 to MI, so start from 1
            let lower = if lower_bound < 1 { 1 } else { lower_bound };
            let upper = std::cmp::min(a_i, b_j);

            for k in lower..=upper {
                // Calculate PMF using our own hypergeometric distribution implementation
                let p = hypergeometric_pmf(n as u64, a_i as u64, b_j as u64, k as u64);
                let term = (k as f64 / n as f64)
                    * ((n as f64 * k as f64) / (a_i as f64 * b_j as f64)).ln();
                emi += p * term;
            }
        }
    }
    emi
}

/// Calculates the Normalized Mutual Information (NMI) between two cluster label assignments.
///
/// NMI measures agreement between cluster assignments by normalizing mutual information with the geometric mean of individual entropies.
///
/// # Parameters
///
/// - `labels_true` - Ground-truth cluster assignments
/// - `labels_pred` - Predicted cluster assignments
///
/// # Examples
/// ```rust
/// use rustyml::metric::normalized_mutual_info;
/// use ndarray::array;
///
/// let true_labels = array![0, 0, 1, 1, 2, 2];
/// let pred_labels = array![0, 0, 1, 2, 1, 2];
///
/// let nmi = normalized_mutual_info(&true_labels, &pred_labels);
/// println!("Normalized Mutual Information: {:.4}", nmi);
/// ```
///
/// # Returns
///
/// - `f64` - Normalized mutual information score between 0.0 and 1.0
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
/// - Panics if input arrays are empty
pub fn normalized_mutual_info<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    if labels_true.len() != labels_pred.len() {
        panic!(
            "Input arrays must have the same length. Predicted: {}, Actual: {}",
            labels_true.len(),
            labels_pred.len()
        );
    }

    if labels_true.is_empty() || labels_pred.is_empty() {
        panic!("Input arrays cannot be empty");
    }

    let n = labels_true.len();

    let labels_true_slice: &[usize] = labels_true.as_slice().unwrap();
    let labels_pred_slice: &[usize] = labels_pred.as_slice().unwrap();

    let (contingency, row_sums, col_sums) =
        contingency_matrix(labels_true_slice, labels_pred_slice);
    let mi = mutual_information(&contingency, n, &row_sums, &col_sums);
    let h_true = entropy_nats(&row_sums, n);
    let h_pred = entropy_nats(&col_sums, n);
    if h_true * h_pred == 0.0 {
        0.0
    } else {
        mi / (h_true * h_pred).sqrt()
    }
}

/// Calculates the Adjusted Mutual Information (AMI) between two cluster label assignments.
///
/// AMI corrects mutual information for chance agreement, with scores near 1.0 for identical clusterings and around 0.0 for random agreement.
///
/// # Parameters
///
/// - `labels_true` - Ground-truth cluster assignments
/// - `labels_pred` - Predicted cluster assignments
///
/// # Examples
/// ```rust
/// use rustyml::metric::adjusted_mutual_info;
/// use ndarray::array;
///
/// let true_labels = array![0, 0, 1, 1, 2, 2];
/// let pred_labels = array![0, 0, 1, 2, 1, 2];
///
/// let ami = adjusted_mutual_info(&true_labels, &pred_labels);
/// println!("Adjusted Mutual Information: {:.4}", ami);
/// ```
///
/// # Returns
///
/// - `f64` - Adjusted mutual information score typically between -1.0 and 1.0
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
/// - Panics if input arrays are empty
pub fn adjusted_mutual_info<S>(
    labels_true: &ArrayBase<S, Ix1>,
    labels_pred: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = usize>,
{
    if labels_true.len() != labels_pred.len() {
        panic!(
            "Input arrays must have the same length. Predicted: {}, Actual: {}",
            labels_true.len(),
            labels_pred.len()
        );
    }

    if labels_true.is_empty() || labels_pred.is_empty() {
        panic!("Input arrays cannot be empty");
    }

    let n = labels_true.len();

    let labels_true_slice: &[usize] = labels_true.as_slice().unwrap();
    let labels_pred_slice: &[usize] = labels_pred.as_slice().unwrap();

    let (contingency, row_sums, col_sums) =
        contingency_matrix(labels_true_slice, labels_pred_slice);
    let mi = mutual_information(&contingency, n, &row_sums, &col_sums);
    let h_true = entropy_nats(&row_sums, n);
    let h_pred = entropy_nats(&col_sums, n);
    let emi = expected_mutual_information(&row_sums, &col_sums, n);
    let denominator = ((h_true + h_pred) / 2.0) - emi;
    if denominator.abs() < 1e-10 {
        1.0
    } else {
        (mi - emi) / denominator
    }
}

/// Calculates the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
///
/// Uses the Mann-Whitney U statistic to measure how well predicted scores rank positive samples above negative samples.
///
/// # Parameters
///
/// - `scores` - Predicted scores or probabilities for each sample
/// - `labels` - True class labels for each sample (true for positive, false for negative)
///
/// # Examples
/// ```rust
/// use rustyml::metric::calculate_auc;
/// use ndarray::array;
///
/// let scores = array![0.1, 0.4, 0.35, 0.8];
/// let labels = array![false, true, false, true];
/// let auc = calculate_auc(&scores, &labels);
/// println!("AUC-ROC: {}", auc);
/// ```
///
/// # Returns
///
/// - `f64` - AUC-ROC score between 0.0 and 1.0
///
/// # Panics
///
/// - Panics if the two arrays have different lengths
/// - Panics if input arrays are empty
/// - Panics if there are no positive or negative samples
pub fn calculate_auc<S1, S2>(scores: &ArrayBase<S1, Ix1>, labels: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = bool>,
{
    if scores.len() != labels.len() {
        panic!(
            "Input arrays must have the same length. Predicted: {}, Actual: {}",
            scores.len(),
            labels.len()
        );
    }
    if scores.is_empty() || labels.is_empty() {
        panic!("Input arrays cannot be empty");
    }

    // Pack the (score, label) pairs into a vector
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(s, &l)| (*s, l))
        .collect();

    // Sort by score in ascending order, using partial_cmp for floating-point numbers
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut pos_count = 0;
    let mut neg_count = 0;
    let mut sum_positive_ranks = 0.0;

    // Assign ranks to each sample in the sorted array (handling ties: use average rank for identical scores)
    let n = pairs.len();
    let mut i = 0;
    let mut rank = 1.0;

    while i < n {
        let current_score = pairs[i].0;
        let mut j = i;
        // Find all samples with the current score
        while j < n && (pairs[j].0 - current_score).abs() < 1e-12 {
            j += 1;
        }
        // Calculate the average rank for the tie: for k identical scores, the average rank is (rank + rank+1 + ... + rank+k-1)/k
        let count = (j - i) as f64;
        let avg_rank = (2.0 * rank + count - 1.0) / 2.0;
        for k in i..j {
            if pairs[k].1 {
                sum_positive_ranks += avg_rank;
                pos_count += 1;
            } else {
                neg_count += 1;
            }
        }
        rank += count;
        i = j;
    }

    // If there are no positive or negative samples, AUC cannot be calculated
    if pos_count == 0 || neg_count == 0 {
        panic!("AUC cannot be calculated because there are no positive or negative samples");
    }

    // Compute the Mann-Whitney U statistic
    let u = sum_positive_ranks - (pos_count as f64 * (pos_count as f64 + 1.0) / 2.0);
    // AUC is equal to the U statistic divided by (n_positive * n_negative)
    u / (pos_count as f64 * neg_count as f64)
}
