use ndarray::{ArrayBase, Data, Ix1};

use super::validate_pair;

/// Probability threshold at or above which a value is treated as the positive class.
const POSITIVE_THRESHOLD: f64 = 0.5;

/// Confusion matrix for binary classification evaluation.
///
/// Stores the counts of true positives, false positives, true negatives, and false negatives, and
/// exposes the metrics derived from them (accuracy, precision, recall, …).
///
/// # Examples
/// ```rust
/// use ndarray::arr1;
/// use rustyml::metric::ConfusionMatrix;
///
/// let y_true = arr1(&[1.0, 0.0, 1.0, 0.0, 1.0]);
/// let y_pred = arr1(&[0.9, 0.2, 0.8, 0.1, 0.7]);
/// let cm = ConfusionMatrix::new(&y_true, &y_pred);
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfusionMatrix {
    tp: usize,
    fp: usize,
    tn: usize,
    fn_: usize,
}

impl ConfusionMatrix {
    /// Builds a confusion matrix from ground-truth labels and predictions.
    ///
    /// Both arrays are binarized at a `0.5` threshold (`>= 0.5` is positive), so they may be hard
    /// `0.0`/`1.0` labels or probabilities. Thresholding `y_true` as well is harmless for hard
    /// labels and lets probabilistic ground truth be used directly.
    ///
    /// # Parameters
    ///
    /// - `y_true` - Ground-truth labels or probabilities (`>= 0.5` treated as positive)
    /// - `y_pred` - Predicted labels or probabilities (`>= 0.5` treated as positive)
    ///
    /// # Returns
    ///
    /// - `Self` - Confusion matrix with populated counts
    ///
    /// # Panics
    ///
    /// - Panics if `y_true` and `y_pred` have different lengths
    /// - Panics if the inputs are empty
    pub fn new<S>(y_true: &ArrayBase<S, Ix1>, y_pred: &ArrayBase<S, Ix1>) -> Self
    where
        S: Data<Elem = f64>,
    {
        validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            let true_positive = t >= POSITIVE_THRESHOLD;
            let pred_positive = p >= POSITIVE_THRESHOLD;
            match (true_positive, pred_positive) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        Self { tp, fp, tn, fn_ }
    }

    /// Returns the raw counts as `(tp, fp, tn, fn)`.
    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.tp, self.fp, self.tn, self.fn_)
    }

    /// Calculates accuracy: `(TP + TN) / (TP + TN + FP + FN)`.
    ///
    /// # Returns
    ///
    /// - `f64` - Accuracy in `[0.0, 1.0]` (returns 0.0 when there are no predictions)
    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.tn + self.fp + self.fn_;
        if total == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / total as f64
    }

    /// Calculates the error rate: `1 - accuracy`.
    ///
    /// # Returns
    ///
    /// - `f64` - Error rate in `[0.0, 1.0]`
    pub fn error_rate(&self) -> f64 {
        1.0 - self.accuracy()
    }

    /// Calculates precision: `TP / (TP + FP)`.
    ///
    /// # Returns
    ///
    /// - `f64` - Precision in `[0.0, 1.0]` (returns 0.0 when there are no positive predictions)
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            return 0.0;
        }
        self.tp as f64 / (self.tp + self.fp) as f64
    }

    /// Calculates recall (sensitivity): `TP / (TP + FN)`.
    ///
    /// # Returns
    ///
    /// - `f64` - Recall in `[0.0, 1.0]` (returns 1.0 when there are no actual positives, avoiding
    ///   a 0/0 division)
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 {
            return 1.0;
        }
        self.tp as f64 / (self.tp + self.fn_) as f64
    }

    /// Calculates specificity: `TN / (TN + FP)`.
    ///
    /// # Returns
    ///
    /// - `f64` - Specificity in `[0.0, 1.0]` (returns 1.0 when there are no actual negatives,
    ///   avoiding a 0/0 division)
    pub fn specificity(&self) -> f64 {
        if self.tn + self.fp == 0 {
            return 1.0;
        }
        self.tn as f64 / (self.tn + self.fp) as f64
    }

    /// Calculates the F1 score: the harmonic mean of precision and recall.
    ///
    /// # Returns
    ///
    /// - `f64` - F1 score in `[0.0, 1.0]` (returns 0.0 when both precision and recall are 0.0)
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            return 0.0;
        }

        2.0 * (precision * recall) / (precision + recall)
    }

    /// Renders the matrix and its derived metrics as a formatted, human-readable table.
    ///
    /// # Returns
    ///
    /// - `String` - Text summary of the counts and metrics (metrics to four decimal places)
    pub fn summary(&self) -> String {
        let sep = "+-----------------+--------------------+--------------------+";
        format!(
            "Confusion Matrix:\n\
             {sep}\n\
             | {:<15} | {:<18} | {:<18} |\n\
             {sep}\n\
             | {:<15} | {:<18} | {:<18} |\n\
             | {:<15} | {:<18} | {:<18} |\n\
             {sep}\n\
             \n\
             Performance Metrics:\n\
             - Accuracy:    {:.4}\n\
             - Error Rate:  {:.4}\n\
             - Precision:   {:.4}\n\
             - Recall:      {:.4}\n\
             - Specificity: {:.4}\n\
             - F1 Score:    {:.4}",
            "",
            "Predicted Positive",
            "Predicted Negative",
            "Actual Positive",
            format!("TP: {}", self.tp),
            format!("FN: {}", self.fn_),
            "Actual Negative",
            format!("FP: {}", self.fp),
            format!("TN: {}", self.tn),
            self.accuracy(),
            self.error_rate(),
            self.precision(),
            self.recall(),
            self.specificity(),
            self.f1_score(),
        )
    }
}

/// Calculates the accuracy of a classification model: the fraction of exactly-matching labels.
///
/// Labels are compared for equality within [`f64::EPSILON`], so this works for integer-valued
/// class labels stored as `f64` in both binary and multi-class settings. The comparison is
/// symmetric, so the argument order does not affect the result.
///
/// # Parameters
///
/// - `y_true` - Ground-truth class labels
/// - `y_pred` - Predicted class labels
///
/// # Returns
///
/// - `f64` - Accuracy in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `y_true` and `y_pred` have different lengths
/// - Panics if the inputs are empty
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::accuracy;
///
/// let y_true = array![0.0, 0.0, 1.0];
/// let y_pred = array![0.0, 1.0, 1.0];
/// let acc = accuracy(&y_true, &y_pred);
/// assert!((acc - 0.6666666666666667).abs() < 1e-6);
/// ```
pub fn accuracy<S>(y_true: &ArrayBase<S, Ix1>, y_pred: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = f64>,
{
    validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(t, p)| (t - p).abs() < f64::EPSILON)
        .count();

    correct as f64 / y_true.len() as f64
}

/// Calculates the Area Under the ROC Curve (ROC AUC) for binary classification.
///
/// Uses the Mann-Whitney U statistic, which equals the probability that a randomly chosen positive
/// sample is scored above a randomly chosen negative one. Tied scores receive their average rank.
///
/// # Parameters
///
/// - `labels` - True class labels (`true` for positive, `false` for negative)
/// - `scores` - Predicted scores or probabilities for each sample
///
/// # Returns
///
/// - `f64` - ROC AUC in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels` and `scores` have different lengths
/// - Panics if the inputs are empty
/// - Panics if the labels do not contain both a positive and a negative sample
///
/// # Examples
/// ```rust
/// use ndarray::array;
/// use rustyml::metric::roc_auc;
///
/// let labels = array![false, true, false, true];
/// let scores = array![0.1, 0.4, 0.35, 0.8];
/// let auc = roc_auc(&labels, &scores);
/// println!("ROC AUC: {}", auc);
/// ```
pub fn roc_auc<S1, S2>(labels: &ArrayBase<S1, Ix1>, scores: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = bool>,
    S2: Data<Elem = f64>,
{
    validate_pair(labels.len(), scores.len(), "labels and scores");

    // Pack (score, label) pairs and sort by score ascending. `total_cmp` gives a total order, so
    // NaN scores are ordered deterministically instead of panicking.
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut pos_count = 0usize;
    let mut neg_count = 0usize;
    let mut sum_positive_ranks = 0.0;

    // Walk the sorted pairs, assigning each tie group its average rank.
    let n = pairs.len();
    let mut i = 0;
    let mut rank = 1.0;
    while i < n {
        let current_score = pairs[i].0;
        let mut j = i;
        while j < n && pairs[j].0 == current_score {
            j += 1;
        }
        // Average rank of the tie group [i, j): (rank + ... + rank + count - 1) / count.
        let count = (j - i) as f64;
        let avg_rank = (2.0 * rank + count - 1.0) / 2.0;
        for pair in &pairs[i..j] {
            if pair.1 {
                sum_positive_ranks += avg_rank;
                pos_count += 1;
            } else {
                neg_count += 1;
            }
        }
        rank += count;
        i = j;
    }

    if pos_count == 0 || neg_count == 0 {
        panic!("invalid input: ROC AUC requires at least one positive and one negative label");
    }

    // AUC = U / (n_pos * n_neg), with U the Mann-Whitney statistic.
    let u = sum_positive_ranks - (pos_count as f64 * (pos_count as f64 + 1.0) / 2.0);
    u / (pos_count as f64 * neg_count as f64)
}
