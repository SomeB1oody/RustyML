//! Classification evaluation metrics
//!
//! Provides binary and multi-class confusion matrices and the scores derived from them (accuracy,
//! precision, recall, F1, MCC, balanced accuracy, Cohen's kappa, log loss, top-k accuracy) plus
//! threshold-curve tools (ROC AUC, ROC curve, average precision, precision-recall curve)

use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2};

use super::validate_pair;

/// Probability threshold at or above which a value is treated as the positive class
const POSITIVE_THRESHOLD: f64 = 0.5;

/// Denominator magnitude below which a chance-corrected score is treated as degenerate (perfect)
const DEGENERATE_DENOM: f64 = 1e-10;

/// Averaging strategy for multi-class precision, recall, and F1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    /// Unweighted mean of the per-class scores - every class counts equally regardless of size
    Macro,
    /// Score from the pooled counts across all classes; for single-label data it equals accuracy
    Micro,
    /// Mean of the per-class scores weighted by each class's support (its number of true samples)
    Weighted,
}

/// Confusion matrix for binary classification evaluation
///
/// Stores the counts of true positives, false positives, true negatives, and false negatives, and
/// exposes the metrics derived from them (accuracy, precision, recall, ...)
///
/// # Examples
///
/// ```rust
/// use ndarray::arr1;
/// use rustyml::metrics::ConfusionMatrix;
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
    /// Builds a confusion matrix from ground-truth labels and predictions
    ///
    /// Both arrays are binarized at a `0.5` threshold (`>= 0.5` is positive), so they may be hard
    /// `0.0`/`1.0` labels or probabilities. Thresholding `y_true` as well is harmless for hard
    /// labels and lets probabilistic ground truth be used directly
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

    /// Returns the raw counts as `(tp, fp, tn, fn)`
    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.tp, self.fp, self.tn, self.fn_)
    }

    /// Calculates accuracy: `(TP + TN) / (TP + TN + FP + FN)`
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

    /// Calculates the error rate: `1 - accuracy`
    ///
    /// # Returns
    ///
    /// - `f64` - Error rate in `[0.0, 1.0]`
    pub fn error_rate(&self) -> f64 {
        1.0 - self.accuracy()
    }

    /// Calculates precision: `TP / (TP + FP)`
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

    /// Calculates recall (sensitivity): `TP / (TP + FN)`
    ///
    /// # Returns
    ///
    /// - `f64` - Recall in `[0.0, 1.0]` (returns 0.0 when there are no actual positives)
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 {
            return 0.0;
        }
        self.tp as f64 / (self.tp + self.fn_) as f64
    }

    /// Calculates specificity: `TN / (TN + FP)`
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

    /// Calculates the F1 score: the harmonic mean of precision and recall
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

    /// Calculates the Matthews correlation coefficient (MCC)
    ///
    /// MCC is a balanced measure usable even when the classes are very different sizes, ranging
    /// from -1 (total disagreement) through 0 (random) to +1 (perfect). Returns 0.0 when any of
    /// the four marginal sums is zero (the coefficient is undefined)
    ///
    /// # Returns
    ///
    /// - `f64` - Matthews correlation coefficient in `[-1.0, 1.0]`
    pub fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let tn = self.tn as f64;
        let fp = self.fp as f64;
        let fn_ = self.fn_ as f64;
        let numerator = tp * tn - fp * fn_;
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculates balanced accuracy: the mean of recall and specificity
    ///
    /// Unlike plain accuracy this is not inflated by a dominant negative class, so it is a better
    /// summary on imbalanced data
    ///
    /// # Returns
    ///
    /// - `f64` - Balanced accuracy in `[0.0, 1.0]`
    pub fn balanced_accuracy(&self) -> f64 {
        (self.recall() + self.specificity()) / 2.0
    }

    /// Renders the matrix and its derived metrics as a formatted, human-readable table
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
             - {:<18} {:.4}\n\
             - {:<18} {:.4}\n\
             - {:<18} {:.4}\n\
             - {:<18} {:.4}\n\
             - {:<18} {:.4}\n\
             - {:<18} {:.4}\n\
             - {:<18} {:.4}\n\
             - {:<18} {:.4}",
            "",
            "Predicted Positive",
            "Predicted Negative",
            "Actual Positive",
            format!("TP: {}", self.tp),
            format!("FN: {}", self.fn_),
            "Actual Negative",
            format!("FP: {}", self.fp),
            format!("TN: {}", self.tn),
            "Accuracy:",
            self.accuracy(),
            "Balanced Accuracy:",
            self.balanced_accuracy(),
            "Error Rate:",
            self.error_rate(),
            "Precision:",
            self.precision(),
            "Recall:",
            self.recall(),
            "Specificity:",
            self.specificity(),
            "F1 Score:",
            self.f1_score(),
            "MCC:",
            self.mcc(),
        )
    }
}

/// Calculates the accuracy of a classification model: the fraction of exactly-matching labels
///
/// Labels are compared for equality within [`f64::EPSILON`], so this works for integer-valued
/// class labels stored as `f64` in both binary and multi-class settings. The comparison is
/// symmetric, so the argument order does not affect the result
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
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::accuracy;
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

/// Rejects `NaN` scores from the input array
///
/// Panics if any score is `NaN`
#[inline]
fn reject_nan_scores<S>(scores: &ArrayBase<S, Ix1>)
where
    S: Data<Elem = f64>,
{
    if scores.iter().any(|s| s.is_nan()) {
        panic!("invalid input: scores must not contain NaN");
    }
}

/// Calculates the Area Under the ROC Curve (ROC AUC) for binary classification
///
/// Uses the Mann-Whitney U statistic, which equals the probability that a randomly chosen positive
/// sample is scored above a randomly chosen negative one. Tied scores receive their average rank
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
/// - Panics if `scores` contains `NaN` (it cannot be ranked meaningfully)
/// - Panics if the labels do not contain both a positive and a negative sample
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::roc_auc;
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
    reject_nan_scores(scores);

    // Pack (score, label) pairs and sort by score ascending
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut pos_count = 0usize;
    let mut neg_count = 0usize;
    let mut sum_positive_ranks = 0.0;

    // Walk the sorted pairs, assigning each tie group its average rank
    let n = pairs.len();
    let mut i = 0;
    let mut rank = 1.0;
    while i < n {
        let current_score = pairs[i].0;
        let mut j = i;
        while j < n && pairs[j].0 == current_score {
            j += 1;
        }
        // Average rank of the tie group [i, j): (rank + ... + rank + count - 1) / count
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

    // AUC = U / (n_pos * n_neg), with U the Mann-Whitney statistic
    let u = sum_positive_ranks - (pos_count as f64 * (pos_count as f64 + 1.0) / 2.0);
    u / (pos_count as f64 * neg_count as f64)
}

/// Confusion matrix for multi-class classification
///
/// Built from two arrays of integer class labels, it holds the full `K x K` count matrix
/// (`matrix[[i, j]]` = number of samples whose true class is the `i`-th label and predicted class
/// is the `j`-th) and derives per-class and averaged precision, recall, and F1. The class axis is
/// the sorted union of the labels seen in either input
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::{Average, MulticlassConfusionMatrix};
///
/// let y_true = array![0, 1, 2, 2, 1];
/// let y_pred = array![0, 2, 2, 2, 1];
/// let cm = MulticlassConfusionMatrix::new(&y_true, &y_pred);
///
/// assert_eq!(cm.n_classes(), 3);
/// assert!((cm.accuracy() - 0.8).abs() < 1e-12); // 4 of 5 correct
/// let _macro_f1 = cm.f1(Average::Macro);
/// println!("{}", cm.summary());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulticlassConfusionMatrix {
    /// Row = true class index, column = predicted class index
    matrix: Array2<usize>,
    /// The sorted distinct labels; index `i` corresponds to `labels[i]`
    labels: Vec<usize>,
}

impl MulticlassConfusionMatrix {
    /// Builds a multi-class confusion matrix from integer ground-truth and predicted labels
    ///
    /// # Parameters
    ///
    /// - `y_true` - Ground-truth class label for each sample
    /// - `y_pred` - Predicted class label for each sample
    ///
    /// # Returns
    ///
    /// - `Self` - Confusion matrix whose class axis is the sorted union of the observed labels
    ///
    /// # Panics
    ///
    /// - Panics if `y_true` and `y_pred` have different lengths
    /// - Panics if the inputs are empty
    pub fn new<S>(y_true: &ArrayBase<S, Ix1>, y_pred: &ArrayBase<S, Ix1>) -> Self
    where
        S: Data<Elem = usize>,
    {
        validate_pair(y_true.len(), y_pred.len(), "y_true and y_pred");

        let mut labels: Vec<usize> = y_true.iter().chain(y_pred.iter()).copied().collect();
        labels.sort_unstable();
        labels.dedup();
        let index: AHashMap<usize, usize> =
            labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();

        let k = labels.len();
        let mut matrix = Array2::<usize>::zeros((k, k));
        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            matrix[[index[&t], index[&p]]] += 1;
        }

        Self { matrix, labels }
    }

    /// Returns a view of the raw `K x K` count matrix (row = true class, column = predicted)
    pub fn matrix(&self) -> ArrayView2<'_, usize> {
        self.matrix.view()
    }

    /// Returns the class labels in the order they index the matrix (sorted ascending)
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Returns the number of distinct classes (the matrix dimension)
    pub fn n_classes(&self) -> usize {
        self.labels.len()
    }

    /// Returns the support of each class: the number of ground-truth samples per class, in label
    /// order
    pub fn support(&self) -> Vec<usize> {
        self.matrix.sum_axis(Axis(1)).to_vec()
    }

    /// Calculates overall accuracy: correct predictions (the matrix trace) over all samples
    ///
    /// # Returns
    ///
    /// - `f64` - Accuracy in `[0.0, 1.0]`
    pub fn accuracy(&self) -> f64 {
        let total: usize = self.matrix.sum();
        if total == 0 {
            return 0.0;
        }
        let correct: usize = (0..self.labels.len()).map(|i| self.matrix[[i, i]]).sum();
        correct as f64 / total as f64
    }

    /// Per-class precision (`TP / (TP + FP)`), in label order; 0.0 for a class never predicted
    pub fn per_class_precision(&self) -> Vec<f64> {
        let predicted = self.matrix.sum_axis(Axis(0));
        (0..self.labels.len())
            .map(|i| {
                let tp = self.matrix[[i, i]];
                if predicted[i] == 0 {
                    0.0
                } else {
                    tp as f64 / predicted[i] as f64
                }
            })
            .collect()
    }

    /// Per-class recall (`TP / (TP + FN)`), in label order; 0.0 for a class with no true samples
    pub fn per_class_recall(&self) -> Vec<f64> {
        let actual = self.matrix.sum_axis(Axis(1));
        (0..self.labels.len())
            .map(|i| {
                let tp = self.matrix[[i, i]];
                if actual[i] == 0 {
                    0.0
                } else {
                    tp as f64 / actual[i] as f64
                }
            })
            .collect()
    }

    /// Per-class F1 score (harmonic mean of precision and recall), in label order
    pub fn per_class_f1(&self) -> Vec<f64> {
        self.per_class_precision()
            .iter()
            .zip(self.per_class_recall().iter())
            .map(|(&p, &r)| {
                if p + r == 0.0 {
                    0.0
                } else {
                    2.0 * p * r / (p + r)
                }
            })
            .collect()
    }

    /// Precision aggregated across classes by the given [`Average`] strategy
    pub fn precision(&self, average: Average) -> f64 {
        self.aggregate(&self.per_class_precision(), average)
    }

    /// Recall aggregated across classes by the given [`Average`] strategy
    pub fn recall(&self, average: Average) -> f64 {
        self.aggregate(&self.per_class_recall(), average)
    }

    /// F1 score aggregated across classes by the given [`Average`] strategy
    pub fn f1(&self, average: Average) -> f64 {
        self.aggregate(&self.per_class_f1(), average)
    }

    /// Combines per-class scores by `average`. For [`Average::Micro`] every per-class metric
    /// collapses to accuracy on single-label data, so accuracy is returned directly
    fn aggregate(&self, per_class: &[f64], average: Average) -> f64 {
        match average {
            Average::Micro => self.accuracy(),
            Average::Macro => {
                if per_class.is_empty() {
                    0.0
                } else {
                    per_class.iter().sum::<f64>() / per_class.len() as f64
                }
            }
            Average::Weighted => {
                let support = self.support();
                let total: usize = support.iter().sum();
                if total == 0 {
                    return 0.0;
                }
                per_class
                    .iter()
                    .zip(support.iter())
                    .map(|(&m, &s)| m * s as f64)
                    .sum::<f64>()
                    / total as f64
            }
        }
    }

    /// Renders the `K x K` count matrix followed by a per-class precision/recall/F1/support table
    /// with accuracy and macro/weighted averages (scikit-learn's `classification_report` style)
    ///
    /// Mirrors [`ConfusionMatrix::summary`]: a `Confusion Matrix:` grid first, then the metrics
    ///
    /// # Returns
    ///
    /// - `String` - Formatted multi-line summary (metrics to four decimal places)
    pub fn summary(&self) -> String {
        // Confusion matrix as a bordered grid (rows = true classes, columns = predicted), with a
        // cell width sized to the widest label or count so it works for any number of classes
        let width = self
            .labels
            .iter()
            .map(|label| label.to_string().len())
            .chain(self.matrix.iter().map(|count| count.to_string().len()))
            .max()
            .unwrap_or(1);
        let mut border = String::from("+");
        for _ in 0..=self.labels.len() {
            border.push_str(&"-".repeat(width + 2));
            border.push('+');
        }
        let make_row = |header: &str, cells: Vec<String>| -> String {
            let mut line = format!("| {header:>width$} |");
            for cell in cells {
                line.push_str(&format!(" {cell:>width$} |"));
            }
            line
        };

        let mut out = String::from("Confusion Matrix (rows = true, columns = predicted):\n");
        out.push_str(&border);
        out.push('\n');
        out.push_str(&make_row(
            "",
            self.labels.iter().map(|l| l.to_string()).collect(),
        ));
        out.push('\n');
        out.push_str(&border);
        out.push('\n');
        for (i, &label) in self.labels.iter().enumerate() {
            let counts = self
                .matrix
                .row(i)
                .iter()
                .map(|count| count.to_string())
                .collect();
            out.push_str(&make_row(&label.to_string(), counts));
            out.push('\n');
        }
        out.push_str(&border);
        out.push('\n');
        out.push('\n');

        // Per-class metrics table
        let precision = self.per_class_precision();
        let recall = self.per_class_recall();
        let f1 = self.per_class_f1();
        let support = self.support();
        let total: usize = support.iter().sum();

        out.push_str(&format!(
            "{:>12} {:>10} {:>10} {:>10} {:>10}\n\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        for (i, &label) in self.labels.iter().enumerate() {
            out.push_str(&format!(
                "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                label, precision[i], recall[i], f1[i], support[i]
            ));
        }
        out.push('\n');
        out.push_str(&format!(
            "{:>12} {:>10} {:>10} {:>10.4} {:>10}\n",
            "accuracy",
            "",
            "",
            self.accuracy(),
            total
        ));
        out.push_str(&format!(
            "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "macro avg",
            self.precision(Average::Macro),
            self.recall(Average::Macro),
            self.f1(Average::Macro),
            total
        ));
        out.push_str(&format!(
            "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            "weighted avg",
            self.precision(Average::Weighted),
            self.recall(Average::Weighted),
            self.f1(Average::Weighted),
            total
        ));
        out
    }
}

/// Calculates the multi-class logarithmic loss (cross-entropy) of predicted probabilities
///
/// For each sample only the probability assigned to its true class contributes:
/// `-mean(ln(p[i, y_true[i]]))`. Each row of `y_prob` is first renormalized to sum to 1 (matching
/// scikit-learn), so rows that do not already sum to 1 are scored consistently; the selected
/// probability is then clamped away from 0 and 1 to keep the logarithm finite. Each value of
/// `y_true` indexes a column of `y_prob`
///
/// # Parameters
///
/// - `y_true` - True class index of each sample (must be a valid column index into `y_prob`)
/// - `y_prob` - Predicted class probabilities, one row per sample and one column per class
///
/// # Returns
///
/// - `f64` - Mean cross-entropy (>= 0.0; lower is better)
///
/// # Panics
///
/// - Panics if `y_true`'s length differs from the number of rows in `y_prob`
/// - Panics if the inputs are empty
/// - Panics if any label is not a valid column index into `y_prob`
///
/// # Examples
///
/// ```rust
/// use ndarray::{array, arr2};
/// use rustyml::metrics::log_loss;
///
/// let y_true = array![0, 1];
/// let y_prob = arr2(&[[0.9, 0.1], [0.2, 0.8]]);
/// let loss = log_loss(&y_true, &y_prob);
/// // -(ln(0.9) + ln(0.8)) / 2
/// assert!((loss - 0.164252033).abs() < 1e-6);
/// ```
pub fn log_loss<S1, S2>(y_true: &ArrayBase<S1, Ix1>, y_prob: &ArrayBase<S2, Ix2>) -> f64
where
    S1: Data<Elem = usize>,
    S2: Data<Elem = f64>,
{
    let n = y_true.len();
    if n != y_prob.nrows() {
        panic!("dimension mismatch: expected {n}, found {}", y_prob.nrows());
    }
    if n == 0 {
        panic!("input is empty: y_true and y_prob");
    }

    const EPS: f64 = 1e-15;
    let n_classes = y_prob.ncols();
    let mut total = 0.0;
    for (i, &label) in y_true.iter().enumerate() {
        if label >= n_classes {
            panic!(
                "invalid input: label {label} is out of range for {n_classes} probability columns"
            );
        }
        // Renormalize the row to a probability distribution (scikit-learn does the same); the row
        // sum is floored at EPS so an all-zero row divides cleanly instead of producing NaN
        let row_sum: f64 = y_prob.row(i).sum();
        let p = (y_prob[[i, label]] / row_sum.max(EPS)).clamp(EPS, 1.0 - EPS);
        total -= p.ln();
    }
    total / n as f64
}

/// Calculates Cohen's kappa: agreement between two labelings corrected for chance
///
/// `kappa = (p_o - p_e) / (1 - p_e)`, where `p_o` is the observed agreement (accuracy) and `p_e`
/// the agreement expected from the marginal label frequencies. Scores range from -1 to 1; 1 is
/// perfect agreement and 0 is chance-level. Returns 1.0 when chance agreement is already total
///
/// # Parameters
///
/// - `y_true` - First labeling (e.g. ground truth)
/// - `y_pred` - Second labeling (e.g. predictions)
///
/// # Returns
///
/// - `f64` - Cohen's kappa in `[-1.0, 1.0]`
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
/// use rustyml::metrics::cohen_kappa;
///
/// let y_true = array![0, 1, 0, 1];
/// let y_pred = array![0, 1, 0, 1];
/// assert!((cohen_kappa(&y_true, &y_pred) - 1.0).abs() < 1e-12);
/// ```
pub fn cohen_kappa<S>(y_true: &ArrayBase<S, Ix1>, y_pred: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = usize>,
{
    let cm = MulticlassConfusionMatrix::new(y_true, y_pred);
    let n = cm.matrix.sum() as f64;
    let actual = cm.matrix.sum_axis(Axis(1));
    let predicted = cm.matrix.sum_axis(Axis(0));

    let observed = cm.accuracy();
    let expected: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a as f64 / n) * (p as f64 / n))
        .sum();

    if (1.0 - expected).abs() < DEGENERATE_DENOM {
        1.0
    } else {
        (observed - expected) / (1.0 - expected)
    }
}

/// Calculates top-k accuracy: the fraction of samples whose true class is among the `k`
/// highest-probability predicted classes
///
/// A class ties into the top-k set if fewer than `k` classes are assigned a strictly higher
/// probability, so ties at the boundary count in the sample's favor
///
/// # Parameters
///
/// - `y_true` - True class index of each sample (must be a valid column index into `y_prob`)
/// - `y_prob` - Predicted class probabilities, one row per sample and one column per class
/// - `k` - Number of top predictions to consider
///
/// # Returns
///
/// - `f64` - Top-k accuracy in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `y_true`'s length differs from the number of rows in `y_prob`
/// - Panics if the inputs are empty
/// - Panics if `k` is zero, or if any label is out of range for `y_prob`'s columns
/// - Panics if `y_prob` contains `NaN`: a `NaN` true-class probability makes every `p > true_prob`
///   comparison false, which would otherwise miscount the sample as a hit
///
/// # Examples
///
/// ```rust
/// use ndarray::{array, arr2};
/// use rustyml::metrics::top_k_accuracy;
///
/// let y_true = array![0, 1, 2];
/// let y_prob = arr2(&[[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.5, 0.4]]);
/// // Every true class is within the two most probable, so top-2 accuracy is 1.0
/// assert!((top_k_accuracy(&y_true, &y_prob, 2) - 1.0).abs() < 1e-12);
/// ```
pub fn top_k_accuracy<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_prob: &ArrayBase<S2, Ix2>,
    k: usize,
) -> f64
where
    S1: Data<Elem = usize>,
    S2: Data<Elem = f64>,
{
    let n = y_true.len();
    if n != y_prob.nrows() {
        panic!("dimension mismatch: expected {n}, found {}", y_prob.nrows());
    }
    if n == 0 {
        panic!("input is empty: y_true and y_prob");
    }
    if k == 0 {
        panic!("invalid parameter `k`: must be at least 1");
    }

    let n_classes = y_prob.ncols();
    let mut correct = 0;
    for (i, &label) in y_true.iter().enumerate() {
        if label >= n_classes {
            panic!(
                "invalid input: label {label} is out of range for {n_classes} probability columns"
            );
        }
        let true_prob = y_prob[[i, label]];
        // A NaN anywhere in the row breaks the `p > true_prob` ranking (every comparison with NaN
        // is false), so reject it rather than silently miscount the sample
        let mut n_greater = 0;
        for &p in y_prob.row(i) {
            if p.is_nan() {
                panic!("invalid input: y_prob must not contain NaN");
            }
            if p > true_prob {
                n_greater += 1;
            }
        }
        if n_greater < k {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

/// Sorts samples by score descending and returns, at each distinct score, the cumulative
/// `(threshold, cumulative_true_positives, cumulative_false_positives)`, plus the total positive
/// and negative counts. Shared by the precision-recall / ROC curve builders
///
/// Panics on a length mismatch, empty input, or a `NaN` score (see [`reject_nan_scores`]), so all
/// callers (`average_precision`, `roc_curve`, `precision_recall_curve`) inherit those guarantees
fn ranked_cumulative<S1, S2>(
    labels: &ArrayBase<S1, Ix1>,
    scores: &ArrayBase<S2, Ix1>,
) -> (Vec<(f64, usize, usize)>, usize, usize)
where
    S1: Data<Elem = bool>,
    S2: Data<Elem = f64>,
{
    validate_pair(labels.len(), scores.len(), "labels and scores");
    reject_nan_scores(scores);

    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0)); // descending score

    let total_pos = pairs.iter().filter(|p| p.1).count();
    let total_neg = pairs.len() - total_pos;

    let mut points = Vec::new();
    let (mut tp, mut fp) = (0usize, 0usize);
    let n = pairs.len();
    let mut i = 0;
    while i < n {
        let score = pairs[i].0;
        while i < n && pairs[i].0 == score {
            if pairs[i].1 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        points.push((score, tp, fp));
    }
    (points, total_pos, total_neg)
}

/// Calculates average precision: the area under the precision-recall curve, computed as the
/// precision-weighted sum of recall increments `sum (R_n - R_{n-1}) * P_n`
///
/// # Parameters
///
/// - `labels` - True binary labels (`true` for positive)
/// - `scores` - Predicted scores or probabilities
///
/// # Returns
///
/// - `f64` - Average precision in `[0.0, 1.0]`
///
/// # Panics
///
/// - Panics if `labels` and `scores` have different lengths
/// - Panics if the inputs are empty
/// - Panics if `scores` contains `NaN`
/// - Panics if there are no positive labels
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::average_precision;
///
/// let labels = array![true, false, true, false];
/// let scores = array![0.9, 0.6, 0.4, 0.1];
/// // Ranked T,F,T,F: AP = 1.0*0.5 + (2/3)*0.5 = 0.8333...
/// assert!((average_precision(&labels, &scores) - 0.833333333).abs() < 1e-6);
/// ```
pub fn average_precision<S1, S2>(labels: &ArrayBase<S1, Ix1>, scores: &ArrayBase<S2, Ix1>) -> f64
where
    S1: Data<Elem = bool>,
    S2: Data<Elem = f64>,
{
    let (points, total_pos, _) = ranked_cumulative(labels, scores);
    if total_pos == 0 {
        panic!("invalid input: average precision requires at least one positive label");
    }

    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for &(_, tp, fp) in &points {
        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / total_pos as f64;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    ap
}

/// Computes the Receiver Operating Characteristic (ROC) curve
///
/// Returns `(fpr, tpr, thresholds)`: the false-positive rate and true-positive rate at each
/// distinct score threshold (in decreasing order), preceded by the `(0, 0)` origin. `thresholds`
/// holds the score at each point; its leading entry is set above the maximum score so the origin
/// classifies nothing as positive
///
/// # Parameters
///
/// - `labels` - True binary labels (`true` for positive)
/// - `scores` - Predicted scores or probabilities
///
/// # Returns
///
/// - `(Array1<f64>, Array1<f64>, Array1<f64>)` - `(fpr, tpr, thresholds)`, all of equal length
///
/// # Panics
///
/// - Panics if `labels` and `scores` have different lengths
/// - Panics if the inputs are empty
/// - Panics if `scores` contains `NaN`
/// - Panics if the labels do not contain both a positive and a negative sample
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::roc_curve;
///
/// let labels = array![true, false, true, false];
/// let scores = array![0.9, 0.6, 0.4, 0.1];
/// let (fpr, tpr, thresholds) = roc_curve(&labels, &scores);
/// assert_eq!(fpr.len(), tpr.len());
/// assert_eq!(thresholds.len(), fpr.len());
/// assert_eq!(fpr[0], 0.0);
/// assert_eq!(tpr[tpr.len() - 1], 1.0);
/// ```
pub fn roc_curve<S1, S2>(
    labels: &ArrayBase<S1, Ix1>,
    scores: &ArrayBase<S2, Ix1>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>)
where
    S1: Data<Elem = bool>,
    S2: Data<Elem = f64>,
{
    let (points, total_pos, total_neg) = ranked_cumulative(labels, scores);
    if total_pos == 0 || total_neg == 0 {
        panic!("invalid input: ROC curve requires at least one positive and one negative label");
    }

    let mut fpr = Vec::with_capacity(points.len() + 1);
    let mut tpr = Vec::with_capacity(points.len() + 1);
    let mut thresholds = Vec::with_capacity(points.len() + 1);

    // Origin: no sample classified positive, threshold above the maximum score
    fpr.push(0.0);
    tpr.push(0.0);
    thresholds.push(points[0].0 + 1.0);

    for &(score, tp, fp) in &points {
        tpr.push(tp as f64 / total_pos as f64);
        fpr.push(fp as f64 / total_neg as f64);
        thresholds.push(score);
    }

    (
        Array1::from(fpr),
        Array1::from(tpr),
        Array1::from(thresholds),
    )
}

/// Computes the precision-recall curve
///
/// Returns `(precision, recall, thresholds)` at each distinct score threshold (in decreasing
/// order), with a final `(precision = 1, recall = 0)` point appended that has no threshold. Hence
/// `precision` and `recall` are one element longer than `thresholds`
///
/// # Parameters
///
/// - `labels` - True binary labels (`true` for positive)
/// - `scores` - Predicted scores or probabilities
///
/// # Returns
///
/// - `(Array1<f64>, Array1<f64>, Array1<f64>)` - `(precision, recall, thresholds)`, where
///   `precision.len() == recall.len() == thresholds.len() + 1`
///
/// # Panics
///
/// - Panics if `labels` and `scores` have different lengths
/// - Panics if the inputs are empty
/// - Panics if `scores` contains `NaN`
/// - Panics if there are no positive labels
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::metrics::precision_recall_curve;
///
/// let labels = array![true, false, true, false];
/// let scores = array![0.9, 0.6, 0.4, 0.1];
/// let (precision, recall, thresholds) = precision_recall_curve(&labels, &scores);
/// assert_eq!(precision.len(), recall.len());
/// assert_eq!(thresholds.len(), precision.len() - 1);
/// assert_eq!(precision[precision.len() - 1], 1.0);
/// assert_eq!(recall[recall.len() - 1], 0.0);
/// ```
pub fn precision_recall_curve<S1, S2>(
    labels: &ArrayBase<S1, Ix1>,
    scores: &ArrayBase<S2, Ix1>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>)
where
    S1: Data<Elem = bool>,
    S2: Data<Elem = f64>,
{
    let (points, total_pos, _) = ranked_cumulative(labels, scores);
    if total_pos == 0 {
        panic!("invalid input: precision-recall curve requires at least one positive label");
    }

    let mut precision = Vec::with_capacity(points.len() + 1);
    let mut recall = Vec::with_capacity(points.len() + 1);
    let mut thresholds = Vec::with_capacity(points.len());

    for &(score, tp, fp) in &points {
        precision.push(tp as f64 / (tp + fp) as f64);
        recall.push(tp as f64 / total_pos as f64);
        thresholds.push(score);
    }
    // Closing point of the curve (recall 0, precision 1) has no associated threshold
    precision.push(1.0);
    recall.push(0.0);

    (
        Array1::from(precision),
        Array1::from(recall),
        Array1::from(thresholds),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // MulticlassConfusionMatrix::aggregate empty-support guards: `new()` rejects empty input, so
    // the internal state is built directly to reach the defensive branches in `aggregate`

    /// Helper: a confusion matrix with no classes and a 0x0 count matrix
    fn empty_cm() -> MulticlassConfusionMatrix {
        MulticlassConfusionMatrix {
            matrix: Array2::<usize>::zeros((0, 0)),
            labels: Vec::new(),
        }
    }

    /// Macro averaging over an empty per-class slice returns 0.0
    #[test]
    fn test_aggregate_macro_empty_per_class_is_zero() {
        let cm = empty_cm();
        assert_eq!(cm.aggregate(&[], Average::Macro), 0.0);
    }

    /// Weighted averaging with zero total support returns 0.0
    #[test]
    fn test_aggregate_weighted_zero_support_is_zero() {
        let cm = empty_cm();
        assert_eq!(cm.aggregate(&[], Average::Weighted), 0.0);
    }
}
