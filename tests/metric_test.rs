#![cfg(feature = "metric")]

use approx::assert_abs_diff_eq;
use ndarray::{Array1, array};
use rustyml::metric::*;

fn assert_float_eq(a: f64, b: f64) {
    assert!((a - b).abs() < f64::EPSILON, "Expected {}, got {}", b, a);
}

// ===================== Regression metrics =====================

#[test]
fn test_root_mean_squared_error() {
    // Identical inputs: zero error
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_abs_diff_eq!(root_mean_squared_error(&y_true.view(), &y_pred.view()), 0.0);

    // Constant error of 1.0
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![2.0, 3.0, 4.0, 5.0, 6.0];
    assert_abs_diff_eq!(root_mean_squared_error(&y_true.view(), &y_pred.view()), 1.0);

    // Constant error of 0.5
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.5, 2.5, 3.5, 4.5, 5.5];
    assert_abs_diff_eq!(root_mean_squared_error(&y_true.view(), &y_pred.view()), 0.5);

    // Negative values
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![-1.0, -2.0, -3.0];
    assert_abs_diff_eq!(
        root_mean_squared_error(&y_true.view(), &y_pred.view()),
        4.320493798938574
    );
}

#[test]
#[should_panic]
fn test_rmse_empty_arrays() {
    let empty: Array1<f64> = array![];
    root_mean_squared_error(&empty.view(), &empty.view());
}

#[test]
#[should_panic]
fn test_rmse_mismatched_lengths() {
    let y_true = array![1.0, 2.0];
    let y_pred = array![1.0, 2.0, 3.0];
    root_mean_squared_error(&y_true.view(), &y_pred.view());
}

#[test]
fn test_mean_squared_error() {
    let y_true = array![3.0, -0.5, 2.0, 7.0];
    let y_pred = array![2.5, 0.0, 2.1, 7.8];
    assert_abs_diff_eq!(
        mean_squared_error(&y_true.view(), &y_pred.view()),
        0.2875,
        epsilon = 1e-10
    );
}

#[test]
#[should_panic]
fn test_mse_empty_arrays() {
    let empty: Array1<f64> = array![];
    mean_squared_error(&empty.view(), &empty.view());
}

#[test]
fn test_mean_absolute_error() {
    // Zero error
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true.view(), &y_pred.view()), 0.0);

    // Constant error
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![2.0, 3.0, 4.0, 5.0, 6.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true.view(), &y_pred.view()), 1.0);

    // Mixed positive and negative errors
    let y_true = array![2.0, 1.0, 3.0];
    let y_pred = array![1.0, 3.0, 2.0];
    assert_abs_diff_eq!(
        mean_absolute_error(&y_true.view(), &y_pred.view()),
        1.3333333333333333,
        epsilon = 1e-10
    );

    // Negative values
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![-1.0, -2.0, -3.0];
    assert_abs_diff_eq!(mean_absolute_error(&y_true.view(), &y_pred.view()), 4.0);
}

#[test]
#[should_panic]
fn test_mae_empty_arrays() {
    let empty: Array1<f64> = array![];
    mean_absolute_error(&empty.view(), &empty.view());
}

#[test]
#[should_panic]
fn test_mae_mismatched_lengths() {
    let y_true = array![1.0, 2.0];
    let y_pred = array![1.0, 2.0, 3.0];
    mean_absolute_error(&y_true.view(), &y_pred.view());
}

#[test]
fn test_r2_score() {
    // Close-to-perfect predictions
    let y_true = array![3.0, 2.0, 5.0, 7.0, 9.0];
    let y_pred = array![2.8, 1.9, 5.2, 7.5, 8.9];
    assert!(r2_score(&y_true.view(), &y_pred.view()) > 0.95);

    // Perfect prediction
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 3.0];
    assert_abs_diff_eq!(r2_score(&y_true.view(), &y_pred.view()), 1.0);

    // Predicting the mean gives R² = 0
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0]; // mean is 3
    let y_pred = array![3.0, 3.0, 3.0, 3.0, 3.0];
    assert_abs_diff_eq!(r2_score(&y_true.view(), &y_pred.view()), 0.0);

    // Zero-variance ground truth with imperfect predictions gives R² = 0
    let y_true = array![7.0, 7.0, 7.0];
    let y_pred = array![6.0, 7.0, 8.0];
    assert_abs_diff_eq!(r2_score(&y_true.view(), &y_pred.view()), 0.0);

    // Zero-variance ground truth with a perfect fit gives R² = 1
    let y_true = array![7.0, 7.0, 7.0];
    let y_pred = array![7.0, 7.0, 7.0];
    assert_abs_diff_eq!(r2_score(&y_true.view(), &y_pred.view()), 1.0);
}

#[test]
#[should_panic]
fn test_r2_empty_arrays() {
    let empty: Array1<f64> = array![];
    r2_score(&empty.view(), &empty.view());
}

// ===================== Confusion matrix =====================

#[test]
fn test_confusion_matrix_new() {
    // Perfect classification
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_eq!(cm.get_counts(), (2, 0, 2, 0));

    // One of each outcome
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_eq!(cm.get_counts(), (1, 1, 1, 1));

    // Probability inputs thresholded at 0.5
    let y_true = array![0.9, 0.1, 0.2, 0.8];
    let y_pred = array![0.6, 0.4, 0.7, 0.3];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_eq!(cm.get_counts(), (1, 1, 1, 1));
}

#[test]
#[should_panic]
fn test_confusion_matrix_new_error() {
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0];
    let _result = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
}

#[test]
fn test_get_counts() {
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_eq!(cm.get_counts(), (1, 1, 1, 1));
}

#[test]
fn test_confusion_matrix_accuracy() {
    // Perfect classification
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.accuracy(), 1.0);

    // 50% correct
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.accuracy(), 0.5);

    // All incorrect
    let y_true = array![0.0, 0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.accuracy(), 0.0);
}

#[test]
fn test_error_rate() {
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.error_rate(), 0.0);

    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.error_rate(), 0.5);

    let y_true = array![0.0, 0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.error_rate(), 1.0);
}

#[test]
fn test_precision() {
    // Perfect precision
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.precision(), 1.0);

    // Partial precision: 2 TP, 1 FP
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.precision(), 2.0 / 3.0);

    // No true positives
    let y_true = array![0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.precision(), 0.0);
}

#[test]
fn test_recall() {
    // Perfect recall
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.recall(), 1.0);

    // Partial recall: 1 TP, 1 FN
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.recall(), 0.5);

    // No actual positives: recall defined as 1.0
    let y_true = array![0.0, 0.0, 0.0];
    let y_pred = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.recall(), 1.0);
}

#[test]
fn test_specificity() {
    // Perfect specificity
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.specificity(), 1.0);

    // Partial specificity: 1 TN, 1 FP
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.specificity(), 0.5);

    // No actual negatives: specificity defined as 1.0
    let y_true = array![1.0, 1.0, 1.0];
    let y_pred = array![1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.specificity(), 1.0);
}

#[test]
fn test_f1_score() {
    // Perfect F1 score
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.f1_score(), 1.0);

    // precision = 1/2, recall = 1/2, F1 = 1/2
    let y_true = array![1.0, 0.0, 0.0, 1.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.f1_score(), 0.5);

    // Precision and recall both 0
    let y_true = array![0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    assert_float_eq(cm.f1_score(), 0.0);
}

#[test]
fn test_summary() {
    let y_true = array![1.0, 0.0, 0.0, 1.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
    let summary = cm.summary();

    for needle in [
        "TP:",
        "FP:",
        "TN:",
        "FN:",
        "Accuracy:",
        "Error Rate:",
        "Precision:",
        "Recall:",
        "Specificity:",
        "F1 Score:",
    ] {
        assert!(summary.contains(needle), "summary missing {needle}");
    }
}

// ===================== Accuracy =====================

#[test]
fn test_accuracy() {
    // Perfect prediction
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(accuracy(&y_true.view(), &y_pred.view()), 1.0);

    // Some errors
    let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];
    assert_eq!(accuracy(&y_true.view(), &y_pred.view()), 0.6);

    // Multi-class
    let y_true = array![0.0, 1.0, 2.0, 2.0, 1.0, 1.0];
    let y_pred = array![0.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    assert_eq!(accuracy(&y_true.view(), &y_pred.view()), 2.0 / 3.0);

    // Small floating-point differences are not exact matches
    let y_true = array![0.0, 1.0];
    let y_pred = array![0.0000001, 1.0];
    assert!(accuracy(&y_true.view(), &y_pred.view()) < 1.0);

    // All predictions wrong
    let y_true = array![0.0, 0.0, 1.0, 1.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    assert_eq!(accuracy(&y_true.view(), &y_pred.view()), 0.0);
}

#[test]
#[should_panic]
fn test_accuracy_with_different_length_arrays() {
    let y_true = array![1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0];
    accuracy(&y_true.view(), &y_pred.view());
}

// ===================== Mutual information =====================

// Identical clusterings score 1.0 for both NMI and AMI.
#[test]
fn test_identical_clusterings() {
    let labels = array![0, 1, 1, 0, 2, 2];
    let nmi = normalized_mutual_info(&labels.view(), &labels.view());
    let ami = adjusted_mutual_info(&labels.view(), &labels.view());
    assert!(
        (nmi - 1.0).abs() < 1e-6,
        "NMI of identical clusterings should be 1.0, got {}",
        nmi
    );
    assert!(
        (ami - 1.0).abs() < 1e-6,
        "AMI of identical clusterings should be 1.0, got {}",
        ami
    );
}

// Different clusterings from the documented example.
#[test]
fn test_different_clusterings() {
    let labels_true = array![0, 0, 1, 1, 2, 2];
    let labels_pred = array![0, 0, 1, 2, 1, 2];

    let nmi = normalized_mutual_info(&labels_true.view(), &labels_pred.view());
    let expected_nmi = 0.578;
    assert!(
        (nmi - expected_nmi).abs() < 0.05,
        "Expected NMI approx {} but got {}",
        expected_nmi,
        nmi
    );

    let ami = adjusted_mutual_info(&labels_true.view(), &labels_pred.view());
    assert!(
        ami < 1.0,
        "AMI should be less than 1.0 for different clusterings, got {}",
        ami
    );
    assert!(ami >= 0.0, "AMI should be non-negative, got {}", ami);
}

#[test]
#[should_panic]
fn test_length_mismatch() {
    let labels_true = array![0, 1, 1];
    let labels_pred = array![0, 1];
    normalized_mutual_info(&labels_true.view(), &labels_pred.view());
}

// All samples in a single cluster: NMI is 0.0 (zero entropy), AMI is 1.0 (degenerate normaliser).
#[test]
fn test_constant_labels() {
    let labels_true = array![0, 0, 0, 0];
    let labels_pred = array![1, 1, 1, 1];

    let nmi = normalized_mutual_info(&labels_true.view(), &labels_pred.view());
    assert_eq!(nmi, 0.0, "NMI for constant labels should be 0.0");

    let ami = adjusted_mutual_info(&labels_true.view(), &labels_pred.view());
    assert_eq!(ami, 1.0, "AMI for constant labels should be 1.0");
}

// ===================== Adjusted Rand Index =====================

#[test]
fn test_adjusted_rand_index_identical() {
    let labels = array![0, 1, 1, 0, 2, 2];
    let ari = adjusted_rand_index(&labels.view(), &labels.view());
    assert!(
        (ari - 1.0).abs() < 1e-9,
        "ARI of identical clusterings should be 1.0, got {}",
        ari
    );
}

#[test]
fn test_adjusted_rand_index_example() {
    let labels_true = array![0, 0, 1, 1, 2, 2];
    let labels_pred = array![0, 0, 1, 2, 1, 2];
    let ari = adjusted_rand_index(&labels_true.view(), &labels_pred.view());
    // Contingency has a single pair agreement; ARI works out to exactly 1/6.
    assert!(
        (ari - 1.0 / 6.0).abs() < 1e-9,
        "Expected ARI 1/6, got {}",
        ari
    );
}

#[test]
fn test_adjusted_rand_index_constant_labels() {
    // Both clusterings trivial: ARI is defined as 1.0.
    let labels_true = array![0, 0, 0, 0];
    let labels_pred = array![1, 1, 1, 1];
    let ari = adjusted_rand_index(&labels_true.view(), &labels_pred.view());
    assert_eq!(ari, 1.0);
}

#[test]
#[should_panic]
fn test_adjusted_rand_index_length_mismatch() {
    let labels_true = array![0, 1, 1];
    let labels_pred = array![0, 1];
    adjusted_rand_index(&labels_true.view(), &labels_pred.view());
}

// ===================== Silhouette score =====================

#[test]
fn test_silhouette_score_well_separated() {
    let x = array![[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]];
    let labels = array![0, 0, 1, 1];
    let score = silhouette_score(&x.view(), &labels.view());
    assert!(
        score > 0.8,
        "Well-separated clusters should score near 1.0, got {}",
        score
    );
}

#[test]
fn test_silhouette_score_overlapping() {
    // Interleaved points labelled into two clusters score poorly (negative).
    let x = array![[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [11.0, 0.0]];
    let labels = array![0, 0, 1, 1];
    let score = silhouette_score(&x.view(), &labels.view());
    assert!(
        score < 0.0,
        "Mislabelled clusters should score below 0, got {}",
        score
    );
}

#[test]
#[should_panic]
fn test_silhouette_score_single_cluster() {
    // Only one distinct cluster: outside the valid 2..=n-1 range.
    let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let labels = array![0, 0, 0];
    silhouette_score(&x.view(), &labels.view());
}

#[test]
#[should_panic]
fn test_silhouette_score_length_mismatch() {
    let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let labels = array![0, 1];
    silhouette_score(&x.view(), &labels.view());
}

// ===================== ROC AUC =====================

#[test]
fn test_roc_auc_valid() {
    let labels = array![false, true, true, false];
    let scores = array![0.1, 0.4, 0.35, 0.8];
    // Sorted: (0.1, F), (0.35, T), (0.4, T), (0.8, F). Positive ranks 2 + 3 = 5.
    // U = 5 - (2*3/2) = 2, with 2 negatives → AUC = 2 / (2*2) = 0.5.
    let auc = roc_auc(&labels.view(), &scores.view());
    assert!((auc - 0.5).abs() < 1e-12);
}

#[test]
#[should_panic]
fn test_roc_auc_empty_input() {
    let labels: Array1<bool> = array![];
    let scores: Array1<f64> = array![];
    let _result = roc_auc(&labels.view(), &scores.view());
}

#[test]
#[should_panic]
fn test_roc_auc_length_mismatch() {
    let labels = array![true];
    let scores = array![0.1, 0.2];
    let _result = roc_auc(&labels.view(), &scores.view());
}

#[test]
#[should_panic]
fn test_roc_auc_no_positive() {
    let labels = array![false, false, false];
    let scores = array![0.2, 0.3, 0.4];
    let _result = roc_auc(&labels.view(), &scores.view());
}

#[test]
#[should_panic]
fn test_roc_auc_no_negative() {
    let labels = array![true, true, true];
    let scores = array![0.2, 0.3, 0.4];
    let _result = roc_auc(&labels.view(), &scores.view());
}

#[test]
fn test_roc_auc_with_ties() {
    let labels = array![true, false, true, false];
    let scores = array![0.5, 0.5, 0.5, 0.5];
    // All tied: average rank 2.5. Positive rank sum = 5.0.
    // U = 5 - (2*3/2) = 2, 2 negatives → AUC = 2 / (2*2) = 0.5.
    let auc = roc_auc(&labels.view(), &scores.view());
    assert!((auc - 0.5).abs() < 1e-12);
}
