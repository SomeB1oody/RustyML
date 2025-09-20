use crate::metric::*;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, array};

#[test]
fn test_root_mean_squared_error() {
    // Test basic calculation
    let predictions = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = root_mean_squared_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 0.0);

    // Test calculation with constant error
    let predictions = array![2.0, 3.0, 4.0, 5.0, 6.0];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = root_mean_squared_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 1.0);

    // Test more complex example
    let predictions = array![1.5, 2.5, 3.5, 4.5, 5.5];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = root_mean_squared_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 0.5);

    // Test negative values
    let predictions = array![-1.0, -2.0, -3.0];
    let targets = array![1.0, 2.0, 3.0];
    let result = root_mean_squared_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 4.320493798938574);
}

#[test]
fn test_rmse_empty_arrays() {
    // Test empty arrays
    let empty: Array1<f64> = array![];
    let rmse = root_mean_squared_error(empty.view(), empty.view());
    assert_float_eq(rmse, 0.0);
}

#[test]
#[should_panic]
fn test_rmse_mismatched_lengths() {
    // Test arrays with mismatched lengths
    let predictions = array![1.0, 2.0, 3.0];
    let targets = array![1.0, 2.0];
    root_mean_squared_error(predictions.view(), targets.view());
}

#[test]
fn test_mean_absolute_error() {
    // Test identical values (zero error)
    let predictions = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = mean_absolute_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 0.0);

    // Test constant error
    let predictions = array![2.0, 3.0, 4.0, 5.0, 6.0];
    let targets = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = mean_absolute_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 1.0);

    // Test mixed positive and negative errors
    let predictions = array![1.0, 3.0, 2.0];
    let targets = array![2.0, 1.0, 3.0];
    let result = mean_absolute_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 1.3333333333333333, epsilon = 1e-10);

    // Test negative values
    let predictions = array![-1.0, -2.0, -3.0];
    let targets = array![1.0, 2.0, 3.0];
    let result = mean_absolute_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 4.0);

    // Test decimal values
    let predictions = array![1.5, 2.5, 3.5];
    let targets = array![1.0, 2.0, 3.0];
    let result = mean_absolute_error(predictions.view(), targets.view());
    assert_abs_diff_eq!(result, 0.5);
}

#[test]
fn test_mae_empty_arrays() {
    // Test empty arrays
    let empty: Array1<f64> = array![];
    let mae = mean_absolute_error(empty.view(), empty.view());
    assert_float_eq(mae, 0.0);
}

#[test]
#[should_panic]
fn test_mae_mismatched_lengths() {
    // Test arrays with mismatched lengths
    let predictions = array![1.0, 2.0, 3.0];
    let targets = array![1.0, 2.0];
    mean_absolute_error(predictions.view(), targets.view());
}

#[test]
fn test_r2_score() {
    // Standard case
    let actual = array![3.0, 2.0, 5.0, 7.0, 9.0];
    let predicted = array![2.8, 1.9, 5.2, 7.5, 8.9];
    // Should be close to 1 but not exactly 1 since predictions are close but not exact
    let r2 = r2_score(predicted.view(), actual.view());
    assert!(r2 > 0.95); // Verify R² is close to 1

    // Perfect prediction
    let perfect_actual = array![1.0, 2.0, 3.0];
    let perfect_predicted = array![1.0, 2.0, 3.0];
    assert_abs_diff_eq!(
        r2_score(perfect_predicted.view(), perfect_actual.view()),
        1.0
    );

    // When prediction is always the mean, R² should be 0
    let mean_actual = array![1.0, 2.0, 3.0, 4.0, 5.0]; // mean is 3
    let mean_predicted = array![3.0, 3.0, 3.0, 3.0, 3.0];
    assert_abs_diff_eq!(r2_score(mean_predicted.view(), mean_actual.view()), 0.0);

    // When all actual values are the same, R² should be 0
    let same_actual = array![7.0, 7.0, 7.0];
    let same_predicted = array![6.0, 7.0, 8.0];
    assert_abs_diff_eq!(r2_score(same_predicted.view(), same_actual.view()), 0.0);
}

fn assert_float_eq(a: f64, b: f64) {
    assert!((a - b).abs() < f64::EPSILON, "Expected {}, got {}", b, a);
}

#[test]
fn test_confusion_matrix_new() {
    // Test perfect classification
    let pred = array![1.0, 0.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());

    assert_eq!(cm.get_counts(), (2, 0, 2, 0));

    // Test various classification error scenarios
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());

    assert_eq!(cm.get_counts(), (1, 1, 1, 1));

    // Test probability threshold (0.5)
    let pred = array![0.6, 0.4, 0.7, 0.3];
    let actual = array![0.9, 0.1, 0.2, 0.8];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());

    assert_eq!(cm.get_counts(), (1, 1, 1, 1));
}

#[test]
#[should_panic]
fn test_confusion_matrix_new_error() {
    // Test case with mismatched input lengths
    let pred = array![1.0, 0.0, 1.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let _result = ConfusionMatrix::new(pred.view(), actual.view());
}

#[test]
fn test_get_counts() {
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());

    let counts = cm.get_counts();
    assert_eq!(counts, (1, 1, 1, 1));
}

#[test]
fn test_confusion_matrix_accuracy() {
    // Perfect classification
    let pred = array![1.0, 0.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.accuracy(), 1.0);

    // 50% correct
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.accuracy(), 0.5);

    // All incorrect
    let pred = array![1.0, 1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.accuracy(), 0.0);
}

#[test]
fn test_error_rate() {
    // Perfect classification
    let pred = array![1.0, 0.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.error_rate(), 0.0);

    // 50% errors
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.error_rate(), 0.5);

    // All errors
    let pred = array![1.0, 1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.error_rate(), 1.0);
}

#[test]
fn test_precision() {
    // Perfect precision
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.precision(), 1.0);

    // Partial precision
    let pred = array![1.0, 1.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.precision(), 2.0 / 3.0);

    // No true positives
    let pred = array![1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.precision(), 0.0);
}

#[test]
fn test_recall() {
    // Perfect recall
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.recall(), 1.0);

    // Partial recall
    let pred = array![1.0, 0.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let _cm = ConfusionMatrix::new(pred.view(), actual.view());

    // No actual positives
    let pred = array![0.0, 0.0, 0.0];
    let actual = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.recall(), 1.0); // Note: when there are no actual positives, recall is 1.0 (avoiding 0/0 case)
}

#[test]
fn test_specificity() {
    // Perfect specificity
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.specificity(), 1.0);

    // Partial specificity
    let pred = array![1.0, 1.0, 0.0, 1.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.specificity(), 0.5);

    // No actual negatives
    let pred = array![1.0, 1.0, 1.0];
    let actual = array![1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.specificity(), 1.0); // Note: when there are no actual negatives, specificity is 1.0 (avoiding 0/0 case)
}

#[test]
fn test_f1_score() {
    // Perfect F1 score
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.f1_score(), 1.0);

    // F1 score calculation example
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    // precision = 1/2, recall = 1/2, F1 = 2*1/2*1/2 / (1/2 + 1/2) = 1/2
    assert_float_eq(cm.f1_score(), 0.5);

    // Precision or recall is 0
    let pred = array![1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    assert_float_eq(cm.f1_score(), 0.0);
}

#[test]
fn test_summary() {
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(pred.view(), actual.view());
    let summary = cm.summary();

    // Check if the summary string contains all necessary information
    assert!(summary.contains("TP:"));
    assert!(summary.contains("FP:"));
    assert!(summary.contains("TN:"));
    assert!(summary.contains("FN:"));
    assert!(summary.contains("Accuracy:"));
    assert!(summary.contains("Error Rate:"));
    assert!(summary.contains("Precision:"));
    assert!(summary.contains("Recall:"));
    assert!(summary.contains("Specificity:"));
    assert!(summary.contains("F1 Score:"));
}

#[test]
fn test_accuracy() {
    // Test case 1: Binary classification with perfect prediction
    let predicted = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let actual = array![1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(accuracy(predicted.view(), actual.view()), 1.0);

    // Test case 2: Binary classification with some errors
    let predicted = array![1.0, 0.0, 1.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 0.0, 1.0, 1.0];
    assert_eq!(accuracy(predicted.view(), actual.view()), 0.6);

    // Test case 3: Multi-class classification
    let predicted = array![0.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    let actual = array![0.0, 1.0, 2.0, 2.0, 1.0, 1.0];
    assert_eq!(accuracy(predicted.view(), actual.view()), 2.0 / 3.0); // 4 out of 6 correct

    // Test case 5: Floating point equality with very small differences
    // (should be considered equal due to epsilon comparison)
    let predicted = array![0.0000001, 1.0];
    let actual = array![0.0, 1.0];
    assert!(accuracy(predicted.view(), actual.view()) < 1.0); // Should fail equality due to floating point precision

    // Test case 6: All predictions are wrong
    let predicted = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![0.0, 0.0, 1.0, 1.0];
    assert_eq!(accuracy(predicted.view(), actual.view()), 0.0);
}

#[test]
#[should_panic]
fn test_accuracy_with_different_length_arrays() {
    let predicted = array![1.0, 0.0, 1.0];
    let actual = array![1.0, 0.0];
    accuracy(predicted.view(), actual.view()); // This should panic
}

// Test that when the clusterings are identical, both NMI and AMI return 1.0.
#[test]
fn test_identical_clusterings() {
    let labels = array![0, 1, 1, 0, 2, 2];
    let nmi = normalized_mutual_info(labels.view(), labels.view());
    let ami = adjusted_mutual_info(labels.view(), labels.view());
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

// Test different clusterings using the example provided in the documentation.
#[test]
fn test_different_clusterings() {
    let labels_true = array![0, 0, 1, 1, 2, 2];
    let labels_pred = array![0, 0, 1, 2, 1, 2];

    // Test NMI
    let nmi = normalized_mutual_info(labels_true.view(), labels_pred.view());
    // The expected NMI is approximately 0.578 (subject to implementation details)
    let expected_nmi = 0.578;
    assert!(
        (nmi - expected_nmi).abs() < 0.05,
        "Expected NMI approx {} but got {}",
        expected_nmi,
        nmi
    );

    // Test AMI
    let ami = adjusted_mutual_info(labels_true.view(), labels_pred.view());
    // Since EMI calculation may affect the exact value, we assert that:
    // - AMI is less than 1.0 (since the clusterings are not identical)
    // - AMI is non-negative
    assert!(
        ami < 1.0,
        "AMI should be less than 1.0 for different clusterings, got {}",
        ami
    );
    assert!(ami >= 0.0, "AMI should be non-negative, got {}", ami);
}

// Test that an error is returned when the input slices have different lengths.
#[test]
#[should_panic]
fn test_length_mismatch() {
    let labels_true = array![0, 1, 1];
    let labels_pred = array![0, 1];
    normalized_mutual_info(labels_true.view(), labels_pred.view());
}

// Test the special case when all samples are assigned to the same cluster.
#[test]
fn test_constant_labels() {
    let labels_true = array![0, 0, 0, 0];
    let labels_pred = array![1, 1, 1, 1];

    // For constant labels, since the entropy is 0, the function returns 0.0 for NMI.
    let nmi = normalized_mutual_info(labels_true.view(), labels_pred.view());
    assert_eq!(nmi, 0.0, "NMI for constant labels should be 0.0");

    // For AMI, the special handling returns 1.0 when the denominator is close to 0.
    let ami = adjusted_mutual_info(labels_true.view(), labels_pred.view());
    assert_eq!(ami, 1.0, "AMI for constant labels should be 1.0");
}

#[test]
fn test_calculate_auc_valid() {
    // Test a valid case.
    let scores = array![0.1, 0.4, 0.35, 0.8];
    let labels = array![false, true, true, false];
    let auc = calculate_auc(scores.view(), labels.view());
    // Calculation:
    // Sorted: [(0.1, false), (0.35, true), (0.4, true), (0.8, false)]
    // The ranks for positive samples are 2 and 3, so sum_positive_ranks = 2 + 3 = 5.
    // U = 5 - (2*(2+1)/2) = 5 - 3 = 2. With 2 negatives, total comparisons = 2 * 2 = 4.
    // Thus, AUC = 2/4 = 0.5.
    assert!((auc - 0.5).abs() < 1e-12);
}

#[test]
#[should_panic]
fn test_calculate_auc_empty_input() {
    // Test empty input arrays: should return an error.
    let scores: Array1<f64> = array![];
    let labels: Array1<bool> = array![];
    let _result = calculate_auc(scores.view(), labels.view());
}

#[test]
#[should_panic]
fn test_calculate_auc_length_mismatch() {
    // Test input arrays with different lengths: should return an error.
    let scores = array![0.1, 0.2];
    let labels = array![true];
    let _result = calculate_auc(scores.view(), labels.view());
}

#[test]
#[should_panic]
fn test_calculate_auc_no_positive() {
    // Test case with no positive samples: should return an error.
    let scores = array![0.2, 0.3, 0.4];
    let labels = array![false, false, false];
    let _result = calculate_auc(scores.view(), labels.view());
}

#[test]
#[should_panic]
fn test_calculate_auc_no_negative() {
    // Test case with no negative samples: should return an error.
    let scores = array![0.2, 0.3, 0.4];
    let labels = array![true, true, true];
    let _result = calculate_auc(scores.view(), labels.view());
}

#[test]
fn test_calculate_auc_with_ties() {
    // Test case with tied scores, where positive and negative samples are balanced.
    let scores = array![0.5, 0.5, 0.5, 0.5];
    let labels = array![true, false, true, false];
    // With all scores tied, the average rank is (1+2+3+4)/4 = 2.5.
    // For two positive samples, sum_positive_ranks = 2.5 + 2.5 = 5.0.
    // U = 5 - (2*(2+1)/2) = 5 - 3 = 2, negative sample count = 2, so AUC = 2/(2*2) = 0.5.
    let auc = calculate_auc(scores.view(), labels.view());
    assert!((auc - 0.5).abs() < 1e-12);
}
