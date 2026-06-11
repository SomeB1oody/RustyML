//! Integration tests for `metrics::classification`: confusion matrices,
//! accuracy, ROC/PR curves, log loss, Cohen's kappa, and top-k accuracy

use approx::assert_abs_diff_eq;
use ndarray::{arr2, array};
use rustyml::metrics::{
    Average, ConfusionMatrix, MulticlassConfusionMatrix, accuracy, average_precision, cohen_kappa,
    log_loss, precision_recall_curve, roc_auc, roc_curve, top_k_accuracy,
};

// ConfusionMatrix::new - TP/FP/TN/FN counting

#[test]
fn cm_new_perfect_predictions() {
    // y_true=[1,0,1,0], y_pred=[1,0,1,0] -> TP=2, FP=0, TN=2, FN=0
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (2, 0, 2, 0));
}

#[test]
fn cm_new_all_wrong() {
    // y_true=[1,0,1,0], y_pred=[0,1,0,1] -> TP=0, FP=2, TN=0, FN=2
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![0.0, 1.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (0, 2, 0, 2));
}

#[test]
fn cm_new_mixed() {
    // y_true=[1,1,1,1,1,0,0,0], y_pred=[1,1,1,0,0,0,0,1] -> TP=3, FN=2, TN=2, FP=1
    let y_true = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (3, 1, 2, 2));
}

#[test]
fn cm_new_thresholding_at_0_5() {
    // Probabilities >= 0.5 are positive; y_true=[0.9,0.1,0.6,0.4], y_pred=[0.8,0.2,0.3,0.7]
    // gives TP=1, FP=1, TN=1, FN=1
    let y_true = array![0.9, 0.1, 0.6, 0.4];
    let y_pred = array![0.8, 0.2, 0.3, 0.7];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (1, 1, 1, 1));
}

// ConfusionMatrix derived metrics (tp=3, fp=1, tn=2, fn=2)
// from y_true=[1,1,1,1,1,0,0,0], y_pred=[1,1,1,0,0,0,0,1]

fn cm_3_1_2_2() -> ConfusionMatrix {
    let y_true = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    ConfusionMatrix::new(&y_true, &y_pred)
}

#[test]
fn cm_accuracy_partial() {
    // accuracy = (TP+TN)/(TP+FP+TN+FN) = 5/8 = 0.625
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.accuracy(), 0.625, epsilon = 1e-12);
}

#[test]
fn cm_accuracy_range() {
    // accuracy must be in [0, 1]
    let cm = cm_3_1_2_2();
    let acc = cm.accuracy();
    assert!((0.0..=1.0).contains(&acc));
}

#[test]
fn cm_accuracy_plus_error_rate_equals_one() {
    // error_rate = 1 - accuracy always
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.accuracy() + cm.error_rate(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_accuracy_perfect() {
    // tp=2, fp=0, tn=2, fn=0 -> accuracy = 1.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.accuracy(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_accuracy_all_wrong() {
    // tp=0, fp=2, tn=0, fn=2 -> accuracy = 0.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![0.0, 0.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.accuracy(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_precision_partial() {
    // precision = TP/(TP+FP) = 3/4 = 0.75
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.precision(), 0.75, epsilon = 1e-12);
}

#[test]
fn cm_precision_perfect() {
    // tp=2, fp=0 -> precision = 1.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.precision(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_precision_no_positive_predictions() {
    // tp=0, fp=0 (all predicted negative) -> precision = 0.0 (convention)
    let y_true = array![1.0, 0.0];
    let y_pred = array![0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    // TP=0, FP=0, TN=1, FN=1
    assert_eq!(cm.get_counts(), (0, 0, 1, 1));
    assert_abs_diff_eq!(cm.precision(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_recall_partial() {
    // recall = TP/(TP+FN) = 3/5 = 0.6
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.recall(), 0.6, epsilon = 1e-12);
}

#[test]
fn cm_recall_no_actual_positives() {
    // tp=0, fn=0 (all actual negatives) -> recall = 0.0 (sklearn zero_division=0 convention,
    // consistent with MulticlassConfusionMatrix::per_class_recall)
    let y_true = array![0.0, 0.0];
    let y_pred = array![0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    // TN=1, FP=1, TP=0, FN=0
    assert_eq!(cm.get_counts(), (0, 1, 1, 0));
    assert_abs_diff_eq!(cm.recall(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_specificity_partial() {
    // specificity = TN/(TN+FP) = 2/3
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.specificity(), 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn cm_specificity_no_actual_negatives() {
    // tn=0, fp=0 (all actual positives) -> specificity = 1.0 (convention)
    let y_true = array![1.0, 1.0];
    let y_pred = array![1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    // TP=1, FN=1, FP=0, TN=0
    assert_eq!(cm.get_counts(), (1, 0, 0, 1));
    assert_abs_diff_eq!(cm.specificity(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_f1_partial() {
    // f1 = 2*prec*rec/(prec+rec) = 2*(3/4)*(3/5)/((3/4)+(3/5)) = 2/3
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.f1_score(), 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn cm_f1_perfect() {
    // precision=1.0, recall=1.0 -> f1 = 1.0
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.f1_score(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_f1_zero_precision_and_recall() {
    // TP=0, FP=0, FN=1, TN=1: precision=0.0 (guard), recall=0.0 -> f1 = 0.0 (guard)
    let y_true = array![1.0, 0.0];
    let y_pred = array![0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.f1_score(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_mcc_partial() {
    // mcc = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    //     = 4 / sqrt(240) ~= 0.25820
    let cm = cm_3_1_2_2();
    let expected = 4.0 / (240.0_f64).sqrt();
    assert_abs_diff_eq!(cm.mcc(), expected, epsilon = 1e-9);
}

#[test]
fn cm_mcc_perfect() {
    // TP=2, FP=0, TN=2, FN=0 -> (4-0)/sqrt(2*2*2*2) = 1.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.mcc(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_mcc_degenerate_all_predicted_positive() {
    // TP=2, FP=2, TN=0, FN=0 -> denom = sqrt(4*2*2*0) = 0 -> mcc = 0.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.mcc(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_mcc_degenerate_all_predicted_negative() {
    // TP=0, FP=0, TN=2, FN=2 -> denom = sqrt(0*2*2*4) = 0 -> mcc = 0.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![0.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.mcc(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_balanced_accuracy_partial() {
    // balanced_accuracy = (recall + specificity)/2 = (3/5 + 2/3)/2 = 19/30
    let cm = cm_3_1_2_2();
    let expected = 19.0 / 30.0;
    assert_abs_diff_eq!(cm.balanced_accuracy(), expected, epsilon = 1e-12);
}

#[test]
fn cm_balanced_accuracy_vs_accuracy_imbalanced() {
    // All predictions positive on imbalanced data y_true=[1,1,1,1,0], y_pred=[1,1,1,1,1]:
    // accuracy = 4/5 = 0.8 but balanced_accuracy = (1.0+0.0)/2 = 0.5
    let y_true = array![1.0, 1.0, 1.0, 1.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.accuracy(), 0.8, epsilon = 1e-12);
    assert_abs_diff_eq!(cm.balanced_accuracy(), 0.5, epsilon = 1e-12);
}

#[test]
fn cm_summary_contains_metric_labels() {
    let cm = cm_3_1_2_2();
    let s = cm.summary();
    assert!(s.contains("Accuracy:"));
    assert!(s.contains("Precision:"));
    assert!(s.contains("Recall:"));
    assert!(s.contains("F1 Score:"));
    assert!(s.contains("MCC:"));
    assert!(s.contains("Balanced Accuracy:"));
    assert!(s.contains("TP:"));
    assert!(s.contains("FP:"));
    assert!(s.contains("TN:"));
    assert!(s.contains("FN:"));
}

// ConfusionMatrix panic paths

#[test]
#[should_panic(expected = "dimension mismatch")]
fn cm_new_length_mismatch_panics() {
    let y_true = array![1.0, 0.0, 1.0];
    let y_pred = array![1.0, 0.0];
    let _ = ConfusionMatrix::new(&y_true, &y_pred);
}

#[test]
#[should_panic(expected = "input is empty")]
fn cm_new_empty_input_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = ConfusionMatrix::new(&y_true, &y_pred);
}

// Free function: accuracy

#[test]
fn accuracy_perfect() {
    // All labels match -> accuracy = 1.0
    let y_true = array![0.0, 1.0, 2.0, 3.0];
    let y_pred = array![0.0, 1.0, 2.0, 3.0];
    assert_abs_diff_eq!(accuracy(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

#[test]
fn accuracy_all_wrong() {
    // No labels match -> accuracy = 0.0
    let y_true = array![0.0, 1.0, 2.0];
    let y_pred = array![1.0, 2.0, 0.0];
    assert_abs_diff_eq!(accuracy(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

#[test]
fn accuracy_partial() {
    // y_true=[0,0,1], y_pred=[0,1,1] -> 2 correct out of 3 = 2/3
    let y_true = array![0.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 1.0];
    assert_abs_diff_eq!(accuracy(&y_true, &y_pred), 2.0 / 3.0, epsilon = 1e-9);
}

#[test]
fn accuracy_range_invariant() {
    let y_true = array![0.0, 1.0, 2.0, 1.0, 0.0];
    let y_pred = array![0.0, 2.0, 2.0, 1.0, 1.0];
    let acc = accuracy(&y_true, &y_pred);
    assert!((0.0..=1.0).contains(&acc));
}

#[test]
fn accuracy_symmetry() {
    // accuracy is symmetric in its arguments because the epsilon comparison is symmetric
    let y_true = array![0.0, 1.0, 2.0];
    let y_pred = array![0.0, 2.0, 2.0];
    assert_abs_diff_eq!(
        accuracy(&y_true, &y_pred),
        accuracy(&y_pred, &y_true),
        epsilon = 1e-12
    );
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn accuracy_length_mismatch_panics() {
    let y_true = array![0.0, 1.0];
    let y_pred = array![0.0, 1.0, 2.0];
    let _ = accuracy(&y_true, &y_pred);
}

#[test]
#[should_panic(expected = "input is empty")]
fn accuracy_empty_panics() {
    let y_true: ndarray::Array1<f64> = array![];
    let y_pred: ndarray::Array1<f64> = array![];
    let _ = accuracy(&y_true, &y_pred);
}

// roc_auc

#[test]
fn roc_auc_perfect() {
    // Perfect ranking (all positives score higher than all negatives) -> AUC = 1.0
    // labels=[T,T,F,F], scores=[0.9,0.8,0.2,0.1]
    let labels = array![true, true, false, false];
    let scores = array![0.9, 0.8, 0.2, 0.1];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 1.0, epsilon = 1e-12);
}

#[test]
fn roc_auc_worst() {
    // Worst ranking (all positives score lower than all negatives) -> AUC = 0.0
    // labels=[T,T,F,F], scores=[0.1,0.2,0.8,0.9]
    let labels = array![true, true, false, false];
    let scores = array![0.1, 0.2, 0.8, 0.9];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.0, epsilon = 1e-12);
}

#[test]
fn roc_auc_random_half() {
    // Random ranking -> AUC = 0.5
    // labels=[F,T,T,F], scores=[0.1,0.35,0.4,0.8]
    let labels = array![false, true, true, false];
    let scores = array![0.1, 0.35, 0.4, 0.8];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.5, epsilon = 1e-12);
}

#[test]
fn roc_auc_with_all_tied_scores() {
    // All scores equal -> every sample gets average rank 2.5 -> AUC = 0.5
    let labels = array![true, true, false, false];
    let scores = array![0.5, 0.5, 0.5, 0.5];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.5, epsilon = 1e-12);
}

#[test]
fn roc_auc_partial_ties() {
    // Two tie groups (scores=[0.8,0.8,0.2,0.2], labels=[T,F,T,F]) with averaged
    // ranks 1.5 and 3.5 per group -> AUC = 0.5
    let labels = array![true, false, true, false];
    let scores = array![0.8, 0.8, 0.2, 0.2];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.5, epsilon = 1e-12);
}

#[test]
fn roc_auc_range_invariant() {
    let labels = array![true, false, true, false, true];
    let scores = array![0.9, 0.6, 0.7, 0.3, 0.4];
    let auc = roc_auc(&labels, &scores);
    assert!((0.0..=1.0).contains(&auc));
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn roc_auc_length_mismatch_panics() {
    let labels = array![true, false];
    let scores = array![0.9];
    let _ = roc_auc(&labels, &scores);
}

#[test]
#[should_panic(expected = "input is empty")]
fn roc_auc_empty_panics() {
    let labels: ndarray::Array1<bool> = array![];
    let scores: ndarray::Array1<f64> = array![];
    let _ = roc_auc(&labels, &scores);
}

#[test]
#[should_panic(expected = "at least one positive and one negative")]
fn roc_auc_no_positive_panics() {
    let labels = array![false, false, false];
    let scores = array![0.9, 0.5, 0.1];
    let _ = roc_auc(&labels, &scores);
}

#[test]
#[should_panic(expected = "at least one positive and one negative")]
fn roc_auc_no_negative_panics() {
    let labels = array![true, true, true];
    let scores = array![0.9, 0.5, 0.1];
    let _ = roc_auc(&labels, &scores);
}

// MulticlassConfusionMatrix

// y_true=[0,1,2,2,1], y_pred=[0,2,2,2,1] -> matrix (row=true, col=pred)
// [[1,0,0],[0,1,1],[0,0,2]]
fn mcm_3class() -> MulticlassConfusionMatrix {
    let y_true = array![0usize, 1, 2, 2, 1];
    let y_pred = array![0usize, 2, 2, 2, 1];
    MulticlassConfusionMatrix::new(&y_true, &y_pred)
}

#[test]
fn mcm_n_classes_and_labels() {
    let cm = mcm_3class();
    assert_eq!(cm.n_classes(), 3);
    assert_eq!(cm.labels(), &[0usize, 1, 2]);
}

#[test]
fn mcm_matrix_cell_values() {
    // Expected matrix [[1,0,0],[0,1,1],[0,0,2]]
    let cm = mcm_3class();
    let m = cm.matrix();
    assert_eq!(m[[0, 0]], 1);
    assert_eq!(m[[0, 1]], 0);
    assert_eq!(m[[0, 2]], 0);
    assert_eq!(m[[1, 0]], 0);
    assert_eq!(m[[1, 1]], 1);
    assert_eq!(m[[1, 2]], 1);
    assert_eq!(m[[2, 0]], 0);
    assert_eq!(m[[2, 1]], 0);
    assert_eq!(m[[2, 2]], 2);
}

#[test]
fn mcm_support() {
    // row sums: [1, 2, 2]
    let cm = mcm_3class();
    assert_eq!(cm.support(), vec![1, 2, 2]);
}

#[test]
fn mcm_accuracy() {
    // trace = 1+1+2 = 4, total = 5 -> accuracy = 0.8
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.accuracy(), 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_precision() {
    // col sums [1, 1, 3] -> precision per class = [1.0, 1.0, 2/3]
    let cm = mcm_3class();
    let prec = cm.per_class_precision();
    assert_eq!(prec.len(), 3);
    assert_abs_diff_eq!(prec[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prec[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prec[2], 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_recall() {
    // row sums [1, 2, 2] -> recall per class = [1.0, 0.5, 1.0]
    let cm = mcm_3class();
    let rec = cm.per_class_recall();
    assert_eq!(rec.len(), 3);
    assert_abs_diff_eq!(rec[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rec[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(rec[2], 1.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_f1() {
    // per-class f1 = [1.0, 2/3, 0.8] from the per-class precision and recall
    let cm = mcm_3class();
    let f1 = cm.per_class_f1();
    assert_eq!(f1.len(), 3);
    assert_abs_diff_eq!(f1[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f1[1], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f1[2], 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_precision_macro() {
    // macro = (1.0 + 1.0 + 2/3) / 3 = 8/9
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.precision(Average::Macro), 8.0 / 9.0, epsilon = 1e-12);
}

#[test]
fn mcm_recall_macro() {
    // macro = (1.0 + 0.5 + 1.0) / 3 = 5/6
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.recall(Average::Macro), 5.0 / 6.0, epsilon = 1e-12);
}

#[test]
fn mcm_f1_macro() {
    // macro = (1.0 + 2/3 + 4/5) / 3 = 37/45
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.f1(Average::Macro), 37.0 / 45.0, epsilon = 1e-12);
}

#[test]
fn mcm_precision_micro() {
    // Micro == accuracy for single-label data -> 0.8
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.precision(Average::Micro), 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_recall_micro() {
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.recall(Average::Micro), 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_f1_micro() {
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.f1(Average::Micro), 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_precision_weighted() {
    // weighted = sum(prec_i * support_i) / total
    //   = (1.0*1 + 1.0*2 + (2/3)*2) / 5 = 13/15
    let cm = mcm_3class();
    assert_abs_diff_eq!(
        cm.precision(Average::Weighted),
        13.0 / 15.0,
        epsilon = 1e-12
    );
}

#[test]
fn mcm_recall_weighted() {
    // weighted = (1.0*1 + 0.5*2 + 1.0*2) / 5 = 4/5
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.recall(Average::Weighted), 4.0 / 5.0, epsilon = 1e-12);
}

#[test]
fn mcm_f1_weighted() {
    // weighted = (1.0*1 + (2/3)*2 + (4/5)*2) / 5 = 59/75
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.f1(Average::Weighted), 59.0 / 75.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_precision_zero_guard() {
    // Classes 1 and 2 never predicted (y_true=[0,1,2], y_pred=[0,0,0]) -> col sums
    // [3,0,0], so precision for the unpredicted classes is 0.0 by guard
    let y_true = array![0usize, 1, 2];
    let y_pred = array![0usize, 0, 0];
    let cm = MulticlassConfusionMatrix::new(&y_true, &y_pred);
    let prec = cm.per_class_precision();
    assert_abs_diff_eq!(prec[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prec[2], 0.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_recall_zero_guard() {
    // Classes 1 and 2 appear only in y_pred (y_true=[0,0,0], y_pred=[0,1,2]) -> row
    // sums [3,0,0], so recall for the absent classes is 0.0 by guard
    let y_true = array![0usize, 0, 0];
    let y_pred = array![0usize, 1, 2];
    let cm = MulticlassConfusionMatrix::new(&y_true, &y_pred);
    let rec = cm.per_class_recall();
    assert_abs_diff_eq!(rec[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rec[2], 0.0, epsilon = 1e-12);
}

#[test]
fn mcm_perfect_accuracy() {
    let y_true = array![0usize, 1, 2, 0, 1];
    let y_pred = array![0usize, 1, 2, 0, 1];
    let cm = MulticlassConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.accuracy(), 1.0, epsilon = 1e-12);
}

#[test]
fn mcm_single_class_only_in_pred() {
    // Label discovery includes classes seen only in y_pred
    // y_true=[0,0], y_pred=[0,1] -> labels {0,1}
    let y_true = array![0usize, 0];
    let y_pred = array![0usize, 1];
    let cm = MulticlassConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.n_classes(), 2);
    assert_eq!(cm.labels(), &[0usize, 1]);
}

#[test]
fn mcm_summary_contains_keywords() {
    let cm = mcm_3class();
    let s = cm.summary();
    assert!(s.contains("accuracy"));
    assert!(s.contains("precision"));
    assert!(s.contains("recall"));
    assert!(s.contains("f1-score"));
    assert!(s.contains("support"));
    assert!(s.contains("macro avg"));
    assert!(s.contains("weighted avg"));
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn mcm_length_mismatch_panics() {
    let y_true = array![0usize, 1, 2];
    let y_pred = array![0usize, 1];
    let _ = MulticlassConfusionMatrix::new(&y_true, &y_pred);
}

#[test]
#[should_panic(expected = "input is empty")]
fn mcm_empty_panics() {
    let y_true: ndarray::Array1<usize> = ndarray::Array1::from(vec![]);
    let y_pred: ndarray::Array1<usize> = ndarray::Array1::from(vec![]);
    let _ = MulticlassConfusionMatrix::new(&y_true, &y_pred);
}

// log_loss

#[test]
fn log_loss_two_samples() {
    // y_true=[0,1], y_prob=[[0.9,0.1],[0.2,0.8]]
    // loss = -(ln(0.9) + ln(0.8)) / 2 ~= 0.16425203348
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[0.9, 0.1], [0.2, 0.8]]);
    let expected = -(0.9_f64.ln() + 0.8_f64.ln()) / 2.0;
    assert_abs_diff_eq!(log_loss(&y_true, &y_prob), expected, epsilon = 1e-9);
}

#[test]
fn log_loss_perfect_probs() {
    // Perfect probs clamped to 1-EPS -> loss = -ln(1-1e-15) ~= 1e-15 (finite, not inf)
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let loss = log_loss(&y_true, &y_prob);
    assert!(loss.is_finite() && loss >= 0.0);
    assert!(loss < 1e-10);
}

#[test]
fn log_loss_zero_prob_clamped() {
    // p=0.0 for the true class clamps to EPS=1e-15 -> loss = -ln(1e-15) ~= 34.5, finite (not +inf)
    let y_true = array![0usize];
    let y_prob = arr2(&[[0.0, 1.0]]);
    let loss = log_loss(&y_true, &y_prob);
    assert!(loss.is_finite());
    assert!(loss > 30.0); // -ln(1e-15) ~= 34.5
}

#[test]
fn log_loss_renormalizes_rows() {
    // Row [2,2] does not sum to 1; sklearn renormalizes to [0.5, 0.5], so the loss for true
    // class 0 is -ln(0.5) = ln(2). Without renormalization it would clamp 2.0 -> ~1 and give ~0.
    let y_true = array![0usize];
    let y_prob = arr2(&[[2.0, 2.0]]);
    let expected = 2.0_f64.ln();
    assert_abs_diff_eq!(log_loss(&y_true, &y_prob), expected, epsilon = 1e-9);
}

#[test]
fn log_loss_multiclass_three_classes() {
    // y_true=[0,1,2], y_prob=[[0.8,0.1,0.1],[0.1,0.7,0.2],[0.2,0.2,0.6]]
    // loss = -(ln(0.8)+ln(0.7)+ln(0.6))/3
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]]);
    let expected = -(0.8_f64.ln() + 0.7_f64.ln() + 0.6_f64.ln()) / 3.0;
    assert_abs_diff_eq!(log_loss(&y_true, &y_prob), expected, epsilon = 1e-9);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn log_loss_row_mismatch_panics() {
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.9, 0.1], [0.2, 0.8]]);
    let _ = log_loss(&y_true, &y_prob);
}

#[test]
#[should_panic(expected = "input is empty")]
fn log_loss_empty_panics() {
    let y_true: ndarray::Array1<usize> = array![];
    let y_prob: ndarray::Array2<f64> = ndarray::Array2::zeros((0, 2));
    let _ = log_loss(&y_true, &y_prob);
}

#[test]
#[should_panic(expected = "out of range")]
fn log_loss_label_out_of_range_panics() {
    let y_true = array![0usize, 5];
    let y_prob = arr2(&[[0.9, 0.1], [0.2, 0.8]]);
    let _ = log_loss(&y_true, &y_prob);
}

// cohen_kappa

#[test]
fn cohen_kappa_perfect() {
    // Perfect agreement -> kappa = 1.0
    let y_true = array![0usize, 1, 0, 1];
    let y_pred = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_chance_level() {
    // 2 classes, balanced (y_true=[0,0,1,1], y_pred=[0,1,0,1]): observed = expected = 0.5
    // -> kappa = 0.0
    let y_true = array![0usize, 0, 1, 1];
    let y_pred = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_negative() {
    // Systematic disagreement (y_true=[0,0,1,1], y_pred=[1,1,0,0]): observed=0.0,
    // expected=0.5 -> kappa = -1.0
    let y_true = array![0usize, 0, 1, 1];
    let y_pred = array![1usize, 1, 0, 0];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), -1.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_degenerate_all_same_label() {
    // All labels identical -> p_e approaches 1.0 -> degenerate denominator, guard returns 1.0
    let y_true = array![0usize, 0, 0, 0];
    let y_pred = array![0usize, 0, 0, 0];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_range_invariant() {
    let y_true = array![0usize, 1, 2, 0, 1];
    let y_pred = array![0usize, 2, 2, 1, 1];
    let k = cohen_kappa(&y_true, &y_pred);
    assert!((-1.0..=1.0).contains(&k));
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn cohen_kappa_length_mismatch_panics() {
    let y_true = array![0usize, 1, 2];
    let y_pred = array![0usize, 1];
    let _ = cohen_kappa(&y_true, &y_pred);
}

#[test]
#[should_panic(expected = "input is empty")]
fn cohen_kappa_empty_panics() {
    let y_true: ndarray::Array1<usize> = ndarray::Array1::from(vec![]);
    let y_pred: ndarray::Array1<usize> = ndarray::Array1::from(vec![]);
    let _ = cohen_kappa(&y_true, &y_pred);
}

// top_k_accuracy

#[test]
fn top_k_accuracy_top1() {
    // k=1 on y_true=[0,1,2], y_prob=[[0.7,0.2,0.1],[0.3,0.3,0.4],[0.1,0.5,0.4]]:
    // only sample 0 has the true class as the top score -> top1 = 1/3
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.5, 0.4]]);
    assert_abs_diff_eq!(
        top_k_accuracy(&y_true, &y_prob, 1),
        1.0 / 3.0,
        epsilon = 1e-9
    );
}

#[test]
fn top_k_accuracy_top2() {
    // k=2 on the same setup: every true class is within the top 2 -> top2 = 1.0
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.5, 0.4]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 2), 1.0, epsilon = 1e-12);
}

#[test]
fn top_k_accuracy_k_ge_n_classes() {
    // k >= n_classes: at most k-1 classes can be strictly higher -> every sample in top-k
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.5, 0.4]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 3), 1.0, epsilon = 1e-12);
}

#[test]
fn top_k_accuracy_all_wrong_top1() {
    // Both samples have two classes scoring higher than the true class -> top1 = 0.0
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[0.1, 0.5, 0.4], [0.5, 0.1, 0.4]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 1), 0.0, epsilon = 1e-12);
}

#[test]
fn top_k_accuracy_tie_boundary_counts_in_favor() {
    // Tied probs at the boundary count in the sample's favor (no strictly higher score),
    // so y_true=[0], y_prob=[[0.5,0.5,0.5]] is correct for any k>=1
    let y_true = array![0usize];
    let y_prob = arr2(&[[0.5, 0.5, 0.5]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 1), 1.0, epsilon = 1e-12);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn top_k_accuracy_length_mismatch_panics() {
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.7, 0.3], [0.4, 0.6]]);
    let _ = top_k_accuracy(&y_true, &y_prob, 1);
}

#[test]
#[should_panic(expected = "input is empty")]
fn top_k_accuracy_empty_panics() {
    let y_true: ndarray::Array1<usize> = array![];
    let y_prob: ndarray::Array2<f64> = ndarray::Array2::zeros((0, 2));
    let _ = top_k_accuracy(&y_true, &y_prob, 1);
}

#[test]
#[should_panic(expected = "must be at least 1")]
fn top_k_accuracy_k_zero_panics() {
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[0.7, 0.3], [0.4, 0.6]]);
    let _ = top_k_accuracy(&y_true, &y_prob, 0);
}

#[test]
#[should_panic(expected = "out of range")]
fn top_k_accuracy_label_out_of_range_panics() {
    let y_true = array![0usize, 5];
    let y_prob = arr2(&[[0.9, 0.1], [0.2, 0.8]]);
    let _ = top_k_accuracy(&y_true, &y_prob, 1);
}

#[test]
#[should_panic(expected = "must not contain NaN")]
fn top_k_accuracy_nan_true_prob_panics() {
    // A NaN true-class probability used to be miscounted as a hit (n_greater = 0); now rejected
    let y_true = array![0usize];
    let y_prob = arr2(&[[f64::NAN, 0.9]]);
    let _ = top_k_accuracy(&y_true, &y_prob, 1);
}

// average_precision

#[test]
fn average_precision_known_value() {
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1]; ranked descending the positive
    // thresholds contribute 0.5 and 1/3 -> AP = 5/6
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let expected = 5.0 / 6.0;
    assert_abs_diff_eq!(
        average_precision(&labels, &scores),
        expected,
        epsilon = 1e-9
    );
}

#[test]
fn average_precision_perfect() {
    // All positives ranked above all negatives -> AP = 1.0
    // labels=[T,T,F,F], scores=[0.9,0.8,0.3,0.1]
    let labels = array![true, true, false, false];
    let scores = array![0.9, 0.8, 0.3, 0.1];
    assert_abs_diff_eq!(average_precision(&labels, &scores), 1.0, epsilon = 1e-12);
}

#[test]
fn average_precision_single_positive() {
    // Single positive at the highest score -> AP = 1.0
    // labels=[T,F,F,F], scores=[0.9,0.6,0.4,0.1]
    let labels = array![true, false, false, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    assert_abs_diff_eq!(average_precision(&labels, &scores), 1.0, epsilon = 1e-12);
}

#[test]
fn average_precision_single_positive_worst_rank() {
    // Single positive at the lowest score (labels=[F,F,F,T],
    // scores=[0.9,0.6,0.4,0.1]) -> P=1/4 at recall 1.0, so AP = 0.25
    let labels = array![false, false, false, true];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    assert_abs_diff_eq!(average_precision(&labels, &scores), 0.25, epsilon = 1e-12);
}

#[test]
fn average_precision_range_invariant() {
    let labels = array![true, false, true, false, true];
    let scores = array![0.9, 0.6, 0.7, 0.3, 0.4];
    let ap = average_precision(&labels, &scores);
    assert!((0.0..=1.0).contains(&ap));
}

#[test]
#[should_panic(expected = "at least one positive label")]
fn average_precision_no_positive_panics() {
    let labels = array![false, false, false];
    let scores = array![0.9, 0.5, 0.1];
    let _ = average_precision(&labels, &scores);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn average_precision_length_mismatch_panics() {
    let labels = array![true, false];
    let scores = array![0.9];
    let _ = average_precision(&labels, &scores);
}

#[test]
#[should_panic(expected = "input is empty")]
fn average_precision_empty_panics() {
    let labels: ndarray::Array1<bool> = array![];
    let scores: ndarray::Array1<f64> = array![];
    let _ = average_precision(&labels, &scores);
}

// roc_curve

#[test]
fn roc_curve_specific_points() {
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1] -> fpr=[0,0,0.5,0.5,1], tpr=[0,0.5,0.5,1,1],
    // thresholds=[1.9,0.9,0.6,0.4,0.1] (origin point prepended)
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (fpr, tpr, thresholds) = roc_curve(&labels, &scores);

    assert_eq!(fpr.len(), 5);
    assert_eq!(tpr.len(), 5);
    assert_eq!(thresholds.len(), 5);

    // Origin
    assert_abs_diff_eq!(fpr[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr[0], 0.0, epsilon = 1e-12);

    // Point after threshold=0.9: 1 TP, 0 FP
    assert_abs_diff_eq!(fpr[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(thresholds[1], 0.9, epsilon = 1e-12);

    // Point after threshold=0.6: 1 TP, 1 FP
    assert_abs_diff_eq!(fpr[2], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr[2], 0.5, epsilon = 1e-12);

    // Point after threshold=0.4: 2 TP, 1 FP
    assert_abs_diff_eq!(fpr[3], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr[3], 1.0, epsilon = 1e-12);

    // Final point: 2 TP, 2 FP -> (1.0, 1.0)
    assert_abs_diff_eq!(fpr[4], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr[4], 1.0, epsilon = 1e-12);
}

#[test]
fn roc_curve_first_point_is_origin() {
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (fpr, tpr, _) = roc_curve(&labels, &scores);
    assert_abs_diff_eq!(fpr[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr[0], 0.0, epsilon = 1e-12);
}

#[test]
fn roc_curve_last_tpr_is_one() {
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (_, tpr, _) = roc_curve(&labels, &scores);
    let n = tpr.len();
    assert_abs_diff_eq!(tpr[n - 1], 1.0, epsilon = 1e-12);
}

#[test]
fn roc_curve_lengths_equal() {
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (fpr, tpr, thresholds) = roc_curve(&labels, &scores);
    assert_eq!(fpr.len(), tpr.len());
    assert_eq!(fpr.len(), thresholds.len());
}

#[test]
fn roc_curve_trapezoidal_area_matches_roc_auc() {
    // Trapezoidal integral of the curve must equal roc_auc on the same data
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1] -> AUC = 0.75
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let auc_direct = roc_auc(&labels, &scores);
    let (fpr, tpr, _) = roc_curve(&labels, &scores);
    let n = fpr.len();
    let trapezoid_area: f64 = (0..n - 1)
        .map(|i| (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0)
        .sum();
    assert_abs_diff_eq!(trapezoid_area, auc_direct, epsilon = 1e-12);
    assert_abs_diff_eq!(trapezoid_area, 0.75, epsilon = 1e-12);
}

#[test]
fn roc_curve_fpr_nondecreasing() {
    // fpr must be non-decreasing (sorted ascending from the origin)
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (fpr, _, _) = roc_curve(&labels, &scores);
    for i in 0..fpr.len() - 1 {
        assert!(fpr[i] <= fpr[i + 1], "fpr not non-decreasing at i={}", i);
    }
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn roc_curve_length_mismatch_panics() {
    let labels = array![true, false, true];
    let scores = array![0.9, 0.6];
    let _ = roc_curve(&labels, &scores);
}

#[test]
#[should_panic(expected = "input is empty")]
fn roc_curve_empty_panics() {
    let labels: ndarray::Array1<bool> = array![];
    let scores: ndarray::Array1<f64> = array![];
    let _ = roc_curve(&labels, &scores);
}

#[test]
#[should_panic(expected = "at least one positive and one negative")]
fn roc_curve_no_positive_panics() {
    let labels = array![false, false, false];
    let scores = array![0.9, 0.5, 0.1];
    let _ = roc_curve(&labels, &scores);
}

#[test]
#[should_panic(expected = "at least one positive and one negative")]
fn roc_curve_no_negative_panics() {
    let labels = array![true, true, true];
    let scores = array![0.9, 0.5, 0.1];
    let _ = roc_curve(&labels, &scores);
}

// precision_recall_curve

#[test]
fn precision_recall_curve_specific_points() {
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1] -> precision=[1,0.5,2/3,0.5,1(close)],
    // recall=[0.5,0.5,1,1,0(close)], thresholds=[0.9,0.6,0.4,0.1]
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (precision, recall, thresholds) = precision_recall_curve(&labels, &scores);

    // Length invariant: precision.len() == recall.len(), thresholds.len() == precision.len()-1
    assert_eq!(precision.len(), recall.len());
    assert_eq!(thresholds.len(), precision.len() - 1);

    // Closing point (last)
    let last = precision.len() - 1;
    assert_abs_diff_eq!(precision[last], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(recall[last], 0.0, epsilon = 1e-12);

    // Intermediate points
    assert_abs_diff_eq!(precision[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(recall[0], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(thresholds[0], 0.9, epsilon = 1e-12);

    assert_abs_diff_eq!(precision[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(recall[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(thresholds[1], 0.6, epsilon = 1e-12);

    assert_abs_diff_eq!(precision[2], 2.0 / 3.0, epsilon = 1e-9);
    assert_abs_diff_eq!(recall[2], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(thresholds[2], 0.4, epsilon = 1e-12);

    assert_abs_diff_eq!(precision[3], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(recall[3], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(thresholds[3], 0.1, epsilon = 1e-12);
}

#[test]
fn precision_recall_curve_closing_point() {
    // Final entry is always (precision=1.0, recall=0.0)
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (precision, recall, _) = precision_recall_curve(&labels, &scores);
    let n = precision.len();
    assert_abs_diff_eq!(precision[n - 1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(recall[n - 1], 0.0, epsilon = 1e-12);
}

#[test]
fn precision_recall_curve_length_invariant() {
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (precision, recall, thresholds) = precision_recall_curve(&labels, &scores);
    assert_eq!(precision.len(), recall.len());
    assert_eq!(thresholds.len(), precision.len() - 1);
}

#[test]
fn precision_recall_curve_recall_nondecreasing_before_close() {
    // Recall is non-decreasing across the curve (excluding the closing 0.0 point),
    // since going from highest to lowest threshold accumulates tp
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (_, recall, _) = precision_recall_curve(&labels, &scores);
    let n = recall.len();
    // Last element is the closing 0.0 point (recall goes 0.5,0.5,1.0,1.0,0.0);
    // check all points except the closing one are non-decreasing
    for i in 0..n - 2 {
        assert!(
            recall[i] <= recall[i + 1],
            "recall not non-decreasing at i={}",
            i
        );
    }
}

#[test]
fn precision_recall_curve_perfect() {
    // All positives scored above all negatives -> P=1.0 at recall=1.0 before close
    // labels=[T,T,F,F], scores=[0.9,0.8,0.3,0.1]
    let labels = array![true, true, false, false];
    let scores = array![0.9, 0.8, 0.3, 0.1];
    let (precision, recall, _) = precision_recall_curve(&labels, &scores);
    // First two points should have precision=1.0 (no FP yet)
    assert_abs_diff_eq!(precision[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(precision[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(recall[1], 1.0, epsilon = 1e-12);
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn precision_recall_curve_length_mismatch_panics() {
    let labels = array![true, false, true];
    let scores = array![0.9, 0.6];
    let _ = precision_recall_curve(&labels, &scores);
}

#[test]
#[should_panic(expected = "input is empty")]
fn precision_recall_curve_empty_panics() {
    let labels: ndarray::Array1<bool> = array![];
    let scores: ndarray::Array1<f64> = array![];
    let _ = precision_recall_curve(&labels, &scores);
}

#[test]
#[should_panic(expected = "at least one positive label")]
fn precision_recall_curve_no_positive_panics() {
    let labels = array![false, false, false];
    let scores = array![0.9, 0.5, 0.1];
    let _ = precision_recall_curve(&labels, &scores);
}

// NaN-score contract: scores cannot be ranked meaningfully (f64::total_cmp would silently treat
// a NaN as the most confident prediction), so the ranking metrics reject NaN, matching sklearn

#[test]
#[should_panic(expected = "must not contain NaN")]
fn roc_auc_with_nan_score_panics() {
    let labels = array![false, true, false, true];
    let scores = array![0.1, 0.4, 0.35, f64::NAN];
    let _ = roc_auc(&labels, &scores);
}

#[test]
#[should_panic(expected = "must not contain NaN")]
fn average_precision_with_nan_score_panics() {
    let labels = array![true, true, false, false];
    let scores = array![f64::NAN, 0.8, 0.3, 0.1];
    let _ = average_precision(&labels, &scores);
}

#[test]
#[should_panic(expected = "must not contain NaN")]
fn roc_curve_with_nan_score_panics() {
    let labels = array![true, false, true, false];
    let scores = array![f64::NAN, 0.6, 0.4, 0.1];
    let _ = roc_curve(&labels, &scores);
}

#[test]
#[should_panic(expected = "must not contain NaN")]
fn precision_recall_curve_with_nan_score_panics() {
    let labels = array![true, false, true, false];
    let scores = array![0.9, f64::NAN, 0.4, 0.1];
    let _ = precision_recall_curve(&labels, &scores);
}
