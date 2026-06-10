//! Integration tests for `metrics::classification`.
//!
//! Expected values are derived from mathematical definitions and hand-checked arithmetic.
//! All derivations are commented inline.

use approx::assert_abs_diff_eq;
use ndarray::{arr2, array};
use rustyml::metrics::{
    Average, ConfusionMatrix, MulticlassConfusionMatrix, accuracy, average_precision, cohen_kappa,
    log_loss, precision_recall_curve, roc_auc, roc_curve, top_k_accuracy,
};

// ──────────────────────────────────────────────────────────────────
// ConfusionMatrix::new  –  TP/FP/TN/FN counting
// ──────────────────────────────────────────────────────────────────

#[test]
fn cm_new_perfect_predictions() {
    // y_true = [1,0,1,0], y_pred = [1,0,1,0]  →  TP=2, FP=0, TN=2, FN=0
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (2, 0, 2, 0));
}

#[test]
fn cm_new_all_wrong() {
    // y_true = [1,0,1,0], y_pred = [0,1,0,1]  →  TP=0, FP=2, TN=0, FN=2
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![0.0, 1.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (0, 2, 0, 2));
}

#[test]
fn cm_new_mixed() {
    // y_true = [1,1,1,1,1,0,0,0], y_pred = [1,1,1,0,0,0,0,1]
    // Positives:  TP=3 (y_true=1, y_pred=1), FN=2 (y_true=1, y_pred=0)
    // Negatives:  TN=2 (y_true=0, y_pred=0), FP=1 (y_true=0, y_pred=1)
    let y_true = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (3, 1, 2, 2));
}

#[test]
fn cm_new_thresholding_at_0_5() {
    // Probabilities: values >= 0.5 → positive. 0.6→1, 0.4→0, 0.9→1, 0.1→0.
    // y_true = [0.9,0.1,0.6,0.4], y_pred = [0.8,0.2,0.3,0.7]
    //   true positives:  0.9→T, 0.6→T
    //   pred positives:  0.8→T, 0.7→T (index 0 and 3)
    //   (T,T)→index0:TP, (T,F)→index2:FN, (F,F)→index1:TN, (F,T)→index3:FP
    //   TP=1, FP=1, TN=1, FN=1
    let y_true = array![0.9, 0.1, 0.6, 0.4];
    let y_pred = array![0.8, 0.2, 0.3, 0.7];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_eq!(cm.get_counts(), (1, 1, 1, 1));
}

// ──────────────────────────────────────────────────────────────────
// ConfusionMatrix derived metrics (tp=3, fp=1, tn=2, fn=2)
//
// y_true = [1,1,1,1,1,0,0,0], y_pred = [1,1,1,0,0,0,0,1]
// ──────────────────────────────────────────────────────────────────

fn cm_3_1_2_2() -> ConfusionMatrix {
    let y_true = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    ConfusionMatrix::new(&y_true, &y_pred)
}

#[test]
fn cm_accuracy_partial() {
    // accuracy = (TP+TN)/(TP+FP+TN+FN) = (3+2)/8 = 5/8 = 0.625
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
    // error_rate = 1 - accuracy  always
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.accuracy() + cm.error_rate(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_accuracy_perfect() {
    // tp=4, fp=0, tn=4, fn=0  →  accuracy = 1.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.accuracy(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_accuracy_all_wrong() {
    // tp=0, fp=2, tn=0, fn=2  →  accuracy = 0.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![0.0, 0.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.accuracy(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_precision_partial() {
    // precision = TP/(TP+FP) = 3/(3+1) = 3/4 = 0.75
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.precision(), 0.75, epsilon = 1e-12);
}

#[test]
fn cm_precision_perfect() {
    // tp=2, fp=0  →  precision = 2/2 = 1.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.precision(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_precision_no_positive_predictions() {
    // tp=0, fp=0 (all predicted negative)  →  precision = 0.0 (convention)
    let y_true = array![1.0, 0.0];
    let y_pred = array![0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    // TP=0, FP=0, TN=1, FN=1
    assert_eq!(cm.get_counts(), (0, 0, 1, 1));
    assert_abs_diff_eq!(cm.precision(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_recall_partial() {
    // recall = TP/(TP+FN) = 3/(3+2) = 3/5 = 0.6
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.recall(), 0.6, epsilon = 1e-12);
}

#[test]
fn cm_recall_no_actual_positives() {
    // tp=0, fn=0 (all actual negatives)  →  recall = 1.0 (convention)
    let y_true = array![0.0, 0.0];
    let y_pred = array![0.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    // TN=1, FP=1, TP=0, FN=0
    assert_eq!(cm.get_counts(), (0, 1, 1, 0));
    assert_abs_diff_eq!(cm.recall(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_specificity_partial() {
    // specificity = TN/(TN+FP) = 2/(2+1) = 2/3
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.specificity(), 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn cm_specificity_no_actual_negatives() {
    // tn=0, fp=0 (all actual positives)  →  specificity = 1.0 (convention)
    let y_true = array![1.0, 1.0];
    let y_pred = array![1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    // TP=1, FN=1, FP=0, TN=0
    assert_eq!(cm.get_counts(), (1, 0, 0, 1));
    assert_abs_diff_eq!(cm.specificity(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_f1_partial() {
    // f1 = 2*prec*rec/(prec+rec) = 2*(3/4)*(3/5)/((3/4)+(3/5))
    //    = 2*(9/20) / (27/20) = (18/20)/(27/20) = 18/27 = 2/3
    let cm = cm_3_1_2_2();
    assert_abs_diff_eq!(cm.f1_score(), 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn cm_f1_perfect() {
    // precision=1.0, recall=1.0  →  f1 = 1.0
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.f1_score(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_f1_zero_precision_and_recall() {
    // TP=0, FP=0, FN=1, TN=1:  precision=0.0 (guard), recall=0/(0+1)=0.0 → f1 = 0.0 (guard)
    let y_true = array![1.0, 0.0];
    let y_pred = array![0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.f1_score(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_mcc_partial() {
    // mcc = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    //     = (3*2 - 1*2) / sqrt(4*5*3*4)
    //     = (6-2) / sqrt(240)
    //     = 4 / sqrt(240)
    //   sqrt(240) = 4*sqrt(15) ≈ 15.4919333848
    //   mcc ≈ 4/15.4919 ≈ 0.25820
    let cm = cm_3_1_2_2();
    let expected = 4.0 / (240.0_f64).sqrt();
    assert_abs_diff_eq!(cm.mcc(), expected, epsilon = 1e-9);
}

#[test]
fn cm_mcc_perfect() {
    // TP=2, FP=0, TN=2, FN=0  →  (4-0)/sqrt(2*2*2*2) = 4/4 = 1.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.mcc(), 1.0, epsilon = 1e-12);
}

#[test]
fn cm_mcc_degenerate_all_predicted_positive() {
    // TP=2, FP=2, TN=0, FN=0  →  denom = sqrt(4*2*2*0) = 0  →  mcc = 0.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![1.0, 1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.mcc(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_mcc_degenerate_all_predicted_negative() {
    // TP=0, FP=0, TN=2, FN=2  →  denom = sqrt(0*2*2*4) = 0  →  mcc = 0.0
    let y_true = array![1.0, 1.0, 0.0, 0.0];
    let y_pred = array![0.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&y_true, &y_pred);
    assert_abs_diff_eq!(cm.mcc(), 0.0, epsilon = 1e-12);
}

#[test]
fn cm_balanced_accuracy_partial() {
    // balanced_accuracy = (recall + specificity)/2
    //   = (3/5 + 2/3)/2 = (0.6 + 0.6667)/2 ≈ 0.6333...
    //   = (0.6 + 2/3)/2 = (9/15 + 10/15)/2 = (19/15)/2 = 19/30
    let cm = cm_3_1_2_2();
    let expected = 19.0 / 30.0;
    assert_abs_diff_eq!(cm.balanced_accuracy(), expected, epsilon = 1e-12);
}

#[test]
fn cm_balanced_accuracy_vs_accuracy_imbalanced() {
    // With all predictions positive on imbalanced data:
    // y_true = [1,1,1,1,0], y_pred = [1,1,1,1,1]
    // TP=4, FP=1, TN=0, FN=0
    // accuracy = 4/5 = 0.8
    // recall = 4/(4+0)=1.0, specificity = 0/(0+1)=0.0
    // balanced_accuracy = (1.0+0.0)/2 = 0.5  (different from plain accuracy)
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

// ──────────────────────────────────────────────────────────────────
// ConfusionMatrix panic paths
// ──────────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────────
// Free function: accuracy
// ──────────────────────────────────────────────────────────────────

#[test]
fn accuracy_perfect() {
    // All labels match  →  accuracy = 1.0
    let y_true = array![0.0, 1.0, 2.0, 3.0];
    let y_pred = array![0.0, 1.0, 2.0, 3.0];
    assert_abs_diff_eq!(accuracy(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

#[test]
fn accuracy_all_wrong() {
    // No labels match  →  accuracy = 0.0
    let y_true = array![0.0, 1.0, 2.0];
    let y_pred = array![1.0, 2.0, 0.0];
    assert_abs_diff_eq!(accuracy(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

#[test]
fn accuracy_partial() {
    // y_true = [0,0,1]  y_pred = [0,1,1]  →  2 correct out of 3  →  2/3
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
    // accuracy(y_true, y_pred) == accuracy(y_pred, y_true) because epsilon comparison is symmetric
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

// ──────────────────────────────────────────────────────────────────
// roc_auc
// ──────────────────────────────────────────────────────────────────

#[test]
fn roc_auc_perfect() {
    // Perfect ranking: all positives score higher than all negatives.
    // labels=[T,T,F,F], scores=[0.9,0.8,0.2,0.1]
    // sorted ascending: (0.1,F)→rank1, (0.2,F)→rank2, (0.8,T)→rank3, (0.9,T)→rank4
    // sum_pos_ranks = 3+4 = 7; pos_count=2, neg_count=2
    // U = 7 - 2*3/2 = 7-3 = 4; AUC = 4/(2*2) = 1.0
    let labels = array![true, true, false, false];
    let scores = array![0.9, 0.8, 0.2, 0.1];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 1.0, epsilon = 1e-12);
}

#[test]
fn roc_auc_worst() {
    // Worst ranking: all positives score lower than all negatives.
    // labels=[T,T,F,F], scores=[0.1,0.2,0.8,0.9]
    // sorted ascending: (0.1,T)→rank1, (0.2,T)→rank2, (0.8,F)→rank3, (0.9,F)→rank4
    // sum_pos_ranks = 1+2 = 3; U = 3-3 = 0; AUC = 0/(2*2) = 0.0
    let labels = array![true, true, false, false];
    let scores = array![0.1, 0.2, 0.8, 0.9];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.0, epsilon = 1e-12);
}

#[test]
fn roc_auc_random_half() {
    // Random ranking ≈ AUC 0.5.
    // labels=[F,T,T,F], scores=[0.1,0.35,0.4,0.8]
    // sorted ascending: (0.1,F)→rank1, (0.35,T)→rank2, (0.4,T)→rank3, (0.8,F)→rank4
    // sum_pos_ranks = 2+3 = 5; U = 5 - 2*3/2 = 5-3 = 2; AUC = 2/(2*2) = 0.5
    let labels = array![false, true, true, false];
    let scores = array![0.1, 0.35, 0.4, 0.8];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.5, epsilon = 1e-12);
}

#[test]
fn roc_auc_with_all_tied_scores() {
    // All scores equal → every class gets average rank = (1+2+3+4)/4 = 2.5
    // labels=[T,T,F,F]; sum_pos_ranks = 2*2.5 = 5; U = 5-3 = 2; AUC = 2/4 = 0.5
    let labels = array![true, true, false, false];
    let scores = array![0.5, 0.5, 0.5, 0.5];
    assert_abs_diff_eq!(roc_auc(&labels, &scores), 0.5, epsilon = 1e-12);
}

#[test]
fn roc_auc_partial_ties() {
    // Two tie groups: scores=[0.8,0.8,0.2,0.2], labels=[T,F,T,F]
    // sorted ascending: (0.2,T)rank1.5, (0.2,F)rank1.5, (0.8,T)rank3.5, (0.8,F)rank3.5
    // avg_rank group [0.2]: (2*1+2-1)/2 = 3/2 = 1.5; group [0.8]: (2*3+2-1)/2 = 7/2 = 3.5
    // pos T's at ranks 1.5 and 3.5; sum_pos_ranks = 5; U = 5-3 = 2; AUC = 2/4 = 0.5
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

// ──────────────────────────────────────────────────────────────────
// MulticlassConfusionMatrix
// ──────────────────────────────────────────────────────────────────

// y_true=[0,1,2,2,1], y_pred=[0,2,2,2,1]
// Matrix (row=true, col=pred):
//   class 0: pred 0 → (0,0)=1
//   class 1: pred 2 → (1,2)=1; pred 1 → (1,1)=1
//   class 2: pred 2 → (2,2)=2
//   [[1,0,0],[0,1,1],[0,0,2]]
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
    // trace = 1+1+2 = 4, total = 5 → accuracy = 4/5 = 0.8
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.accuracy(), 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_precision() {
    // col sums: [1, 1, 3]
    // class 0: tp=1/col0=1 = 1.0
    // class 1: tp=1/col1=1 = 1.0
    // class 2: tp=2/col2=3 = 2/3
    let cm = mcm_3class();
    let prec = cm.per_class_precision();
    assert_eq!(prec.len(), 3);
    assert_abs_diff_eq!(prec[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prec[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prec[2], 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_recall() {
    // row sums: [1, 2, 2]
    // class 0: 1/1=1.0, class 1: 1/2=0.5, class 2: 2/2=1.0
    let cm = mcm_3class();
    let rec = cm.per_class_recall();
    assert_eq!(rec.len(), 3);
    assert_abs_diff_eq!(rec[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rec[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(rec[2], 1.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_f1() {
    // class 0: 2*1.0*1.0/(1.0+1.0) = 1.0
    // class 1: 2*1.0*0.5/(1.0+0.5) = 1.0/1.5 = 2/3
    // class 2: 2*(2/3)*1.0/((2/3)+1.0) = (4/3)/(5/3) = 4/5 = 0.8
    let cm = mcm_3class();
    let f1 = cm.per_class_f1();
    assert_eq!(f1.len(), 3);
    assert_abs_diff_eq!(f1[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f1[1], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f1[2], 0.8, epsilon = 1e-12);
}

#[test]
fn mcm_precision_macro() {
    // macro = (1.0 + 1.0 + 2/3) / 3 = (2 + 2/3) / 3 = (8/3) / 3 = 8/9
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.precision(Average::Macro), 8.0 / 9.0, epsilon = 1e-12);
}

#[test]
fn mcm_recall_macro() {
    // macro = (1.0 + 0.5 + 1.0) / 3 = 2.5/3 = 5/6
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.recall(Average::Macro), 5.0 / 6.0, epsilon = 1e-12);
}

#[test]
fn mcm_f1_macro() {
    // macro = (1.0 + 2/3 + 4/5) / 3
    //   = (15/15 + 10/15 + 12/15) / 3 = (37/15) / 3 = 37/45
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.f1(Average::Macro), 37.0 / 45.0, epsilon = 1e-12);
}

#[test]
fn mcm_precision_micro() {
    // Micro == accuracy for single-label data  → 0.8
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
    //   = (1.0*1 + 1.0*2 + (2/3)*2) / 5
    //   = (1 + 2 + 4/3) / 5
    //   = (3 + 4/3) / 5 = (9/3 + 4/3) / 5 = (13/3) / 5 = 13/15
    let cm = mcm_3class();
    assert_abs_diff_eq!(
        cm.precision(Average::Weighted),
        13.0 / 15.0,
        epsilon = 1e-12
    );
}

#[test]
fn mcm_recall_weighted() {
    // weighted = (1.0*1 + 0.5*2 + 1.0*2) / 5 = (1 + 1 + 2) / 5 = 4/5
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.recall(Average::Weighted), 4.0 / 5.0, epsilon = 1e-12);
}

#[test]
fn mcm_f1_weighted() {
    // weighted = (1.0*1 + (2/3)*2 + (4/5)*2) / 5
    //   = (1 + 4/3 + 8/5) / 5
    //   = (15/15 + 20/15 + 24/15) / 5 = (59/15) / 5 = 59/75
    let cm = mcm_3class();
    assert_abs_diff_eq!(cm.f1(Average::Weighted), 59.0 / 75.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_precision_zero_guard() {
    // Class 2 never predicted → col 2 sum = 0 → precision[2] = 0.0
    // y_true=[0,1,2], y_pred=[0,0,0]
    //   matrix: [[1,0,0],[0,0,0],[0,0,0]] — but col sums: [2,0,0] wait:
    //   (0→0)=TP for 0, (1→0)=FN for 1 landing in col0, (2→0)=FN for 2 landing in col0
    //   matrix: [[1,0,0],[1,0,0],[1,0,0]]  col sums: [3,0,0]
    //   prec[0]=1/3, prec[1]=0.0 (col1 sum=0), prec[2]=0.0 (col2 sum=0)
    let y_true = array![0usize, 1, 2];
    let y_pred = array![0usize, 0, 0];
    let cm = MulticlassConfusionMatrix::new(&y_true, &y_pred);
    let prec = cm.per_class_precision();
    assert_abs_diff_eq!(prec[1], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(prec[2], 0.0, epsilon = 1e-12);
}

#[test]
fn mcm_per_class_recall_zero_guard() {
    // Class appears only in y_pred but not y_true → row sum = 0 → recall = 0.0
    // y_true=[0,0,0], y_pred=[0,1,2]
    //   labels = {0,1,2} (sorted union); true 0→pred 0: matrix[0,0]+=1; true 0→pred 1: matrix[0,1]+=1; true 0→pred 2: matrix[0,2]+=1
    //   matrix: [[1,1,1],[0,0,0],[0,0,0]]  row sums: [3,0,0]
    //   recall[0]=1/3, recall[1]=0.0 (row sum=0), recall[2]=0.0 (row sum=0)
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
    // Verify that label discovery includes classes seen only in y_pred
    // y_true=[0,0], y_pred=[0,1]  → labels should be {0,1}
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

// ──────────────────────────────────────────────────────────────────
// log_loss
// ──────────────────────────────────────────────────────────────────

#[test]
fn log_loss_two_samples() {
    // y_true=[0,1], y_prob=[[0.9,0.1],[0.2,0.8]]
    // loss = -(ln(0.9) + ln(0.8)) / 2
    //   ln(0.9) ≈ -0.10536051565; ln(0.8) ≈ -0.22314355131
    //   loss = (0.10536 + 0.22314) / 2 ≈ 0.16425203348
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[0.9, 0.1], [0.2, 0.8]]);
    let expected = -(0.9_f64.ln() + 0.8_f64.ln()) / 2.0;
    assert_abs_diff_eq!(log_loss(&y_true, &y_prob), expected, epsilon = 1e-9);
}

#[test]
fn log_loss_perfect_probs() {
    // Perfect probs: p[i,true_class] ≈ 1.0 (clamped to 1-EPS)
    // EPS=1e-15; loss = -ln(1-1e-15) ≈ 1e-15 (tiny but finite, not inf)
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let loss = log_loss(&y_true, &y_prob);
    assert!(loss.is_finite() && loss >= 0.0);
    // Very low loss
    assert!(loss < 1e-10);
}

#[test]
fn log_loss_zero_prob_clamped() {
    // p=0.0 for the true class: clamped to EPS=1e-15, loss = -ln(1e-15) ≈ 34.5
    // Should be finite (not +inf)
    let y_true = array![0usize];
    let y_prob = arr2(&[[0.0, 1.0]]);
    let loss = log_loss(&y_true, &y_prob);
    assert!(loss.is_finite());
    assert!(loss > 30.0); // -ln(1e-15) ≈ 34.5
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

// ──────────────────────────────────────────────────────────────────
// cohen_kappa
// ──────────────────────────────────────────────────────────────────

#[test]
fn cohen_kappa_perfect() {
    // Perfect agreement → kappa = 1.0
    let y_true = array![0usize, 1, 0, 1];
    let y_pred = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), 1.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_chance_level() {
    // 2 classes, balanced: y_true=[0,0,1,1], y_pred=[0,1,0,1]
    // Matrix: [[1,1],[1,1]]; n=4
    // observed = 2/4 = 0.5
    // expected = (2/4)*(2/4) + (2/4)*(2/4) = 0.25+0.25 = 0.5
    // kappa = (0.5-0.5)/(1-0.5) = 0.0
    let y_true = array![0usize, 0, 1, 1];
    let y_pred = array![0usize, 1, 0, 1];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), 0.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_negative() {
    // Systematic disagreement: y_true=[0,0,1,1], y_pred=[1,1,0,0]
    // Matrix: [[0,2],[2,0]]; n=4
    // observed = 0/4 = 0.0
    // expected = (2/4)*(2/4) + (2/4)*(2/4) = 0.5
    // kappa = (0.0-0.5)/(1-0.5) = -1.0
    let y_true = array![0usize, 0, 1, 1];
    let y_pred = array![1usize, 1, 0, 0];
    assert_abs_diff_eq!(cohen_kappa(&y_true, &y_pred), -1.0, epsilon = 1e-12);
}

#[test]
fn cohen_kappa_degenerate_all_same_label() {
    // All labels identical → p_e approaches 1.0 → degenerate denominator → returns 1.0
    // y_true=[0,0,0,0], y_pred=[0,0,0,0]
    // Matrix: [[4]]; n=4; observed=1.0; expected=(4/4)*(4/4)=1.0
    // 1-p_e ≈ 0 → guard returns 1.0
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

// ──────────────────────────────────────────────────────────────────
// top_k_accuracy
// ──────────────────────────────────────────────────────────────────

#[test]
fn top_k_accuracy_top1() {
    // y_true=[0,1,2], y_prob=[[0.7,0.2,0.1],[0.3,0.3,0.4],[0.1,0.5,0.4]]
    // k=1: sample 0: true=0,p=0.7; higher_count=0 < 1 → correct
    //      sample 1: true=1,p=0.3; higher: p[2]=0.4 → count=1 >= 1 → wrong
    //      sample 2: true=2,p=0.4; higher: p[1]=0.5 → count=1 >= 1 → wrong
    //  top1 = 1/3
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
    // k=2: same setup
    //   sample 0: higher_count=0 < 2 → correct
    //   sample 1: higher_count=1 < 2 → correct
    //   sample 2: higher_count=1 < 2 → correct
    //   top2 = 1.0
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.5, 0.4]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 2), 1.0, epsilon = 1e-12);
}

#[test]
fn top_k_accuracy_k_ge_n_classes() {
    // k >= n_classes: at most k-1 classes can be strictly higher → every sample in top-k
    let y_true = array![0usize, 1, 2];
    let y_prob = arr2(&[[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.5, 0.4]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 3), 1.0, epsilon = 1e-12);
}

#[test]
fn top_k_accuracy_all_wrong_top1() {
    // Sample 0: true=0,p=0.1; classes 1 and 2 scored higher → count=2 >= 1 → wrong
    // Sample 1: true=1,p=0.1; classes 0 and 2 scored higher → count=2 >= 1 → wrong
    // top1 = 0.0
    let y_true = array![0usize, 1];
    let y_prob = arr2(&[[0.1, 0.5, 0.4], [0.5, 0.1, 0.4]]);
    assert_abs_diff_eq!(top_k_accuracy(&y_true, &y_prob, 1), 0.0, epsilon = 1e-12);
}

#[test]
fn top_k_accuracy_tie_boundary_counts_in_favor() {
    // Tied probs at boundary: ties count in sample's favor
    // y_true=[0], y_prob=[[0.5,0.5,0.5]]: true=0,p=0.5; higher_count=0 (no strictly higher) → correct for any k>=1
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

// ──────────────────────────────────────────────────────────────────
// average_precision
// ──────────────────────────────────────────────────────────────────

#[test]
fn average_precision_known_value() {
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1]
    // Ranked descending: (0.9,T),(0.6,F),(0.4,T),(0.1,F); total_pos=2
    // threshold 0.9: tp=1,fp=0 → P=1/1=1.0, R=1/2=0.5, ΔR=0.5 → contrib=0.5*1.0=0.5
    // threshold 0.6: tp=1,fp=1 → P=0.5, R=0.5, ΔR=0.0 → contrib=0.0
    // threshold 0.4: tp=2,fp=1 → P=2/3, R=1.0, ΔR=0.5 → contrib=0.5*(2/3)=1/3
    // threshold 0.1: tp=2,fp=2 → P=0.5, R=1.0, ΔR=0.0 → contrib=0.0
    // AP = 0.5 + 1/3 = 5/6
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
    // All positives ranked above all negatives → AP = 1.0
    // labels=[T,T,F,F], scores=[0.9,0.8,0.3,0.1]
    // (0.9,T): P=1/1=1.0, R=0.5, ΔR=0.5 → 0.5
    // (0.8,T): P=2/2=1.0, R=1.0, ΔR=0.5 → 0.5
    // (0.3,F): P=2/3, R=1.0, ΔR=0 → 0
    // (0.1,F): P=0.5, R=1.0, ΔR=0 → 0
    // AP = 1.0
    let labels = array![true, true, false, false];
    let scores = array![0.9, 0.8, 0.3, 0.1];
    assert_abs_diff_eq!(average_precision(&labels, &scores), 1.0, epsilon = 1e-12);
}

#[test]
fn average_precision_single_positive() {
    // Single positive at highest score → AP = 1.0
    // labels=[T,F,F,F], scores=[0.9,0.6,0.4,0.1]
    // (0.9,T): P=1/1=1.0, R=1/1=1.0, ΔR=1.0 → 1.0
    // remaining: ΔR=0
    // AP = 1.0
    let labels = array![true, false, false, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    assert_abs_diff_eq!(average_precision(&labels, &scores), 1.0, epsilon = 1e-12);
}

#[test]
fn average_precision_single_positive_worst_rank() {
    // Single positive at lowest score → AP = 1/4 * (1/4) = ...
    // labels=[F,F,F,T], scores=[0.9,0.6,0.4,0.1]; total_pos=1
    // (0.9,F): P=0/1=0, R=0, ΔR=0 → 0
    // (0.6,F): P=0/2=0, R=0, ΔR=0 → 0
    // (0.4,F): P=0/3=0, R=0, ΔR=0 → 0
    // (0.1,T): P=1/4, R=1/1=1.0, ΔR=1.0 → 1*(1/4)=0.25
    // AP = 0.25
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

// ──────────────────────────────────────────────────────────────────
// roc_curve
// ──────────────────────────────────────────────────────────────────

#[test]
fn roc_curve_specific_points() {
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1]
    // total_pos=2, total_neg=2
    // ranked_cumulative (descending): (0.9,tp=1,fp=0),(0.6,tp=1,fp=1),(0.4,tp=2,fp=1),(0.1,tp=2,fp=2)
    // With origin (fpr=0,tpr=0):
    //   fpr: [0.0, 0.0, 0.5, 0.5, 1.0]
    //   tpr: [0.0, 0.5, 0.5, 1.0, 1.0]
    //   thresholds: [0.9+1=1.9, 0.9, 0.6, 0.4, 0.1]
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

    // Final point: 2 TP, 2 FP → (1.0, 1.0)
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
    // Trapezoidal integral of the curve must equal roc_auc on same data.
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1]
    // Hand-calculated AUC = 0.75 (see roc_auc_random test derivations and roc_curve_specific_points)
    // Trapezoid sum: Σ(fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])/2
    //   i=0: (0-0)*(0.5+0)/2 = 0
    //   i=1: (0.5-0)*(0.5+0.5)/2 = 0.5*0.5 = 0.25
    //   i=2: (0.5-0.5)*(1.0+0.5)/2 = 0
    //   i=3: (1.0-0.5)*(1.0+1.0)/2 = 0.5*1.0 = 0.5
    //   total = 0.75
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
    // fpr must be non-decreasing (sorted ascending from origin)
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

// ──────────────────────────────────────────────────────────────────
// precision_recall_curve
// ──────────────────────────────────────────────────────────────────

#[test]
fn precision_recall_curve_specific_points() {
    // labels=[T,F,T,F], scores=[0.9,0.6,0.4,0.1]; total_pos=2
    // ranked descending: (0.9,tp=1,fp=0),(0.6,tp=1,fp=1),(0.4,tp=2,fp=1),(0.1,tp=2,fp=2)
    // precision = [1/1=1.0, 1/2=0.5, 2/3, 2/4=0.5, (closing)1.0]
    // recall    = [1/2=0.5, 1/2=0.5, 2/2=1.0, 1.0, (closing)0.0]
    // thresholds= [0.9, 0.6, 0.4, 0.1]  (len = precision.len()-1)
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
    // Recall is non-decreasing across the curve points (excluding the closing 0.0 point).
    // Because we go from highest threshold to lowest, tp accumulates.
    let labels = array![true, false, true, false];
    let scores = array![0.9, 0.6, 0.4, 0.1];
    let (_, recall, _) = precision_recall_curve(&labels, &scores);
    let n = recall.len();
    // The last element is the closing 0.0 point (recall goes 0.5,0.5,1.0,1.0,0.0)
    // Check all points except the closing one are non-decreasing.
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
    // All positives scored above all negatives: P=1.0 at recall=1.0 (before close)
    // labels=[T,T,F,F], scores=[0.9,0.8,0.3,0.1]
    // (0.9,tp=1,fp=0): P=1.0, R=0.5
    // (0.8,tp=2,fp=0): P=1.0, R=1.0
    // (0.3,tp=2,fp=1): P=2/3, R=1.0
    // (0.1,tp=2,fp=2): P=0.5, R=1.0
    // closing: P=1.0, R=0.0
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
// ──────────────────────────────────────────────────────────────────
// NaN-score ordering contract
//
// roc_auc sorts pairs by `score.total_cmp(&score)` and ranked_cumulative
// (shared by roc_curve / precision_recall_curve / average_precision) sorts by
// the same total order. `f64::total_cmp` is IEEE-754 totalOrder, which places a
// *positive* NaN (the default `f64::NAN`, sign bit 0) ABOVE every finite value
// and above +inf. So a NaN score must NOT panic, must produce a deterministic
// result (identical across repeated calls), and the finite outputs stay finite.
// ──────────────────────────────────────────────────────────────────

#[test]
fn roc_auc_with_nan_score_is_deterministic_and_finite() {
    // labels=[F,T,F,T], scores=[0.1, 0.4, 0.35, NaN].
    // Ascending by total_cmp (positive NaN sorts last):
    //   (0.1,F)→rank1, (0.35,F)→rank2, (0.4,T)→rank3, (NaN,T)→rank4
    // (NaN forms its own tie group: `NaN == NaN` is false, so no averaging.)
    // sum_pos_ranks = 3 + 4 = 7; pos_count=2, neg_count=2
    // U = 7 - 2*3/2 = 7 - 3 = 4; AUC = 4/(2*2) = 1.0
    let labels = array![false, true, false, true];
    let scores = array![0.1, 0.4, 0.35, f64::NAN];

    let auc1 = roc_auc(&labels, &scores);
    let auc2 = roc_auc(&labels, &scores);

    // Contract: deterministic (call twice → identical) and finite (no panic).
    assert!(auc1.is_finite(), "AUC must be finite even with a NaN score");
    assert_eq!(
        auc1.to_bits(),
        auc2.to_bits(),
        "roc_auc must order NaN deterministically (identical across calls)"
    );
    // Derived value: positive NaN ranks highest, so the two positives occupy
    // ranks 3 and 4 → perfect separation → AUC = 1.0.
    assert_abs_diff_eq!(auc1, 1.0, epsilon = 1e-12);
}

#[test]
fn average_precision_with_nan_score_is_deterministic_and_finite() {
    // Exercises the shared `ranked_cumulative` helper (descending total_cmp sort)
    // through a public entry point. labels=[T,T,F,F], scores=[NaN, 0.8, 0.3, 0.1].
    // Descending by total_cmp (positive NaN sorts first):
    //   (NaN,T),(0.8,T),(0.3,F),(0.1,F); total_pos=2
    // (NaN,T): tp=1,fp=0 → P=1.0, R=0.5, ΔR=0.5 → 0.5
    // (0.8,T): tp=2,fp=0 → P=1.0, R=1.0, ΔR=0.5 → 0.5
    // (0.3,F): tp=2,fp=1 → P=2/3, R=1.0, ΔR=0.0 → 0
    // (0.1,F): tp=2,fp=2 → P=0.5, R=1.0, ΔR=0.0 → 0
    // AP = 1.0
    let labels = array![true, true, false, false];
    let scores = array![f64::NAN, 0.8, 0.3, 0.1];

    let ap1 = average_precision(&labels, &scores);
    let ap2 = average_precision(&labels, &scores);

    assert!(ap1.is_finite(), "AP must be finite even with a NaN score");
    assert_eq!(
        ap1.to_bits(),
        ap2.to_bits(),
        "ranked_cumulative must order NaN deterministically (identical across calls)"
    );
    assert_abs_diff_eq!(ap1, 1.0, epsilon = 1e-12);
}

#[test]
fn roc_curve_with_nan_score_is_deterministic_and_no_panic() {
    // Also drives `ranked_cumulative`. labels=[T,F,T,F], scores=[NaN,0.6,0.4,0.1];
    // both a positive and a negative are present, so no panic on label balance.
    // Descending by total_cmp (positive NaN first):
    //   (NaN,T,tp=1,fp=0),(0.6,F,tp=1,fp=1),(0.4,T,tp=2,fp=1),(0.1,F,tp=2,fp=2)
    // total_pos=2, total_neg=2; with origin:
    //   fpr = [0.0, 0.0, 0.5, 0.5, 1.0]
    //   tpr = [0.0, 0.5, 0.5, 1.0, 1.0]
    // (fpr/tpr are count ratios → always finite. thresholds[0] = NaN+1 = NaN by
    // design, so threshold finiteness is intentionally NOT asserted.)
    let labels = array![true, false, true, false];
    let scores = array![f64::NAN, 0.6, 0.4, 0.1];

    let (fpr1, tpr1, thr1) = roc_curve(&labels, &scores);
    let (fpr2, tpr2, thr2) = roc_curve(&labels, &scores);

    // No panic + deterministic across repeated calls (bit-identical, incl. NaN).
    assert_eq!(fpr1.len(), 5);
    assert_eq!(tpr1.len(), 5);
    assert_eq!(thr1.len(), 5);
    for i in 0..fpr1.len() {
        assert!(fpr1[i].is_finite(), "fpr[{i}] must be finite");
        assert!(tpr1[i].is_finite(), "tpr[{i}] must be finite");
        assert_eq!(
            fpr1[i].to_bits(),
            fpr2[i].to_bits(),
            "fpr[{i}] nondeterministic"
        );
        assert_eq!(
            tpr1[i].to_bits(),
            tpr2[i].to_bits(),
            "tpr[{i}] nondeterministic"
        );
        assert_eq!(
            thr1[i].to_bits(),
            thr2[i].to_bits(),
            "threshold[{i}] nondeterministic"
        );
    }
    // Derived finite points (positive NaN is treated as the highest score).
    assert_abs_diff_eq!(fpr1[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr1[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr1[1], 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(fpr1[4], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tpr1[4], 1.0, epsilon = 1e-12);
}
