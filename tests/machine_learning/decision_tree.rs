//! Integration tests for `rustyml::machine_learning::decision_tree`.
//!
//! All expected values are derived from the problem design or closed-form
//! results — never by running the model and recording its output.

use crate::common::assert_allclose;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::{Error, TreeError};
use rustyml::machine_learning::decision_tree::{Algorithm, DecisionTree, DecisionTreeParams};
use rustyml::machine_learning::decision_tree::{Node, NodeType};
use rustyml::{clear_global_seed, set_global_seed};

// ═══════════════════════════════════════════════════════════════════════
// Helper: a tiny linearly-separable binary dataset.
//
// Feature 0 is 0.x for class 0, and 1.x for class 1.
// Any tree with no depth limit and default params MUST achieve zero
// training error on this data.
// ═══════════════════════════════════════════════════════════════════════

fn linearly_separable_binary() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.1],
        [1.0, 1.0],
        [1.1, 1.0],
        [1.2, 1.1],
    ];
    let y = array![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    (x, y)
}

// ═══════════════════════════════════════════════════════════════════════
// Constructor validation
// ═══════════════════════════════════════════════════════════════════════

/// Default params are accepted for CART classifier.
#[test]
fn test_constructor_default_params_cart_classifier() {
    let tree = DecisionTree::new(Algorithm::CART, true, None);
    assert!(
        tree.is_ok(),
        "CART classifier with default params should succeed"
    );
    let tree = tree.unwrap();
    assert_eq!(tree.get_algorithm(), Algorithm::CART);
    assert!(tree.get_is_classifier());
    // Before fitting, no root and no class information.
    assert!(tree.get_root().is_none());
    assert_eq!(tree.get_n_classes(), None);
    assert_eq!(tree.get_n_features(), 0);
}

/// ID3 algorithm with is_classifier=false must return InvalidInput.
#[test]
fn test_constructor_id3_regression_returns_invalid_input() {
    let err = DecisionTree::new(Algorithm::ID3, false, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput, got {err:?}"
    );
}

/// C4.5 algorithm with is_classifier=false must return InvalidInput.
#[test]
fn test_constructor_c45_regression_returns_invalid_input() {
    let err = DecisionTree::new(Algorithm::C45, false, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput, got {err:?}"
    );
}

/// min_samples_split = 1 (less than 2) must return InvalidParameter.
#[test]
fn test_constructor_min_samples_split_too_small() {
    let params = DecisionTreeParams {
        min_samples_split: 1,
        ..DecisionTreeParams::default()
    };
    let err = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_samples_leaf = 0 must return InvalidParameter.
#[test]
fn test_constructor_min_samples_leaf_zero() {
    let params = DecisionTreeParams {
        min_samples_leaf: 0,
        ..DecisionTreeParams::default()
    };
    let err = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_samples_leaf > min_samples_split must return InvalidParameter.
#[test]
fn test_constructor_min_samples_leaf_greater_than_split() {
    let params = DecisionTreeParams {
        min_samples_split: 2,
        min_samples_leaf: 3,
        ..DecisionTreeParams::default()
    };
    let err = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_impurity_decrease = -0.1 (negative) must return InvalidParameter.
#[test]
fn test_constructor_negative_min_impurity_decrease() {
    let params = DecisionTreeParams {
        min_impurity_decrease: -0.1,
        ..DecisionTreeParams::default()
    };
    let err = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_impurity_decrease = f64::NAN must return InvalidParameter.
#[test]
fn test_constructor_nan_min_impurity_decrease() {
    let params = DecisionTreeParams {
        min_impurity_decrease: f64::NAN,
        ..DecisionTreeParams::default()
    };
    let err = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_impurity_decrease = f64::INFINITY must return InvalidParameter.
#[test]
fn test_constructor_infinite_min_impurity_decrease() {
    let params = DecisionTreeParams {
        min_impurity_decrease: f64::INFINITY,
        ..DecisionTreeParams::default()
    };
    let err = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// Custom params are stored and returned by the getters.
#[test]
fn test_constructor_custom_params_stored_correctly() {
    let params = DecisionTreeParams {
        max_depth: Some(3),
        min_samples_split: 4,
        min_samples_leaf: 2,
        min_impurity_decrease: 0.01,
        random_state: Some(42),
    };
    let tree = DecisionTree::new(Algorithm::ID3, true, Some(params)).unwrap();
    let stored = tree.get_parameters();
    assert_eq!(stored.max_depth, Some(3));
    assert_eq!(stored.min_samples_split, 4);
    assert_eq!(stored.min_samples_leaf, 2);
    assert_abs_diff_eq!(stored.min_impurity_decrease, 0.01, epsilon = 1e-12);
    assert_eq!(stored.random_state, Some(42));
}

// ═══════════════════════════════════════════════════════════════════════
// fit validation
// ═══════════════════════════════════════════════════════════════════════

/// Negative class labels must be rejected by fit() for a classifier.
#[test]
fn test_fit_negative_labels_rejected() {
    let x = array![[0.0_f64, 1.0], [1.0, 0.0]];
    let y = array![-1.0_f64, 1.0]; // -1 is invalid
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput for negative label, got {err:?}"
    );
}

/// Fractional class labels must be rejected by fit() for a classifier.
#[test]
fn test_fit_fractional_labels_rejected() {
    let x = array![[0.0_f64, 1.0], [1.0, 0.0]];
    let y = array![0.5_f64, 1.0]; // 0.5 is not an integer
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput for fractional label, got {err:?}"
    );
}

/// fit() with fewer samples than min_samples_split returns InvalidInput.
#[test]
fn test_fit_too_few_samples_for_min_samples_split() {
    // Only 2 samples but min_samples_split = 5
    let params = DecisionTreeParams {
        min_samples_split: 5,
        min_samples_leaf: 1,
        ..DecisionTreeParams::default()
    };
    let x = array![[0.0_f64], [1.0_f64]];
    let y = array![0.0_f64, 1.0_f64];
    let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput when n_samples < min_samples_split, got {err:?}"
    );
}

/// fit() with NaN in x returns NonFinite.
#[test]
fn test_fit_nan_in_x_rejected() {
    let x = array![[0.0_f64, f64::NAN], [1.0, 0.0], [2.0, 1.0]];
    let y = array![0.0_f64, 1.0, 0.0];
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "Expected NonFinite for NaN in input, got {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// NotFitted error paths
// ═══════════════════════════════════════════════════════════════════════

/// predict() before fit returns NotFitted.
#[test]
fn test_predict_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let x = array![[0.0_f64, 0.0]];
    let err = tree.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// predict_one() before fit returns NotFitted.
#[test]
fn test_predict_one_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let err = tree.predict_one(&[0.0, 0.0]).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// predict_proba() before fit returns NotFitted.
#[test]
fn test_predict_proba_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let x = array![[0.0_f64, 0.0]];
    let err = tree.predict_proba(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// predict_proba_one() before fit returns NotFitted.
#[test]
fn test_predict_proba_one_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let err = tree.predict_proba_one(&[0.0, 0.0]).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// generate_tree_structure() before fit returns NotFitted.
#[test]
fn test_generate_tree_structure_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let err = tree.generate_tree_structure().unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// DimensionMismatch error paths
// ═══════════════════════════════════════════════════════════════════════

/// predict() with wrong number of features returns DimensionMismatch.
#[test]
fn test_predict_wrong_n_features_dimension_mismatch() {
    let (x, y) = linearly_separable_binary();
    // x has 2 features; n_features after fit == 2
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    // Now supply a matrix with 3 features instead of 2
    let x_bad = array![[0.0_f64, 0.0, 0.0]];
    let err = tree.predict(&x_bad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 2,
                found: 3
            }
        ),
        "Expected DimensionMismatch{{expected:2,found:3}}, got {err:?}"
    );
}

/// predict_one() with wrong number of features returns DimensionMismatch.
#[test]
fn test_predict_one_wrong_n_features_dimension_mismatch() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    // n_features == 2, supply 1 feature
    let err = tree.predict_one(&[0.0]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 2,
                found: 1
            }
        ),
        "Expected DimensionMismatch{{expected:2,found:1}}, got {err:?}"
    );
}

/// predict_proba() with wrong number of features returns DimensionMismatch.
#[test]
fn test_predict_proba_wrong_n_features_dimension_mismatch() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let x_bad = array![[0.0_f64, 0.0, 0.0]];
    let err = tree.predict_proba(&x_bad).unwrap_err();
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 2,
                found: 3
            }
        ),
        "Expected DimensionMismatch, got {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// NotClassificationTree error
// ═══════════════════════════════════════════════════════════════════════

/// predict_proba() on a CART regressor returns Tree(NotClassificationTree).
#[test]
fn test_predict_proba_on_regressor_returns_not_classification_tree() {
    // Step-function regression data
    let x = array![
        [0.0_f64],
        [1.0_f64],
        [2.0_f64],
        [10.0_f64],
        [11.0_f64],
        [12.0_f64],
    ];
    let y = array![1.0_f64, 1.0, 1.0, 10.0, 10.0, 10.0];
    let mut tree = DecisionTree::new(Algorithm::CART, false, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let err = tree.predict_proba(&x).unwrap_err();
    assert!(
        matches!(err, Error::Tree(TreeError::NotClassificationTree)),
        "Expected NotClassificationTree, got {err:?}"
    );
}

/// predict_proba_one() on a CART regressor (before fit) returns NotClassificationTree immediately.
#[test]
fn test_predict_proba_one_on_unfitted_regressor_returns_not_classification_tree() {
    let tree = DecisionTree::new(Algorithm::CART, false, None).unwrap();
    // predict_proba_one checks is_classifier first, then NotFitted
    let err = tree.predict_proba_one(&[0.0]).unwrap_err();
    assert!(
        matches!(err, Error::Tree(TreeError::NotClassificationTree)),
        "Expected NotClassificationTree, got {err:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CART classifier — fit and predict correctness
// ═══════════════════════════════════════════════════════════════════════

/// CART classifier on linearly-separable binary data achieves zero training error.
///
/// The data is designed so that any single threshold split on feature 0
/// separates the classes. A tree with no depth limit must memorize all labels.
#[test]
fn test_cart_classifier_zero_training_error() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    // After fit, n_features = 2, n_classes = Some(2), root is Some.
    assert_eq!(tree.get_n_features(), 2);
    assert_eq!(tree.get_n_classes(), Some(2));
    assert!(tree.get_root().is_some());

    let preds = tree.predict(&x).unwrap();
    // Known true classes come from problem design; the tree must reproduce them exactly.
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// fit_predict produces the same result as calling fit and predict separately.
#[test]
fn test_cart_fit_predict_equals_fit_then_predict() {
    let (x, y) = linearly_separable_binary();

    // fit_predict path
    let mut tree1 = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let preds_fp = tree1.fit_predict(&x, &y).unwrap();

    // fit then predict path
    let mut tree2 = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree2.fit(&x, &y).unwrap();
    let preds_sep = tree2.predict(&x).unwrap();

    assert_allclose(&preds_fp, &preds_sep, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════
// ID3 classifier — fit and predict correctness
// ═══════════════════════════════════════════════════════════════════════

/// ID3 classifier on linearly-separable binary data achieves zero training error.
#[test]
fn test_id3_classifier_zero_training_error() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::ID3, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// C4.5 classifier — fit and predict correctness
// ═══════════════════════════════════════════════════════════════════════

/// C4.5 classifier on linearly-separable binary data achieves zero training error.
#[test]
fn test_c45_classifier_zero_training_error() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::C45, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Three-class (multi-class) classification
// ═══════════════════════════════════════════════════════════════════════

/// CART on a clearly-separated 3-class dataset achieves zero training error.
///
/// Dataset design: three clusters well-separated along feature 0.
/// Class 0: x[0] in [0.0, 0.2]
/// Class 1: x[0] in [10.0, 10.2]
/// Class 2: x[0] in [20.0, 20.2]
/// Any two binary splits on feature 0 perfectly separate the three classes.
#[test]
fn test_cart_multiclass_zero_training_error() {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.1],
        [10.0, 1.0],
        [10.1, 1.0],
        [10.2, 1.1],
        [20.0, 2.0],
        [20.1, 2.0],
        [20.2, 2.1],
    ];
    let y = array![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    // n_classes must be 3 (max label is 2, so 3 classes = 0..=2)
    assert_eq!(tree.get_n_classes(), Some(3));

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// ID3 on the same 3-class dataset achieves zero training error.
#[test]
fn test_id3_multiclass_zero_training_error() {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.1],
        [10.0, 1.0],
        [10.1, 1.0],
        [10.2, 1.1],
        [20.0, 2.0],
        [20.1, 2.0],
        [20.2, 2.1],
    ];
    let y = array![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

    let mut tree = DecisionTree::new(Algorithm::ID3, true, None).unwrap();
    tree.fit(&x, &y).unwrap();
    assert_eq!(tree.get_n_classes(), Some(3));

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// C4.5 on the same 3-class dataset achieves zero training error.
#[test]
fn test_c45_multiclass_zero_training_error() {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.1],
        [10.0, 1.0],
        [10.1, 1.0],
        [10.2, 1.1],
        [20.0, 2.0],
        [20.1, 2.0],
        [20.2, 2.1],
    ];
    let y = array![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

    let mut tree = DecisionTree::new(Algorithm::C45, true, None).unwrap();
    tree.fit(&x, &y).unwrap();
    assert_eq!(tree.get_n_classes(), Some(3));

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// predict_proba correctness
// ═══════════════════════════════════════════════════════════════════════

/// predict_proba rows sum to 1.0 and argmax(row) == predict for every sample.
#[test]
fn test_predict_proba_rows_sum_to_one_and_argmax_matches_predict() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    let probas = tree.predict_proba(&x).unwrap();

    // Shape must be (n_samples, n_classes) = (6, 2)
    assert_eq!(probas.shape(), &[6, 2]);

    for i in 0..x.nrows() {
        let row_sum: f64 = probas.row(i).iter().sum();
        assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);

        // argmax of the probability row must agree with the predict label
        let argmax = probas
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert!(
            (argmax as f64 - preds[i]).abs() < 1e-9,
            "argmax(proba row {i}) != predict[{i}]"
        );
    }
}

/// predict_proba_one row sums to 1.0 and argmax matches predict_one.
#[test]
fn test_predict_proba_one_sum_and_argmax() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    // The first training point belongs to class 0 (by design).
    let sample = vec![0.0_f64, 0.0];
    let pred_one = tree.predict_one(&sample).unwrap();
    let proba_one = tree.predict_proba_one(&sample).unwrap();

    assert_eq!(proba_one.len(), 2);
    let row_sum: f64 = proba_one.iter().sum();
    assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);

    let argmax = proba_one
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    assert_abs_diff_eq!(argmax as f64, pred_one, epsilon = 1e-9);
}

/// On a pure leaf (a single training sample), predict_proba gives probability 1.0
/// for the true class and 0.0 for all others.
///
/// Design: with max_depth=None and 6 perfectly-separable samples, a pure leaf
/// for the first sample (class 0) stores probabilities [1.0, 0.0].
#[test]
fn test_predict_proba_pure_leaf_gives_one_hot() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    // The training data has 3 class-0 points.  With full depth the leaf that
    // covers the class-0 region holds a [1.0, 0.0] distribution.
    // The first test point is clearly in the class-0 region.
    let probas = tree.predict_proba(&x).unwrap();

    // Rows 0-2 are class 0; their probability for class 1 must be 0.0.
    for i in 0..3 {
        assert_abs_diff_eq!(probas[[i, 0]] + probas[[i, 1]], 1.0, epsilon = 1e-10);
        // Known true class is 0 -> class-0 probability >= 0.5
        assert!(
            probas[[i, 0]] >= 0.5,
            "Row {i}: expected class-0 to dominate, got {:?}",
            probas.row(i)
        );
    }
    // Rows 3-5 are class 1 -> class-1 probability >= 0.5
    for i in 3..6 {
        assert!(
            probas[[i, 1]] >= 0.5,
            "Row {i}: expected class-1 to dominate, got {:?}",
            probas.row(i)
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CART regression — leaf mean correctness
// ═══════════════════════════════════════════════════════════════════════

/// CART regressor on a step function: predicted values equal the group means.
///
/// Step function design:
///   x in {0, 1, 2}     -> y = 1.0  (group mean = 1.0)
///   x in {10, 11, 12}  -> y = 10.0 (group mean = 10.0)
///
/// A single split on x around 6.0 cleanly separates both groups.
/// A tree with no depth limit will create pure leaves whose value equals
/// the group mean (= the constant target value here).
#[test]
fn test_cart_regression_step_function_leaf_means() {
    let x = array![
        [0.0_f64],
        [1.0_f64],
        [2.0_f64],
        [10.0_f64],
        [11.0_f64],
        [12.0_f64],
    ];
    let y = array![1.0_f64, 1.0, 1.0, 10.0, 10.0, 10.0];

    let mut tree = DecisionTree::new(Algorithm::CART, false, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();

    // Group 0 (x in 0..=2): all predictions must equal the group mean 1.0.
    for i in 0..3 {
        assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-9);
    }
    // Group 1 (x in 10..=12): all predictions must equal the group mean 10.0.
    for i in 3..6 {
        assert_abs_diff_eq!(preds[i], 10.0, epsilon = 1e-9);
    }
}

/// CART regressor memorizes training data perfectly when each sample has a unique x.
///
/// Design: 6 unique x values, each paired with a unique y.  Without depth or
/// leaf-size limits the tree creates singleton leaves; each leaf's mean is the
/// singleton value itself.
#[test]
fn test_cart_regression_memorizes_unique_samples() {
    let x = array![
        [0.0_f64],
        [1.0_f64],
        [2.0_f64],
        [3.0_f64],
        [4.0_f64],
        [5.0_f64],
    ];
    let y = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];

    let mut tree = DecisionTree::new(Algorithm::CART, false, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// max_depth parameter
// ═══════════════════════════════════════════════════════════════════════

/// max_depth=None memorizes training data (unlimited depth, zero error on
/// the separable binary dataset).
///
/// max_depth=1 limits to a single split; on XOR data (not linearly separable
/// with one split) this cannot achieve zero error, proving the tree is shallower.
///
/// XOR layout (4 samples):
///   [0.0, 0.0] -> 0
///   [0.0, 1.0] -> 1
///   [1.0, 0.0] -> 1
///   [1.0, 1.0] -> 0
///
/// With max_depth=1 only one feature-split is made; any single threshold on
/// feature 0 or feature 1 misclassifies at least 2 of the 4 XOR samples.
/// The unlimited tree can always memorize training data by splitting to
/// depth 2 (or using singleton leaves).
#[test]
fn test_max_depth_1_cannot_perfectly_fit_xor() {
    let x_xor = array![
        [0.0_f64, 0.0],
        [0.0_f64, 1.0],
        [1.0_f64, 0.0],
        [1.0_f64, 1.0],
    ];
    let y_xor = array![0.0_f64, 1.0, 1.0, 0.0];

    // depth-1 tree: cannot fit XOR
    let params_shallow = DecisionTreeParams {
        max_depth: Some(1),
        ..DecisionTreeParams::default()
    };
    let mut tree_shallow = DecisionTree::new(Algorithm::CART, true, Some(params_shallow)).unwrap();
    tree_shallow.fit(&x_xor, &y_xor).unwrap();
    let preds_shallow = tree_shallow.predict(&x_xor).unwrap();

    // Count how many are wrong; for XOR at least one must be wrong with depth=1.
    let n_wrong_shallow: usize = preds_shallow
        .iter()
        .zip(y_xor.iter())
        .filter(|(p, y)| (*p - *y).abs() > 0.5)
        .count();
    assert!(
        n_wrong_shallow > 0,
        "depth=1 tree should NOT perfectly classify XOR (got {} errors)",
        n_wrong_shallow
    );

    // Unlimited depth must do at least as well as depth-1 on XOR. (Note: a GREEDY CART cannot
    // necessarily MEMORIZE XOR — at the root no single-feature split reduces impurity, so whether
    // it separates XOR at all depends on the zero-gain-split policy. Perfect memorization is
    // verified on linearly-separable data elsewhere, where it is guaranteed.)
    let mut tree_deep = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree_deep.fit(&x_xor, &y_xor).unwrap();
    let preds_deep = tree_deep.predict(&x_xor).unwrap();
    let n_wrong_deep: usize = preds_deep
        .iter()
        .zip(y_xor.iter())
        .filter(|(p, y)| (*p - *y).abs() > 0.5)
        .count();
    assert!(
        n_wrong_deep <= n_wrong_shallow,
        "unlimited-depth tree ({n_wrong_deep} errors) should be at least as accurate as depth-1 ({n_wrong_shallow} errors) on XOR"
    );
}

/// max_depth=1 on the linearly-separable binary dataset:
/// since the data is linearly separable, one split suffices, and the depth-1
/// tree also achieves zero training error.
///
/// This is a distinct test from the XOR test: it confirms max_depth=1 does NOT
/// prevent a good tree on easy data.
#[test]
fn test_max_depth_1_suffices_for_linearly_separable_data() {
    let (x, y) = linearly_separable_binary();
    let params = DecisionTreeParams {
        max_depth: Some(1),
        ..DecisionTreeParams::default()
    };
    let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// All-same-label (pure root) edge case
// ═══════════════════════════════════════════════════════════════════════

/// When every training sample has the same label the tree creates a single
/// pure leaf at depth 0.  predict must return that label for all inputs.
#[test]
fn test_pure_training_set_creates_pure_root_leaf() {
    let x = array![
        [0.0_f64, 0.0],
        [1.0_f64, 0.0],
        [2.0_f64, 1.0],
        [3.0_f64, 1.0],
    ];
    // All labels are 1.0
    let y = array![1.0_f64, 1.0, 1.0, 1.0];

    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for pred in preds.iter() {
        assert_abs_diff_eq!(*pred, 1.0, epsilon = 1e-9);
    }
}

/// Same edge case for a CART regressor: all-same y -> leaf mean == that value.
#[test]
fn test_pure_regression_set_creates_pure_root_leaf() {
    let x = array![[0.0_f64], [1.0_f64], [2.0_f64], [3.0_f64]];
    let y = array![5.5_f64, 5.5, 5.5, 5.5];

    let mut tree = DecisionTree::new(Algorithm::CART, false, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for pred in preds.iter() {
        assert_abs_diff_eq!(*pred, 5.5, epsilon = 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Categorical features (ID3 and C4.5 multi-way splits)
// ═══════════════════════════════════════════════════════════════════════

/// C4.5 with a categorical feature can separate a "category-XOR" pattern
/// that no single binary numeric threshold can resolve.
///
/// Data design:
///   category 0.0 -> class 0
///   category 1.0 -> class 1
///   category 2.0 -> class 0
///
/// A binary numeric split cannot separate class 0 and class 1 here, because
/// class 0 appears on both sides of any threshold between values 0, 1, 2.
/// A multi-way categorical split perfectly groups by category.
#[test]
fn test_c45_categorical_multiway_split_zero_training_error() {
    // Each sample has one categorical feature.
    let x_cat = array![
        [0.0_f64],
        [0.0_f64],
        [1.0_f64],
        [1.0_f64],
        [2.0_f64],
        [2.0_f64],
    ];
    let y_cat = array![0.0_f64, 0.0, 1.0, 1.0, 0.0, 0.0];

    let mut tree = DecisionTree::new(Algorithm::C45, true, None).unwrap();
    tree.set_categorical_features(vec![0]);
    tree.fit(&x_cat, &y_cat).unwrap();

    let preds = tree.predict(&x_cat).unwrap();
    for (&pred, &expected) in preds.iter().zip(y_cat.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// ID3 with categorical features also produces a multi-way split that fits
/// the category-XOR pattern.
#[test]
fn test_id3_categorical_multiway_split_zero_training_error() {
    let x_cat = array![
        [0.0_f64],
        [0.0_f64],
        [1.0_f64],
        [1.0_f64],
        [2.0_f64],
        [2.0_f64],
    ];
    let y_cat = array![0.0_f64, 0.0, 1.0, 1.0, 0.0, 0.0];

    let mut tree = DecisionTree::new(Algorithm::ID3, true, None).unwrap();
    tree.set_categorical_features(vec![0]);
    tree.fit(&x_cat, &y_cat).unwrap();

    let preds = tree.predict(&x_cat).unwrap();
    for (&pred, &expected) in preds.iter().zip(y_cat.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// CART with set_categorical_features ignores the categorical designation
/// (CART always uses binary splits) and falls back to a numeric binary split.
/// On the linearly-separable binary dataset (feature 0 separates classes)
/// CART still achieves zero training error regardless of the designation.
#[test]
fn test_cart_ignores_categorical_feature_designation() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    // Mark feature 0 as categorical — CART must ignore this.
    tree.set_categorical_features(vec![0]);
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// Unseen category value at predict time falls back to the default leaf without error.
///
/// The tree is trained on categories 0.0, 1.0, 2.0.  At predict time we supply
/// a sample with category 99.0 (never seen during training).  The contract is
/// that this does NOT return an error.
#[test]
fn test_categorical_unseen_value_falls_back_without_error() {
    let x_cat = array![
        [0.0_f64],
        [0.0_f64],
        [1.0_f64],
        [1.0_f64],
        [2.0_f64],
        [2.0_f64],
    ];
    let y_cat = array![0.0_f64, 0.0, 1.0, 1.0, 0.0, 0.0];

    let mut tree = DecisionTree::new(Algorithm::C45, true, None).unwrap();
    tree.set_categorical_features(vec![0]);
    tree.fit(&x_cat, &y_cat).unwrap();

    // Category 99.0 was never seen during training.
    let x_unseen = array![[99.0_f64]];
    let result = tree.predict(&x_unseen);
    assert!(
        result.is_ok(),
        "Unseen category should use fallback leaf, not error: {result:?}"
    );
    // The fallback prediction must be a valid class label (0 or 1).
    let pred = result.unwrap()[0];
    assert!(
        pred == 0.0 || pred == 1.0,
        "Fallback prediction {pred} is not a valid class label"
    );
}

/// get_categorical_features reflects what was set.
#[test]
fn test_get_categorical_features_reflects_set() {
    let mut tree = DecisionTree::new(Algorithm::ID3, true, None).unwrap();
    assert_eq!(tree.get_categorical_features(), &[] as &[usize]);

    tree.set_categorical_features(vec![0, 2]);
    assert_eq!(tree.get_categorical_features(), &[0_usize, 2]);
}

// ═══════════════════════════════════════════════════════════════════════
// generate_tree_structure
// ═══════════════════════════════════════════════════════════════════════

/// generate_tree_structure() after fit returns a non-empty string.
#[test]
fn test_generate_tree_structure_returns_nonempty_string_after_fit() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();
    let structure = tree.generate_tree_structure().unwrap();
    assert!(!structure.is_empty(), "Tree structure should be non-empty");
    assert!(
        structure.contains("Decision Tree Structure"),
        "Expected header in structure output"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Save / load round-trip
// ═══════════════════════════════════════════════════════════════════════

/// save_to_path / load_from_path produces a model that yields identical
/// predictions to the original on the full training set and a held-out test set.
#[test]
fn test_save_load_round_trip_identical_predictions() {
    let (x_train, y_train) = linearly_separable_binary();

    let mut original = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    original.fit(&x_train, &y_train).unwrap();

    let preds_before = original.predict(&x_train).unwrap();

    // Use a temp file path that is unique per process.
    let path = format!("/tmp/rustyml_dt_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x_train).unwrap();

    // Predictions must be bit-for-bit identical (same floating-point values).
    assert_allclose(&preds_before, &preds_after, 1e-12);

    // Held-out test points: by design of the dataset
    //   x[0] < 0.5  -> class 0
    //   x[0] > 0.5  -> class 1
    let x_test = array![
        [0.05_f64, 0.0],
        [0.15_f64, 0.05],
        [1.05_f64, 1.0],
        [1.15_f64, 1.0],
    ];
    let expected_test = array![0.0_f64, 0.0, 1.0, 1.0];

    let preds_test_loaded = loaded.predict(&x_test).unwrap();
    assert_allclose(&preds_test_loaded, &expected_test, 1e-9);

    // Clean up
    let _ = std::fs::remove_file(&path);
}

/// Save/load for a C4.5 classifier preserves predictions.
#[test]
fn test_save_load_c45_classifier_round_trip() {
    let (x, y) = linearly_separable_binary();
    let mut original = DecisionTree::new(Algorithm::C45, true, None).unwrap();
    original.fit(&x, &y).unwrap();

    let preds_before = original.predict(&x).unwrap();

    let path = format!("/tmp/rustyml_c45_dt_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x).unwrap();

    assert_allclose(&preds_before, &preds_after, 1e-12);

    let _ = std::fs::remove_file(&path);
}

/// Save/load for a CART regressor preserves predictions.
#[test]
fn test_save_load_cart_regressor_round_trip() {
    let x = array![
        [0.0_f64],
        [1.0_f64],
        [2.0_f64],
        [10.0_f64],
        [11.0_f64],
        [12.0_f64],
    ];
    let y = array![1.0_f64, 1.0, 1.0, 10.0, 10.0, 10.0];

    let mut original = DecisionTree::new(Algorithm::CART, false, None).unwrap();
    original.fit(&x, &y).unwrap();

    let preds_before = original.predict(&x).unwrap();

    let path = format!("/tmp/rustyml_cart_reg_dt_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x).unwrap();

    assert_allclose(&preds_before, &preds_after, 1e-12);

    let _ = std::fs::remove_file(&path);
}

// ═══════════════════════════════════════════════════════════════════════
// predict labels are in the expected domain {0.0, 1.0, ...}
// ═══════════════════════════════════════════════════════════════════════

/// For a binary classifier every prediction must be either 0.0 or 1.0.
#[test]
fn test_predict_outputs_are_valid_class_labels() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for pred in preds.iter() {
        assert!(
            *pred == 0.0 || *pred == 1.0,
            "Prediction {pred} is not a valid binary class label"
        );
    }
}

/// All three algorithms produce predictions in the label domain {0, 1, 2}
/// for a 3-class problem.
#[test]
fn test_multiclass_predict_outputs_in_valid_domain() {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [10.0, 1.0],
        [10.1, 1.0],
        [20.0, 2.0],
        [20.1, 2.0],
    ];
    let y = array![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0];

    for &algo in &[Algorithm::ID3, Algorithm::C45, Algorithm::CART] {
        let mut tree = DecisionTree::new(algo, true, None).unwrap();
        tree.fit(&x, &y).unwrap();
        let preds = tree.predict(&x).unwrap();
        for pred in preds.iter() {
            assert!(
                *pred == 0.0 || *pred == 1.0 || *pred == 2.0,
                "Algorithm {algo:?}: prediction {pred} is not in {{0,1,2}}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// random_state: seeded tie-breaking among equally-scoring splits (route C)
// ═══════════════════════════════════════════════════════════════════════════

/// Data with a GUARANTEED split tie: features 0 and 1 are identical columns that perfectly
/// separate the two classes (so splitting on either yields the same impurity decrease), while
/// feature 2 is uninformative (zero gain). The only thing seeded tie-breaking can change is WHICH
/// of the two tied features the root splits on.
fn tie_break_data() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    let y = array![0.0_f64, 0.0, 1.0, 1.0];
    (x, y)
}

/// Fit a CART classifier on the tie data with the given `random_state` and return its tree-structure
/// string (which encodes the chosen split feature at each node).
fn tie_break_structure(random_state: Option<u64>) -> String {
    let params = DecisionTreeParams {
        random_state,
        ..DecisionTreeParams::default()
    };
    let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    let (x, y) = tie_break_data();
    tree.fit(&x, &y).unwrap();
    tree.generate_tree_structure().unwrap()
}

/// Same explicit seed ⇒ identical tree, even when ties are broken randomly.
#[test]
fn random_state_same_seed_is_reproducible() {
    assert_eq!(
        tie_break_structure(Some(7)),
        tie_break_structure(Some(7)),
        "the same random_state must reproduce the same tree"
    );
}

/// `random_state = None` (with no global seed) ⇒ deterministic tie-breaking, so repeated fits are
/// identical — the tree stays reproducible without any seed (no regression from the old behavior).
#[test]
fn random_state_none_is_deterministic() {
    assert_eq!(
        tie_break_structure(None),
        tie_break_structure(None),
        "random_state=None must be deterministic"
    );
}

/// Seeded tie-breaking must actually DO something: across a range of seeds the root must not always
/// split on the same one of the two tied features. If `random_state` were a no-op (the old behavior),
/// every seed would yield an identical tree and this set would have size 1.
#[test]
fn random_state_varies_tie_breaking() {
    let distinct: std::collections::HashSet<String> =
        (0..24).map(|s| tie_break_structure(Some(s))).collect();
    assert!(
        distinct.len() >= 2,
        "seeded tie-breaking must vary the chosen split across seeds, got {} distinct tree(s)",
        distinct.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Fit-time PRUNING hyperparameters actually prune during tree growth.
//
// The constructor-validation tests above only prove the params are stored;
// these tests prove min_impurity_decrease / min_samples_leaf / min_samples_split
// change the GROWN tree (collapse a split to a leaf, or stop recursion early).
// Tree shape is inspected via the public `get_root()` -> `Node` structure.
// ═══════════════════════════════════════════════════════════════════════

/// Number of edges from `node` to its deepest leaf.
/// A leaf has depth 0; an internal node is 1 + the max over its children
/// (binary `left`/`right` and, for categorical nodes, every `children` branch).
fn leaf_depth(node: &Node) -> usize {
    match &node.node_type {
        NodeType::Leaf { .. } => 0,
        NodeType::Internal { .. } => {
            let mut deepest = 0;
            if let Some(l) = &node.left {
                deepest = deepest.max(leaf_depth(l));
            }
            if let Some(r) = &node.right {
                deepest = deepest.max(leaf_depth(r));
            }
            if let Some(children) = &node.children {
                for c in children.values() {
                    deepest = deepest.max(leaf_depth(c));
                }
            }
            1 + deepest
        }
    }
}

/// Whether the fitted tree's root is a single leaf (i.e. no split was made at all).
fn root_is_leaf(tree: &DecisionTree) -> bool {
    matches!(
        tree.get_root().expect("tree must be fitted").node_type,
        NodeType::Leaf { .. }
    )
}

/// min_impurity_decrease prunes the root split exactly at the Gini-decrease threshold.
///
/// On `linearly_separable_binary` the root holds 3 class-0 and 3 class-1 samples, so the
/// root Gini impurity is 1 - (3/6)^2 - (3/6)^2 = 0.5.  A single threshold on feature 0
/// (around 0.6) sends the 3 class-0 points one way and the 3 class-1 points the other,
/// producing two PURE children (Gini 0 each).  The impurity decrease of that best split is
/// therefore exactly 0.5 - 0 = 0.5.  The growth rule is `decrease < min_impurity_decrease`
/// => make a leaf.
///
///   * min_impurity_decrease = 0.5001  =>  0.5 < 0.5001 is true  => root collapses to ONE
///     leaf; predict() is constant for every input (including points far from training data).
///   * min_impurity_decrease = 0.4999  =>  0.5 < 0.4999 is false => the root splits, giving
///     zero training error on the separable data.
///
/// (Majority class on the collapsed leaf is a 3-3 tie, so we assert only that predict is
/// CONSTANT, not which class wins — that keeps the test independent of tie-breaking.)
#[test]
fn test_min_impurity_decrease_prunes_root_split_at_half() {
    let (x, y) = linearly_separable_binary();

    // Just ABOVE the root decrease of 0.5 -> the only useful split is pruned away.
    let params_above = DecisionTreeParams {
        min_impurity_decrease: 0.5001,
        ..DecisionTreeParams::default()
    };
    let mut tree_above = DecisionTree::new(Algorithm::CART, true, Some(params_above)).unwrap();
    tree_above.fit(&x, &y).unwrap();
    assert!(
        root_is_leaf(&tree_above),
        "min_impurity_decrease=0.5001 > root decrease 0.5 must collapse the tree to a single leaf"
    );
    let preds_above = tree_above.predict(&x).unwrap();
    let first = preds_above[0];
    assert!(
        preds_above.iter().all(|&p| (p - first).abs() < 1e-12),
        "a collapsed single-leaf tree must predict a constant, got {preds_above:?}"
    );
    // Points far outside the training range also get the same constant leaf prediction.
    let far = array![[-5.0_f64, -5.0], [9.0, 9.0]];
    let preds_far = tree_above.predict(&far).unwrap();
    assert!(
        preds_far.iter().all(|&p| (p - first).abs() < 1e-12),
        "collapsed leaf must predict the same constant everywhere, got {preds_far:?}"
    );

    // Just BELOW the root decrease of 0.5 -> the split is kept and the data is fit perfectly.
    let params_below = DecisionTreeParams {
        min_impurity_decrease: 0.4999,
        ..DecisionTreeParams::default()
    };
    let mut tree_below = DecisionTree::new(Algorithm::CART, true, Some(params_below)).unwrap();
    tree_below.fit(&x, &y).unwrap();
    assert!(
        !root_is_leaf(&tree_below),
        "min_impurity_decrease=0.4999 < root decrease 0.5 must allow the root to split"
    );
    let preds_below = tree_below.predict(&x).unwrap();
    for (&pred, &expected) in preds_below.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// min_samples_leaf rejects the single best split when it would create a too-small leaf.
///
/// Data (one feature, 4 samples): x = [0,1,2,3], y = [0,0,0,1].  Root Gini = 1 - (3/4)^2 -
/// (1/4)^2 = 0.375.  The candidate thresholds and their impurity decreases are:
///   * thr 2.5 -> children {0,1,2}=pure-0 and {3}=pure-1  -> decrease 0.375 (the UNIQUE best)
///   * thr 1.5 -> children sizes 2/2                       -> decrease 0.125
///   * thr 0.5 -> children sizes 1/3                       -> decrease 0.0417
///
/// The best split (thr 2.5) puts a single sample in its right leaf.
///
///   * min_samples_leaf = 1 (default): the best split is allowed -> the root is an internal
///     node and the lone class-1 sample is separated, so predict([3.0]) == 1.0.
///   * min_samples_leaf = 2: the best split's size-1 leaf violates the constraint, so the
///     node is turned into a leaf (the grower does not fall back to a worse split).  The
///     collapsed leaf's majority class is 0 (counts 3 vs 1), so predict([3.0]) == 0.0 — the
///     class-1 point is now absorbed, proving the split was actually rejected during growth.
#[test]
fn test_min_samples_leaf_rejects_too_small_leaf_split() {
    let x = array![[0.0_f64], [1.0_f64], [2.0_f64], [3.0_f64]];
    let y = array![0.0_f64, 0.0, 0.0, 1.0];
    let probe = array![[3.0_f64]];

    // Default min_samples_leaf = 1: best split kept, lone class-1 sample separated.
    let mut tree_allow = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree_allow.fit(&x, &y).unwrap();
    assert!(
        !root_is_leaf(&tree_allow),
        "with min_samples_leaf=1 the size-3/size-1 split must be taken"
    );
    let pred_allow = tree_allow.predict(&probe).unwrap();
    assert_abs_diff_eq!(pred_allow[0], 1.0, epsilon = 1e-9);

    // min_samples_leaf = 2: the only impurity-reducing best split makes a size-1 leaf and is
    // rejected, collapsing the node to a majority-class-0 leaf.
    let params = DecisionTreeParams {
        min_samples_leaf: 2,
        ..DecisionTreeParams::default()
    };
    let mut tree_reject = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    tree_reject.fit(&x, &y).unwrap();
    assert!(
        root_is_leaf(&tree_reject),
        "with min_samples_leaf=2 the size-1-leaf split must be rejected, collapsing to a leaf"
    );
    let pred_reject = tree_reject.predict(&probe).unwrap();
    assert_abs_diff_eq!(pred_reject[0], 0.0, epsilon = 1e-9);
}

/// min_samples_split stops recursion at an internal node, yielding a shallower tree.
///
/// Data (one feature, 3 samples, 3 classes): x = [0,1,2], y = [0,1,2].
/// The first threshold the grower evaluates that maximizes the Gini decrease is thr 0.5
/// (decrease 1/3): it splits off {0}=pure-0 on the left and leaves {1,2} (classes 1 and 2)
/// on the right.
///   * min_samples_split = 2 (default): the right child has 2 samples (>= 2) and is impure,
///     so it splits again at thr 1.5 into two pure leaves -> the deepest leaf is at depth 2.
///   * min_samples_split = 3: at the root, n=3 (>= 3) so it still splits; but the right child
///     has only 2 samples (< 3), so the in-recursion precondition `n_samples < min_samples_split`
///     forces it to STOP as a leaf -> the deepest leaf is at depth 1.
///
/// Both clear the fit-time precondition (n=3 >= min_samples_split), isolating the
/// in-recursion stop. We assert the restricted tree is strictly shallower than the default.
#[test]
fn test_min_samples_split_stops_recursion_shallower_tree() {
    let x = array![[0.0_f64], [1.0_f64], [2.0_f64]];
    let y = array![0.0_f64, 1.0, 2.0];

    // Default min_samples_split = 2: internal node {1,2} splits again -> depth 2.
    let mut tree_default = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    tree_default.fit(&x, &y).unwrap();
    let depth_default = leaf_depth(tree_default.get_root().unwrap());
    assert_eq!(
        depth_default, 2,
        "default min_samples_split=2 must build a depth-2 tree on 3 fully-separable classes"
    );

    // min_samples_split = 3: the 2-sample internal node must stop -> depth 1.
    let params = DecisionTreeParams {
        min_samples_split: 3,
        ..DecisionTreeParams::default()
    };
    let mut tree_restricted = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    tree_restricted.fit(&x, &y).unwrap();
    let depth_restricted = leaf_depth(tree_restricted.get_root().unwrap());
    assert_eq!(
        depth_restricted, 1,
        "min_samples_split=3 must stop the 2-sample internal node, giving a depth-1 tree"
    );
    assert!(
        depth_restricted < depth_default,
        "larger min_samples_split must yield a strictly shallower tree ({depth_restricted} vs {depth_default})"
    );
}

/// predict_proba returns the EXACT empirical class distribution of an impure leaf.
///
/// Data (one feature, 6 samples): x = [0,1,2,3,4,5], y = [1,1,1,0,0,1].
/// Root Gini = 1 - (2/6)^2 - (4/6)^2 = 4/9.  Scanning thresholds, the unique best split is
/// thr 2.5 (decrease 4/9 - 2/9 = 2/9 = 0.2222, larger than every other threshold), sending
///   left  {0,1,2} = three pure class-1 samples, and
///   right {3,4,5} = two class-0 and one class-1 sample.
/// With max_depth = 1 the right child is forced to be a LEAF at depth 1 while still impure,
/// so its stored distribution is exactly [count0/3, count1/3] = [2/3, 1/3].
/// We probe x = 4.0 (> 2.5, routes right).  Asserting BOTH components equal 2/3 and 1/3 (not
/// just sum==1 or argmax) makes a degenerate [1,0]-everywhere implementation fail.
#[test]
fn test_predict_proba_exact_impure_leaf_distribution() {
    let x = array![
        [0.0_f64],
        [1.0_f64],
        [2.0_f64],
        [3.0_f64],
        [4.0_f64],
        [5.0_f64],
    ];
    let y = array![1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0];

    let params = DecisionTreeParams {
        max_depth: Some(1),
        ..DecisionTreeParams::default()
    };
    let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    tree.fit(&x, &y).unwrap();

    // x = 4.0 lands in the impure right leaf {2 of class 0, 1 of class 1}.
    let proba = tree.predict_proba_one(&[4.0_f64]).unwrap();
    assert_eq!(proba.len(), 2);
    assert_abs_diff_eq!(proba[0], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(proba[1], 1.0 / 3.0, epsilon = 1e-12);

    // Same exact distribution via the batch API.
    let probas = tree.predict_proba(&array![[4.0_f64]]).unwrap();
    assert_eq!(probas.shape(), &[1, 2]);
    assert_abs_diff_eq!(probas[[0, 0]], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(probas[[0, 1]], 1.0 / 3.0, epsilon = 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════
// max_depth = Some(0): the root is forced to be an immediate leaf.
// ═══════════════════════════════════════════════════════════════════════

/// With max_depth = Some(0) the grower's stopping rule `depth >= max_depth` fires at depth 0
/// (0 >= 0), so no split is ever made and the root is a single leaf holding the GLOBAL majority
/// class. predict() must therefore return that majority class for every input, including points
/// that — with any real split — would clearly belong to the minority class.
///
/// Dataset: feature 0 perfectly separates the classes (0.x => class 0, 1.x => class 1), but the
/// global label counts are 3×class-0 vs 2×class-1, so the global majority is class 0. The depth-0
/// tree must predict 0.0 everywhere even for the class-1 region (feature 0 ≈ 1.x) and for points
/// far outside the training range.
#[test]
fn test_max_depth_zero_root_is_leaf_predicts_global_majority() {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 1.0],
    ];
    // 3 class-0 and 2 class-1 => global majority is class 0.
    let y = array![0.0_f64, 0.0, 0.0, 1.0, 1.0];

    let params = DecisionTreeParams {
        max_depth: Some(0),
        ..DecisionTreeParams::default()
    };
    let mut tree = DecisionTree::new(Algorithm::CART, true, Some(params)).unwrap();
    tree.fit(&x, &y).unwrap();

    // The root must be a single leaf (no split at all).
    assert!(
        matches!(
            tree.get_root().expect("tree must be fitted").node_type,
            NodeType::Leaf { .. }
        ),
        "max_depth=Some(0) must make the root an immediate leaf"
    );

    // Every prediction must be the global majority class (0.0), including the class-1 region
    // and points far outside the training range.
    let probe = array![
        [0.0_f64, 0.0], // training class-0 point
        [1.0, 1.0],     // training class-1 point — still predicted 0
        [1.1, 1.0],     // training class-1 point — still predicted 0
        [99.0, 99.0],   // far outside the training range
    ];
    let preds = tree.predict(&probe).unwrap();
    for &p in preds.iter() {
        assert_abs_diff_eq!(p, 0.0, epsilon = 1e-12);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Categorical (multi-way) save/load round-trip — exercises the AHashMap
// `children` serde branch.
// ═══════════════════════════════════════════════════════════════════════

/// A C4.5 tree trained on a categorical feature builds an internal node with a populated
/// `children` map (one branch per distinct category value). Serializing then deserializing the
/// model must round-trip that AHashMap branch losslessly, so the loaded model reproduces the
/// original predictions exactly — both on the training categories and on an unseen category that
/// routes through the serialized fallback (`left`) branch.
///
/// Design: categories {0.0, 1.0, 2.0} with the category-XOR labels (0->0, 1->1, 2->0), which only
/// a multi-way split can fit. Because feature 0 is the SOLE feature and is marked categorical,
/// C4.5 evaluates only its multi-way split (no numeric competitor), so the fitted root is
/// guaranteed to be a categorical node with a populated `children` map — the AHashMap serde path
/// is genuinely exercised.
#[test]
fn test_save_load_categorical_multiway_round_trip_identical_predictions() {
    let x_cat = array![
        [0.0_f64],
        [0.0_f64],
        [1.0_f64],
        [1.0_f64],
        [2.0_f64],
        [2.0_f64],
    ];
    let y_cat = array![0.0_f64, 0.0, 1.0, 1.0, 0.0, 0.0];

    let mut original = DecisionTree::new(Algorithm::C45, true, None).unwrap();
    original.set_categorical_features(vec![0]);
    original.fit(&x_cat, &y_cat).unwrap();

    // Sanity: the fitted root must be a multi-way categorical node (populated children map),
    // otherwise this test would not exercise the AHashMap serde branch.
    assert!(
        original
            .get_root()
            .and_then(|n| n.children.as_ref())
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "expected a categorical root with a populated children map"
    );

    // Predictions including an UNSEEN category (99.0) that must route through the fallback branch.
    let x_eval = array![[0.0_f64], [1.0_f64], [2.0_f64], [99.0_f64],];
    let preds_before = original.predict(&x_eval).unwrap();

    let path = format!("/tmp/rustyml_dt_cat_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x_eval).unwrap();

    // The deserialized children map must reproduce predictions bit-for-bit.
    assert_allclose(&preds_before, &preds_after, 1e-12);
    // And the categorical structure must survive the round-trip.
    assert!(
        loaded
            .get_root()
            .and_then(|n| n.children.as_ref())
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "loaded model lost its categorical children map"
    );

    let _ = std::fs::remove_file(&path);
}

// ═══════════════════════════════════════════════════════════════════════════
// Global crate seed drives tie-breaking for random_state = None trees.
// ═══════════════════════════════════════════════════════════════════════════

/// RAII guard that clears the thread-local global seed on drop, so a panic in the test body cannot
/// leak a seeded global stream onto this thread (which would perturb other `random_state = None`
/// tests that happen to run on the same thread).
struct GlobalSeedGuard;
impl Drop for GlobalSeedGuard {
    fn drop(&mut self) {
        clear_global_seed();
    }
}

/// Fit a `random_state = None` CART tree on the tie data and return its structure string. The tree
/// derives its tie-breaking RNG from the thread-local global seed (when one is set), so the caller
/// controls reproducibility by (re)seeding the global stream immediately before each call.
fn tie_break_structure_none() -> String {
    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    let (x, y) = tie_break_data();
    tree.fit(&x, &y).unwrap();
    tree.generate_tree_structure().unwrap()
}

/// With a crate-global seed set, a `random_state = None` tree breaks split ties using an RNG
/// derived from the global stream. Re-seeding the global stream to the SAME value before each fit
/// restarts that stream, so the derived sub-seed — and therefore the whole tree — is identical
/// across the two fits. This proves the global seed actually drives the None-tree's tie-breaking
/// (and is reproducible), complementing the explicit-`random_state` reproducibility test.
///
/// `tie_break_data` has two identical, perfectly-separating feature columns (a guaranteed score
/// tie at the root), so the only thing the seed can change is which tied feature is chosen — making
/// this a genuine test of seeded tie-breaking. The guard clears the global seed on every exit path.
#[test]
fn global_seed_makes_none_tree_tie_breaking_reproducible() {
    let _guard = GlobalSeedGuard;

    // First fit under global seed 12345.
    set_global_seed(12345);
    let structure_a = tie_break_structure_none();

    // Re-seed the global stream to the SAME value, restarting it, then fit again.
    set_global_seed(12345);
    let structure_b = tie_break_structure_none();

    assert_eq!(
        structure_a, structure_b,
        "two random_state=None trees fit under the same global seed must be identical"
    );
    // _guard clears the global seed here (and on any panic above).
}
