//! Integration tests for `rustyml::machine_learning::tree::decision_tree`
//!
//! Expected values are derived from problem design or closed-form results,
//! never by running the model and recording its output

use crate::common::assert_allclose;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::{Error, TreeError};
use rustyml::machine_learning::{Algorithm, DecisionTree};
use rustyml::machine_learning::{Node, NodeType};
use rustyml::{clear_global_seed, set_global_seed};

// Helper: a tiny linearly-separable binary dataset; feature 0 is 0.x for class 0
// and 1.x for class 1, so any default-param tree reaches zero training error
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

// Constructor validation

/// Default params are accepted for CART classifier
#[test]
fn test_constructor_default_params_cart_classifier() {
    let tree = DecisionTree::new(Algorithm::CART, true);
    assert!(
        tree.is_ok(),
        "CART classifier with default params should succeed"
    );
    let tree = tree.unwrap();
    assert_eq!(tree.get_algorithm(), Algorithm::CART);
    assert!(tree.get_is_classifier());
    // Before fitting, no root and no class information
    assert!(tree.get_root().is_none());
    assert_eq!(tree.get_n_classes(), None);
    assert_eq!(tree.get_n_features(), 0);
}

/// ID3 with is_classifier=false must return InvalidInput
#[test]
fn test_constructor_id3_regression_returns_invalid_input() {
    let err = DecisionTree::new(Algorithm::ID3, false).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput, got {err:?}"
    );
}

/// C4.5 with is_classifier=false must return InvalidInput
#[test]
fn test_constructor_c45_regression_returns_invalid_input() {
    let err = DecisionTree::new(Algorithm::C45, false).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput, got {err:?}"
    );
}

/// min_samples_split = 1 (less than 2) must return InvalidParameter
#[test]
fn test_constructor_min_samples_split_too_small() {
    let err = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_samples_split(1)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_samples_leaf = 0 must return InvalidParameter
#[test]
fn test_constructor_min_samples_leaf_zero() {
    let err = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_samples_leaf(0)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_samples_leaf > min_samples_split is rejected at fit time (the two are set
/// independently through the builder, so the cross-field constraint cannot be checked earlier)
#[test]
fn test_min_samples_leaf_greater_than_split_rejected_at_fit() {
    let mut tree = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_samples_split(2)
        .unwrap()
        .with_min_samples_leaf(3)
        .unwrap();
    let x = array![[0.0_f64], [1.0], [2.0]];
    let y = array![0.0_f64, 1.0, 0.0];
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_impurity_decrease = -0.1 (negative) must return InvalidParameter
#[test]
fn test_constructor_negative_min_impurity_decrease() {
    let err = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_impurity_decrease(-0.1)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_impurity_decrease = f64::NAN must return InvalidParameter
#[test]
fn test_constructor_nan_min_impurity_decrease() {
    let err = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_impurity_decrease(f64::NAN)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// min_impurity_decrease = f64::INFINITY must return InvalidParameter
#[test]
fn test_constructor_infinite_min_impurity_decrease() {
    let err = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_impurity_decrease(f64::INFINITY)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "Expected InvalidParameter, got {err:?}"
    );
}

/// Custom params are stored and returned by the getters
#[test]
fn test_constructor_custom_params_stored_correctly() {
    let tree = DecisionTree::new(Algorithm::ID3, true)
        .unwrap()
        .with_max_depth(3)
        .with_min_samples_split(4)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap()
        .with_min_impurity_decrease(0.01)
        .unwrap()
        .with_random_state(42);
    let stored = tree.get_parameters();
    assert_eq!(stored.max_depth, Some(3));
    assert_eq!(stored.min_samples_split, 4);
    assert_eq!(stored.min_samples_leaf, 2);
    assert_abs_diff_eq!(stored.min_impurity_decrease, 0.01, epsilon = 1e-12);
    assert_eq!(stored.random_state, Some(42));
}

// fit validation

/// Negative class labels must be rejected by fit() for a classifier
#[test]
fn test_fit_negative_labels_rejected() {
    let x = array![[0.0_f64, 1.0], [1.0, 0.0]];
    let y = array![-1.0_f64, 1.0]; // -1 is invalid
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput for negative label, got {err:?}"
    );
}

/// Fractional class labels must be rejected by fit() for a classifier
#[test]
fn test_fit_fractional_labels_rejected() {
    let x = array![[0.0_f64, 1.0], [1.0, 0.0]];
    let y = array![0.5_f64, 1.0]; // 0.5 is not an integer
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput for fractional label, got {err:?}"
    );
}

/// fit() with fewer samples than min_samples_split returns InvalidInput
#[test]
fn test_fit_too_few_samples_for_min_samples_split() {
    // Only 2 samples but min_samples_split = 5
    let x = array![[0.0_f64], [1.0_f64]];
    let y = array![0.0_f64, 1.0_f64];
    let mut tree = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_samples_split(5)
        .unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "Expected InvalidInput when n_samples < min_samples_split, got {err:?}"
    );
}

/// fit() with NaN in x returns NonFinite
#[test]
fn test_fit_nan_in_x_rejected() {
    let x = array![[0.0_f64, f64::NAN], [1.0, 0.0], [2.0, 1.0]];
    let y = array![0.0_f64, 1.0, 0.0];
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let err = tree.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "Expected NonFinite for NaN in input, got {err:?}"
    );
}

// NotFitted error paths

/// predict() before fit returns NotFitted
#[test]
fn test_predict_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let x = array![[0.0_f64, 0.0]];
    let err = tree.predict(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// predict_one() before fit returns NotFitted
#[test]
fn test_predict_one_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let err = tree.predict_one(&[0.0, 0.0]).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// predict_proba() before fit returns NotFitted
#[test]
fn test_predict_proba_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let x = array![[0.0_f64, 0.0]];
    let err = tree.predict_proba(&x).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// predict_proba_one() before fit returns NotFitted
#[test]
fn test_predict_proba_one_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let err = tree.predict_proba_one(&[0.0, 0.0]).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

/// generate_tree_structure() before fit returns NotFitted
#[test]
fn test_generate_tree_structure_before_fit_not_fitted() {
    let tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let err = tree.generate_tree_structure().unwrap_err();
    assert!(
        matches!(err, Error::NotFitted("DecisionTree")),
        "Expected NotFitted, got {err:?}"
    );
}

// DimensionMismatch error paths

/// predict() with wrong number of features returns DimensionMismatch
#[test]
fn test_predict_wrong_n_features_dimension_mismatch() {
    let (x, y) = linearly_separable_binary();
    // x has 2 features; n_features after fit == 2
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    // Supply a matrix with 3 features instead of 2
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

/// predict_one() with wrong number of features returns DimensionMismatch
#[test]
fn test_predict_one_wrong_n_features_dimension_mismatch() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
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

/// predict_proba() with wrong number of features returns DimensionMismatch
#[test]
fn test_predict_proba_wrong_n_features_dimension_mismatch() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
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

// NotClassificationTree error

/// predict_proba() on a CART regressor returns Tree(NotClassificationTree)
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
    let mut tree = DecisionTree::new(Algorithm::CART, false).unwrap();
    tree.fit(&x, &y).unwrap();

    let err = tree.predict_proba(&x).unwrap_err();
    assert!(
        matches!(err, Error::Tree(TreeError::NotClassificationTree)),
        "Expected NotClassificationTree, got {err:?}"
    );
}

/// predict_proba_one() on a CART regressor (before fit) returns NotClassificationTree immediately
#[test]
fn test_predict_proba_one_on_unfitted_regressor_returns_not_classification_tree() {
    let tree = DecisionTree::new(Algorithm::CART, false).unwrap();
    // predict_proba_one checks is_classifier first, then NotFitted
    let err = tree.predict_proba_one(&[0.0]).unwrap_err();
    assert!(
        matches!(err, Error::Tree(TreeError::NotClassificationTree)),
        "Expected NotClassificationTree, got {err:?}"
    );
}

// CART classifier - fit and predict correctness

/// CART classifier on linearly-separable binary data achieves zero training error
#[test]
fn test_cart_classifier_zero_training_error() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    // After fit, n_features = 2, n_classes = Some(2), root is Some
    assert_eq!(tree.get_n_features(), 2);
    assert_eq!(tree.get_n_classes(), Some(2));
    assert!(tree.get_root().is_some());

    let preds = tree.predict(&x).unwrap();
    // Known true classes come from problem design; the tree must reproduce them exactly
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// fit_predict produces the same result as calling fit and predict separately
#[test]
fn test_cart_fit_predict_equals_fit_then_predict() {
    let (x, y) = linearly_separable_binary();

    // fit_predict path
    let mut tree1 = DecisionTree::new(Algorithm::CART, true).unwrap();
    let preds_fp = tree1.fit_predict(&x, &y).unwrap();

    // fit then predict path
    let mut tree2 = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree2.fit(&x, &y).unwrap();
    let preds_sep = tree2.predict(&x).unwrap();

    assert_allclose(&preds_fp, &preds_sep, 1e-12);
}

// ID3 classifier - fit and predict correctness

/// ID3 classifier on linearly-separable binary data achieves zero training error
#[test]
fn test_id3_classifier_zero_training_error() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::ID3, true).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// C4.5 classifier - fit and predict correctness

/// C4.5 classifier on linearly-separable binary data achieves zero training error
#[test]
fn test_c45_classifier_zero_training_error() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::C45, true).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// Three-class (multi-class) classification

/// CART on a clearly-separated 3-class dataset achieves zero training error
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

    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    // n_classes must be 3 (max label is 2, so 3 classes = 0..=2)
    assert_eq!(tree.get_n_classes(), Some(3));

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// ID3 on the same 3-class dataset achieves zero training error
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

    let mut tree = DecisionTree::new(Algorithm::ID3, true).unwrap();
    tree.fit(&x, &y).unwrap();
    assert_eq!(tree.get_n_classes(), Some(3));

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// C4.5 on the same 3-class dataset achieves zero training error
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

    let mut tree = DecisionTree::new(Algorithm::C45, true).unwrap();
    tree.fit(&x, &y).unwrap();
    assert_eq!(tree.get_n_classes(), Some(3));

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// predict_proba correctness

/// predict_proba rows sum to 1.0 and argmax(row) == predict for every sample
#[test]
fn test_predict_proba_rows_sum_to_one_and_argmax_matches_predict() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
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

/// predict_proba_one row sums to 1.0 and argmax matches predict_one
#[test]
fn test_predict_proba_one_sum_and_argmax() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    // The first training point belongs to class 0 (by design)
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

/// On a pure leaf, predict_proba gives probability 1.0 for the true class and
/// 0.0 for all others
#[test]
fn test_predict_proba_pure_leaf_gives_one_hot() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    // With full depth the leaf covering the class-0 region holds a [1.0, 0.0]
    // distribution
    let probas = tree.predict_proba(&x).unwrap();

    // Rows 0-2 are class 0; their probability for class 1 must be 0.0
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

// CART regression - leaf mean correctness

/// CART regressor on a step function: predicted values equal the group means
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

    let mut tree = DecisionTree::new(Algorithm::CART, false).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();

    // Group 0 (x in 0..=2): all predictions must equal the group mean 1.0
    for i in 0..3 {
        assert_abs_diff_eq!(preds[i], 1.0, epsilon = 1e-9);
    }
    // Group 1 (x in 10..=12): all predictions must equal the group mean 10.0
    for i in 3..6 {
        assert_abs_diff_eq!(preds[i], 10.0, epsilon = 1e-9);
    }
}

/// CART regressor memorizes training data perfectly when each sample has a unique x
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

    let mut tree = DecisionTree::new(Algorithm::CART, false).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// max_depth parameter

/// max_depth=1 cannot fit XOR with a single split, while unlimited depth does
/// at least as well
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
    let mut tree_shallow = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_max_depth(1);
    tree_shallow.fit(&x_xor, &y_xor).unwrap();
    let preds_shallow = tree_shallow.predict(&x_xor).unwrap();

    // For XOR at least one must be wrong with depth=1
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

    // Unlimited depth must do at least as well as depth-1 on XOR; a greedy CART
    // cannot necessarily memorize XOR, since at the root no single-feature split reduces impurity
    let mut tree_deep = DecisionTree::new(Algorithm::CART, true).unwrap();
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

/// max_depth=1 still achieves zero training error on linearly-separable data,
/// where one split suffices
#[test]
fn test_max_depth_1_suffices_for_linearly_separable_data() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_max_depth(1);
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

// All-same-label (pure root) edge case

/// When every training sample has the same label the tree creates a single pure
/// leaf at depth 0 and predict returns that label for all inputs
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

    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for pred in preds.iter() {
        assert_abs_diff_eq!(*pred, 1.0, epsilon = 1e-9);
    }
}

/// Same edge case for a CART regressor: all-same y -> leaf mean == that value
#[test]
fn test_pure_regression_set_creates_pure_root_leaf() {
    let x = array![[0.0_f64], [1.0_f64], [2.0_f64], [3.0_f64]];
    let y = array![5.5_f64, 5.5, 5.5, 5.5];

    let mut tree = DecisionTree::new(Algorithm::CART, false).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for pred in preds.iter() {
        assert_abs_diff_eq!(*pred, 5.5, epsilon = 1e-9);
    }
}

// Categorical features (ID3 and C4.5 multi-way splits)

/// C4.5 with a categorical feature separates a category-XOR pattern that no
/// single binary numeric threshold can resolve
#[test]
fn test_c45_categorical_multiway_split_zero_training_error() {
    // Each sample has one categorical feature
    let x_cat = array![
        [0.0_f64],
        [0.0_f64],
        [1.0_f64],
        [1.0_f64],
        [2.0_f64],
        [2.0_f64],
    ];
    let y_cat = array![0.0_f64, 0.0, 1.0, 1.0, 0.0, 0.0];

    let mut tree = DecisionTree::new(Algorithm::C45, true).unwrap();
    tree.set_categorical_features(vec![0]);
    tree.fit(&x_cat, &y_cat).unwrap();

    let preds = tree.predict(&x_cat).unwrap();
    for (&pred, &expected) in preds.iter().zip(y_cat.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// ID3 with categorical features also produces a multi-way split that fits the
/// category-XOR pattern
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

    let mut tree = DecisionTree::new(Algorithm::ID3, true).unwrap();
    tree.set_categorical_features(vec![0]);
    tree.fit(&x_cat, &y_cat).unwrap();

    let preds = tree.predict(&x_cat).unwrap();
    for (&pred, &expected) in preds.iter().zip(y_cat.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// CART ignores the categorical designation (it always uses binary splits) and
/// still achieves zero training error on linearly-separable data
#[test]
fn test_cart_ignores_categorical_feature_designation() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    // Mark feature 0 as categorical - CART must ignore this
    tree.set_categorical_features(vec![0]);
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for (&pred, &expected) in preds.iter().zip(y.iter()) {
        assert_abs_diff_eq!(pred, expected, epsilon = 1e-9);
    }
}

/// Unseen category value at predict time falls back to the default leaf without
/// error
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

    let mut tree = DecisionTree::new(Algorithm::C45, true).unwrap();
    tree.set_categorical_features(vec![0]);
    tree.fit(&x_cat, &y_cat).unwrap();

    // Category 99.0 was never seen during training
    let x_unseen = array![[99.0_f64]];
    let result = tree.predict(&x_unseen);
    assert!(
        result.is_ok(),
        "Unseen category should use fallback leaf, not error: {result:?}"
    );
    // The fallback prediction must be a valid class label (0 or 1)
    let pred = result.unwrap()[0];
    assert!(
        pred == 0.0 || pred == 1.0,
        "Fallback prediction {pred} is not a valid class label"
    );
}

/// get_categorical_features reflects what was set
#[test]
fn test_get_categorical_features_reflects_set() {
    let mut tree = DecisionTree::new(Algorithm::ID3, true).unwrap();
    assert_eq!(tree.get_categorical_features(), &[] as &[usize]);

    tree.set_categorical_features(vec![0, 2]);
    assert_eq!(tree.get_categorical_features(), &[0_usize, 2]);
}

// generate_tree_structure

/// generate_tree_structure() after fit returns a non-empty string
#[test]
fn test_generate_tree_structure_returns_nonempty_string_after_fit() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();
    let structure = tree.generate_tree_structure().unwrap();
    assert!(!structure.is_empty(), "Tree structure should be non-empty");
    assert!(
        structure.contains("Decision Tree Structure"),
        "Expected header in structure output"
    );
}

// Save / load round-trip

/// save_to_path / load_from_path yields identical predictions on the training
/// set and a held-out test set
#[test]
fn test_save_load_round_trip_identical_predictions() {
    let (x_train, y_train) = linearly_separable_binary();

    let mut original = DecisionTree::new(Algorithm::CART, true).unwrap();
    original.fit(&x_train, &y_train).unwrap();

    let preds_before = original.predict(&x_train).unwrap();

    // Temp file path unique per process
    let path = format!("/tmp/rustyml_dt_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x_train).unwrap();

    // Predictions must be bit-for-bit identical (same floating-point values)
    assert_allclose(&preds_before, &preds_after, 1e-12);

    // Held-out test points: x[0] < 0.5 -> class 0, x[0] > 0.5 -> class 1
    let x_test = array![
        [0.05_f64, 0.0],
        [0.15_f64, 0.05],
        [1.05_f64, 1.0],
        [1.15_f64, 1.0],
    ];
    let expected_test = array![0.0_f64, 0.0, 1.0, 1.0];

    let preds_test_loaded = loaded.predict(&x_test).unwrap();
    assert_allclose(&preds_test_loaded, &expected_test, 1e-9);

    let _ = std::fs::remove_file(&path);
}

/// Save/load for a C4.5 classifier preserves predictions
#[test]
fn test_save_load_c45_classifier_round_trip() {
    let (x, y) = linearly_separable_binary();
    let mut original = DecisionTree::new(Algorithm::C45, true).unwrap();
    original.fit(&x, &y).unwrap();

    let preds_before = original.predict(&x).unwrap();

    let path = format!("/tmp/rustyml_c45_dt_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x).unwrap();

    assert_allclose(&preds_before, &preds_after, 1e-12);

    let _ = std::fs::remove_file(&path);
}

/// Save/load for a CART regressor preserves predictions
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

    let mut original = DecisionTree::new(Algorithm::CART, false).unwrap();
    original.fit(&x, &y).unwrap();

    let preds_before = original.predict(&x).unwrap();

    let path = format!("/tmp/rustyml_cart_reg_dt_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x).unwrap();

    assert_allclose(&preds_before, &preds_after, 1e-12);

    let _ = std::fs::remove_file(&path);
}

// predict labels are in the expected domain {0.0, 1.0, ...}

/// For a binary classifier every prediction must be either 0.0 or 1.0
#[test]
fn test_predict_outputs_are_valid_class_labels() {
    let (x, y) = linearly_separable_binary();
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).unwrap();

    let preds = tree.predict(&x).unwrap();
    for pred in preds.iter() {
        assert!(
            *pred == 0.0 || *pred == 1.0,
            "Prediction {pred} is not a valid binary class label"
        );
    }
}

/// All three algorithms produce predictions in the label domain {0, 1, 2} for a
/// 3-class problem
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
        let mut tree = DecisionTree::new(algo, true).unwrap();
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

// random_state: seeded tie-breaking among equally-scoring splits

/// Data with a guaranteed split tie: features 0 and 1 are identical columns that
/// perfectly separate the two classes, while feature 2 is uninformative (zero gain)
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

/// Fit a CART classifier on the tie data with the given `random_state` and return
/// its tree-structure string (which encodes the chosen split feature at each node)
fn tie_break_structure(random_state: Option<u64>) -> String {
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    if let Some(seed) = random_state {
        tree = tree.with_random_state(seed);
    }
    let (x, y) = tie_break_data();
    tree.fit(&x, &y).unwrap();
    tree.generate_tree_structure().unwrap()
}

/// Same explicit seed => identical tree, even when ties are broken randomly
#[test]
fn random_state_same_seed_is_reproducible() {
    assert_eq!(
        tie_break_structure(Some(7)),
        tie_break_structure(Some(7)),
        "the same random_state must reproduce the same tree"
    );
}

/// `random_state = None` (with no global seed) => deterministic tie-breaking, so
/// repeated fits are identical
#[test]
fn random_state_none_is_deterministic() {
    assert_eq!(
        tie_break_structure(None),
        tie_break_structure(None),
        "random_state=None must be deterministic"
    );
}

/// Seeded tie-breaking varies the chosen split: across a range of seeds the root
/// does not always split on the same one of the two tied features
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

// Fit-time pruning: these tests prove min_impurity_decrease / min_samples_leaf /
// min_samples_split change the grown tree, inspected via public `get_root()`

/// Number of edges from `node` to its deepest leaf (a leaf has depth 0; an
/// internal node is 1 + the max over its children)
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

/// Whether the fitted tree's root is a single leaf (no split was made at all)
fn root_is_leaf(tree: &DecisionTree) -> bool {
    matches!(
        tree.get_root().expect("tree must be fitted").node_type,
        NodeType::Leaf { .. }
    )
}

/// min_impurity_decrease prunes the root split exactly at the Gini-decrease
/// threshold (root decrease is 0.5; 0.5001 collapses to one leaf, 0.4999 splits)
#[test]
fn test_min_impurity_decrease_prunes_root_split_at_half() {
    let (x, y) = linearly_separable_binary();

    // Just above the root decrease of 0.5 -> the only useful split is pruned away
    let mut tree_above = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_impurity_decrease(0.5001)
        .unwrap();
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
    // Points far outside the training range also get the same constant leaf prediction
    let far = array![[-5.0_f64, -5.0], [9.0, 9.0]];
    let preds_far = tree_above.predict(&far).unwrap();
    assert!(
        preds_far.iter().all(|&p| (p - first).abs() < 1e-12),
        "collapsed leaf must predict the same constant everywhere, got {preds_far:?}"
    );

    // Just below the root decrease of 0.5 -> the split is kept and the data is fit perfectly
    let mut tree_below = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_impurity_decrease(0.4999)
        .unwrap();
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

/// min_samples_leaf rejects the single best split when it would create a too-small
/// leaf (size-1 leaf with min_samples_leaf=2 collapses the node instead)
#[test]
fn test_min_samples_leaf_rejects_too_small_leaf_split() {
    let x = array![[0.0_f64], [1.0_f64], [2.0_f64], [3.0_f64]];
    let y = array![0.0_f64, 0.0, 0.0, 1.0];
    let probe = array![[3.0_f64]];

    // Default min_samples_leaf = 1: best split kept, lone class-1 sample separated
    let mut tree_allow = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree_allow.fit(&x, &y).unwrap();
    assert!(
        !root_is_leaf(&tree_allow),
        "with min_samples_leaf=1 the size-3/size-1 split must be taken"
    );
    let pred_allow = tree_allow.predict(&probe).unwrap();
    assert_abs_diff_eq!(pred_allow[0], 1.0, epsilon = 1e-9);

    // min_samples_leaf = 2: the only impurity-reducing best split makes a size-1
    // leaf and is rejected, collapsing the node to a majority-class-0 leaf
    let mut tree_reject = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_samples_leaf(2)
        .unwrap();
    tree_reject.fit(&x, &y).unwrap();
    assert!(
        root_is_leaf(&tree_reject),
        "with min_samples_leaf=2 the size-1-leaf split must be rejected, collapsing to a leaf"
    );
    let pred_reject = tree_reject.predict(&probe).unwrap();
    assert_abs_diff_eq!(pred_reject[0], 0.0, epsilon = 1e-9);
}

/// min_samples_split stops recursion at an internal node, yielding a strictly
/// shallower tree (depth 1 with min_samples_split=3 vs depth 2 by default)
#[test]
fn test_min_samples_split_stops_recursion_shallower_tree() {
    let x = array![[0.0_f64], [1.0_f64], [2.0_f64]];
    let y = array![0.0_f64, 1.0, 2.0];

    // Default min_samples_split = 2: internal node {1,2} splits again -> depth 2
    let mut tree_default = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree_default.fit(&x, &y).unwrap();
    let depth_default = leaf_depth(tree_default.get_root().unwrap());
    assert_eq!(
        depth_default, 2,
        "default min_samples_split=2 must build a depth-2 tree on 3 fully-separable classes"
    );

    // min_samples_split = 3: the 2-sample internal node must stop -> depth 1
    let mut tree_restricted = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_min_samples_split(3)
        .unwrap();
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

/// predict_proba returns the exact empirical class distribution of an impure leaf
/// (with max_depth=1 the impure right leaf stores [2/3, 1/3])
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

    let mut tree = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_max_depth(1);
    tree.fit(&x, &y).unwrap();

    // x = 4.0 lands in the impure right leaf {2 of class 0, 1 of class 1}
    let proba = tree.predict_proba_one(&[4.0_f64]).unwrap();
    assert_eq!(proba.len(), 2);
    assert_abs_diff_eq!(proba[0], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(proba[1], 1.0 / 3.0, epsilon = 1e-12);

    // Same exact distribution via the batch API
    let probas = tree.predict_proba(&array![[4.0_f64]]).unwrap();
    assert_eq!(probas.shape(), &[1, 2]);
    assert_abs_diff_eq!(probas[[0, 0]], 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(probas[[0, 1]], 1.0 / 3.0, epsilon = 1e-12);
}

// max_depth = Some(0): the root is forced to be an immediate leaf

/// With max_depth = Some(0) no split is made and the root is a single leaf holding
/// the global majority class, so predict returns that class for every input
#[test]
fn test_max_depth_zero_root_is_leaf_predicts_global_majority() {
    let x = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 1.0],
    ];
    // 3 class-0 and 2 class-1 => global majority is class 0
    let y = array![0.0_f64, 0.0, 0.0, 1.0, 1.0];

    let mut tree = DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .with_max_depth(0);
    tree.fit(&x, &y).unwrap();

    // The root must be a single leaf (no split at all)
    assert!(
        matches!(
            tree.get_root().expect("tree must be fitted").node_type,
            NodeType::Leaf { .. }
        ),
        "max_depth=Some(0) must make the root an immediate leaf"
    );

    // Every prediction must be the global majority class (0.0), including the
    // class-1 region and points far outside the training range
    let probe = array![
        [0.0_f64, 0.0], // training class-0 point
        [1.0, 1.0],     // training class-1 point - still predicted 0
        [1.1, 1.0],     // training class-1 point - still predicted 0
        [99.0, 99.0],   // far outside the training range
    ];
    let preds = tree.predict(&probe).unwrap();
    for &p in preds.iter() {
        assert_abs_diff_eq!(p, 0.0, epsilon = 1e-12);
    }
}

// Categorical (multi-way) save/load round-trip - exercises the AHashMap `children` serde branch

/// A C4.5 tree with a populated categorical `children` map round-trips through
/// serde losslessly, reproducing predictions on training and unseen categories
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

    let mut original = DecisionTree::new(Algorithm::C45, true).unwrap();
    original.set_categorical_features(vec![0]);
    original.fit(&x_cat, &y_cat).unwrap();

    // Sanity: the fitted root must be a multi-way categorical node (populated
    // children map), otherwise the AHashMap serde branch is not exercised
    assert!(
        original
            .get_root()
            .and_then(|n| n.children.as_ref())
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "expected a categorical root with a populated children map"
    );

    // Predictions including an unseen category (99.0) that routes through the fallback branch
    let x_eval = array![[0.0_f64], [1.0_f64], [2.0_f64], [99.0_f64],];
    let preds_before = original.predict(&x_eval).unwrap();

    let path = format!("/tmp/rustyml_dt_cat_test_{}.json", std::process::id());
    original.save_to_path(&path).expect("save_to_path failed");

    let loaded = DecisionTree::load_from_path(&path).expect("load_from_path failed");
    let preds_after = loaded.predict(&x_eval).unwrap();

    // The deserialized children map must reproduce predictions bit-for-bit
    assert_allclose(&preds_before, &preds_after, 1e-12);
    // The categorical structure must survive the round-trip
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

// Global crate seed drives tie-breaking for random_state = None trees

/// RAII guard that clears the thread-local global seed on drop, so a panic in the
/// test body cannot leak a seeded global stream onto this thread
struct GlobalSeedGuard;
impl Drop for GlobalSeedGuard {
    fn drop(&mut self) {
        clear_global_seed();
    }
}

/// Fit a `random_state = None` CART tree on the tie data and return its structure
/// string; tie-breaking derives from the thread-local global seed when one is set
fn tie_break_structure_none() -> String {
    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    let (x, y) = tie_break_data();
    tree.fit(&x, &y).unwrap();
    tree.generate_tree_structure().unwrap()
}

/// Re-seeding the global stream to the same value before each fit makes two
/// `random_state = None` trees identical, proving the global seed drives tie-breaking
#[test]
fn global_seed_makes_none_tree_tie_breaking_reproducible() {
    let _guard = GlobalSeedGuard;

    // First fit under global seed 12345
    set_global_seed(12345);
    let structure_a = tie_break_structure_none();

    // Re-seed the global stream to the same value, restarting it, then fit again
    set_global_seed(12345);
    let structure_b = tie_break_structure_none();

    assert_eq!(
        structure_a, structure_b,
        "two random_state=None trees fit under the same global seed must be identical"
    );
    // _guard clears the global seed here (and on any panic above)
}
