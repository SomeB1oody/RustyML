//! Integration tests for `LogisticRegression` and `generate_polynomial_features`
//!
//! Label contract: `predict` returns `Array1<i32>` with values in {0, 1};
//! `predict_proba` returns `Array1<f64>` with values in (0.0, 1.0)

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::RegularizationType;
use rustyml::machine_learning::{LogisticRegression, generate_polynomial_features};

use crate::common::assert_allclose;

// Constructor validation

/// Zero learning rate is rejected with InvalidParameter
#[test]
fn new_zero_learning_rate_is_invalid() {
    let result = LogisticRegression::new(true, 0.0, 100, 1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Negative learning rate is rejected with InvalidParameter
#[test]
fn new_negative_learning_rate_is_invalid() {
    let result = LogisticRegression::new(true, -0.01, 100, 1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// NaN learning rate is rejected with InvalidParameter
#[test]
fn new_nan_learning_rate_is_invalid() {
    let result = LogisticRegression::new(true, f64::NAN, 100, 1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Infinite learning rate is rejected with InvalidParameter
#[test]
fn new_inf_learning_rate_is_invalid() {
    let result = LogisticRegression::new(true, f64::INFINITY, 100, 1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// max_iterations = 0 is rejected with InvalidParameter
#[test]
fn new_zero_max_iterations_is_invalid() {
    let result = LogisticRegression::new(true, 0.1, 0, 1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Tolerance = 0.0 is rejected (must be strictly positive)
#[test]
fn new_zero_tolerance_is_invalid() {
    let result = LogisticRegression::new(true, 0.1, 100, 0.0);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Negative tolerance is rejected with InvalidParameter
#[test]
fn new_negative_tolerance_is_invalid() {
    let result = LogisticRegression::new(true, 0.1, 100, -1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Negative L1 alpha is rejected with InvalidParameter
#[test]
fn new_negative_l1_alpha_is_invalid() {
    let result = LogisticRegression::new(true, 0.1, 100, 1e-4)
        .unwrap()
        .with_regularization(RegularizationType::L1(-0.5));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Negative L2 alpha is rejected with InvalidParameter
#[test]
fn new_negative_l2_alpha_is_invalid() {
    let result = LogisticRegression::new(true, 0.1, 100, 1e-4)
        .unwrap()
        .with_regularization(RegularizationType::L2(-1.0));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// NaN L2 alpha is rejected with InvalidParameter
#[test]
fn new_nan_l2_alpha_is_invalid() {
    let result = LogisticRegression::new(true, 0.1, 100, 1e-4)
        .unwrap()
        .with_regularization(RegularizationType::L2(f64::NAN));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Valid constructor parameters succeed and getters return stored values
#[test]
fn new_valid_params_succeeds_getters_correct() {
    let model = LogisticRegression::new(false, 0.05, 200, 1e-5)
        .expect("valid params should succeed")
        .with_regularization(RegularizationType::L2(0.1))
        .unwrap();

    assert!(!model.get_fit_intercept());
    assert_abs_diff_eq!(model.get_learning_rate(), 0.05, epsilon = 1e-12);
    assert_eq!(model.get_max_iterations(), 200);
    assert_abs_diff_eq!(model.get_tolerance(), 1e-5, epsilon = 1e-15);
    assert_eq!(
        model.get_regularization_type(),
        Some(RegularizationType::L2(0.1))
    );
    assert!(
        model.get_weights().is_none(),
        "weights should be None before fitting"
    );
    assert!(model.get_actual_iterations().is_none());
}

/// Default model has correct parameter values
#[test]
fn default_model_has_correct_params() {
    let model = LogisticRegression::default();
    assert!(model.get_fit_intercept());
    assert_abs_diff_eq!(model.get_learning_rate(), 0.01, epsilon = 1e-12);
    assert_eq!(model.get_max_iterations(), 100);
    assert_abs_diff_eq!(model.get_tolerance(), 1e-4, epsilon = 1e-10);
    assert!(model.get_regularization_type().is_none());
    assert!(model.get_weights().is_none());
}

// Fit validation error paths

/// Empty input data returns EmptyInput
#[test]
fn fit_empty_x_returns_empty_input() {
    let mut model = LogisticRegression::default();
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<f64> = Array1::zeros(0);
    assert!(
        matches!(model.fit(&x, &y), Err(Error::EmptyInput(_))),
        "expected EmptyInput"
    );
}

/// NaN in x returns NonFinite
#[test]
fn fit_nan_in_x_returns_non_finite() {
    let mut model = LogisticRegression::default();
    let x = array![[1.0, f64::NAN], [2.0, 3.0]];
    let y = array![0.0, 1.0];
    assert!(
        matches!(model.fit(&x, &y), Err(Error::NonFinite(_))),
        "expected NonFinite"
    );
}

/// Infinity in x returns NonFinite
#[test]
fn fit_inf_in_x_returns_non_finite() {
    let mut model = LogisticRegression::default();
    let x = array![[1.0, f64::INFINITY], [2.0, 3.0]];
    let y = array![0.0, 1.0];
    assert!(
        matches!(model.fit(&x, &y), Err(Error::NonFinite(_))),
        "expected NonFinite"
    );
}

/// Mismatched x rows (3) and y length (2) returns DimensionMismatch
#[test]
fn fit_xy_dimension_mismatch_returns_dimension_mismatch() {
    let mut model = LogisticRegression::default();
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let y = array![0.0, 1.0];
    assert!(
        matches!(model.fit(&x, &y), Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch"
    );
}

/// y containing 0.5 (non-binary) returns InvalidInput
#[test]
fn fit_non_binary_label_half_returns_invalid_input() {
    let mut model = LogisticRegression::default();
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![0.0, 0.5, 1.0];
    assert!(
        matches!(model.fit(&x, &y), Err(Error::InvalidInput(_))),
        "expected InvalidInput for y containing 0.5"
    );
}

/// y containing 2.0 (non-binary) returns InvalidInput
#[test]
fn fit_non_binary_label_two_returns_invalid_input() {
    let mut model = LogisticRegression::default();
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![2.0, 0.0, 1.0];
    assert!(
        matches!(model.fit(&x, &y), Err(Error::InvalidInput(_))),
        "expected InvalidInput for y containing 2.0"
    );
}

/// y containing -1.0 returns InvalidInput: label domain is strictly {0, 1}, not {-1, +1}
#[test]
fn fit_non_binary_label_minus_one_returns_invalid_input() {
    let mut model = LogisticRegression::default();
    let x = array![[1.0], [2.0]];
    let y = array![-1.0, 1.0];
    assert!(
        matches!(model.fit(&x, &y), Err(Error::InvalidInput(_))),
        "expected InvalidInput for y containing -1.0"
    );
}

// Predict / predict_proba error paths

/// predict on an unfitted model returns NotFitted
#[test]
fn predict_before_fit_returns_not_fitted() {
    let model = LogisticRegression::default();
    let x = array![[1.0, 2.0]];
    assert!(
        matches!(model.predict(&x), Err(Error::NotFitted(_))),
        "expected NotFitted"
    );
}

/// predict_proba on an unfitted model returns NotFitted
#[test]
fn predict_proba_before_fit_returns_not_fitted() {
    let model = LogisticRegression::default();
    let x = array![[1.0, 2.0]];
    assert!(
        matches!(model.predict_proba(&x), Err(Error::NotFitted(_))),
        "expected NotFitted"
    );
}

/// predict with wrong feature count (trained on 2, called with 3) returns DimensionMismatch
#[test]
fn predict_wrong_feature_count_returns_dimension_mismatch() {
    let mut model = LogisticRegression::new(true, 0.1, 500, 1e-6).expect("valid params");
    let x_train = array![[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0],];
    let y_train = array![0.0, 0.0, 1.0, 1.0];
    model.fit(&x_train, &y_train).expect("fit should succeed");

    let x_wrong = array![[1.0, 2.0, 3.0]];
    assert!(
        matches!(
            model.predict(&x_wrong),
            Err(Error::DimensionMismatch { .. })
        ),
        "expected DimensionMismatch"
    );
}

/// predict with NaN input returns NonFinite
#[test]
fn predict_nan_input_returns_non_finite() {
    let mut model = LogisticRegression::new(true, 0.1, 500, 1e-6).expect("valid params");
    let x_train = array![[0.0], [10.0]];
    let y_train = array![0.0, 1.0];
    model.fit(&x_train, &y_train).expect("fit should succeed");

    let x_nan = array![[f64::NAN]];
    assert!(
        matches!(model.predict(&x_nan), Err(Error::NonFinite(_))),
        "expected NonFinite"
    );
}

// Correctness: linearly separable data

/// On perfectly separable data (class 0 at x1 << 0, class 1 at x1 >> 0) all training points are classified correctly
#[test]
fn predict_linearly_separable_classifies_correctly() {
    let mut model = LogisticRegression::new(true, 0.1, 2000, 1e-7).expect("valid params");

    let x_train = array![
        [-10.0, 0.0],
        [-10.0, 5.0],
        [-8.0, -3.0],
        [10.0, 0.0],
        [10.0, -5.0],
        [8.0, 3.0],
    ];
    let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    model.fit(&x_train, &y_train).expect("fit should succeed");

    let preds = model.predict(&x_train).expect("predict should succeed");

    // Predictions must be exactly in {0, 1} (label domain contract)
    for &p in preds.iter() {
        assert!(p == 0 || p == 1, "label {p} outside {{0,1}}");
    }

    // All training points must be classified correctly
    assert_eq!(preds[0], 0, "(-10,0) should be class 0");
    assert_eq!(preds[1], 0, "(-10,5) should be class 0");
    assert_eq!(preds[2], 0, "(-8,-3) should be class 0");
    assert_eq!(preds[3], 1, "(10,0) should be class 1");
    assert_eq!(preds[4], 1, "(10,-5) should be class 1");
    assert_eq!(preds[5], 1, "(8,3) should be class 1");
}

/// predict_proba returns values strictly in (0, 1), with class-0 probabilities < 0.5 and class-1 > 0.5, consistent with predict
#[test]
fn predict_proba_range_and_consistency_with_predict() {
    let mut model = LogisticRegression::new(true, 0.1, 2000, 1e-7).expect("valid params");

    let x_train = array![[-10.0, 0.0], [-10.0, 5.0], [10.0, 0.0], [10.0, -5.0],];
    let y_train = array![0.0, 0.0, 1.0, 1.0];
    model.fit(&x_train, &y_train).expect("fit should succeed");

    let probs = model
        .predict_proba(&x_train)
        .expect("predict_proba should succeed");

    // All probabilities must be strictly within (0, 1)
    for &p in probs.iter() {
        assert!(p > 0.0 && p < 1.0, "probability {p} not in (0,1)");
    }

    // Class-0 samples must have probability < 0.5
    assert!(
        probs[0] < 0.5,
        "class-0 sample prob {} should be < 0.5",
        probs[0]
    );
    assert!(
        probs[1] < 0.5,
        "class-0 sample prob {} should be < 0.5",
        probs[1]
    );

    // Class-1 samples must have probability > 0.5
    assert!(
        probs[2] > 0.5,
        "class-1 sample prob {} should be > 0.5",
        probs[2]
    );
    assert!(
        probs[3] > 0.5,
        "class-1 sample prob {} should be > 0.5",
        probs[3]
    );

    // predict and predict_proba must agree: predict thresholds at 0.5
    let preds = model.predict(&x_train).expect("predict should succeed");
    for (i, (&prob, &pred)) in probs.iter().zip(preds.iter()).enumerate() {
        let expected_label = if prob >= 0.5 { 1 } else { 0 };
        assert_eq!(
            pred, expected_label,
            "predict/predict_proba disagree at sample {i}"
        );
    }
}

/// fit_predict returns the same labels as a separate fit + predict on training data
#[test]
fn fit_predict_agrees_with_fit_then_predict() {
    let x = array![[-5.0, 0.0], [-5.0, 1.0], [5.0, 0.0], [5.0, 1.0],];
    let y = array![0.0, 0.0, 1.0, 1.0];

    let mut model_a = LogisticRegression::new(true, 0.1, 1000, 1e-7).expect("valid params");
    let labels_fit_predict = model_a
        .fit_predict(&x, &y)
        .expect("fit_predict should succeed");

    let mut model_b = LogisticRegression::new(true, 0.1, 1000, 1e-7).expect("valid params");
    model_b.fit(&x, &y).expect("fit should succeed");
    let labels_separate = model_b.predict(&x).expect("predict should succeed");

    assert_eq!(labels_fit_predict, labels_separate);
}

// fit_intercept=false

/// Without an intercept, the model has exactly n_features weights (2 features -> 2 weights, no bias column)
#[test]
fn fit_no_intercept_weight_count_equals_features() {
    let mut model = LogisticRegression::new(false, 0.1, 1000, 1e-7).expect("valid params");

    let x = array![[-5.0, 1.0], [-4.0, 0.0], [4.0, 0.0], [5.0, 1.0],];
    let y = array![0.0, 0.0, 1.0, 1.0];
    model.fit(&x, &y).expect("fit should succeed");

    let w = model
        .get_weights()
        .expect("weights should be Some after fit");
    assert_eq!(w.len(), 2, "without intercept, weight length == n_features");
}

/// With an intercept, the model has n_features + 1 weights (bias is first element)
#[test]
fn fit_with_intercept_weight_count_equals_features_plus_one() {
    let mut model = LogisticRegression::new(true, 0.1, 1000, 1e-7).expect("valid params");

    let x = array![[-5.0, 1.0], [-4.0, 0.0], [4.0, 0.0], [5.0, 1.0],];
    let y = array![0.0, 0.0, 1.0, 1.0];
    model.fit(&x, &y).expect("fit should succeed");

    let w = model
        .get_weights()
        .expect("weights should be Some after fit");
    assert_eq!(
        w.len(),
        3,
        "with intercept, weight length == n_features + 1"
    );
}

// n_iter tracking

/// After fitting, n_iter is Some and at least 1
#[test]
fn fit_sets_n_iter() {
    let mut model = LogisticRegression::new(true, 0.1, 500, 1e-7).expect("valid params");
    let x = array![[-5.0], [5.0]];
    let y = array![0.0, 1.0];
    model.fit(&x, &y).expect("fit should succeed");

    let n = model
        .get_actual_iterations()
        .expect("n_iter should be Some after fit");
    assert!(n >= 1, "n_iter should be at least 1");
    assert!(n <= 500, "n_iter should not exceed max_iter");
}

// Regularization reduces weight magnitude

/// L2 regularization produces a smaller feature-weight norm than no regularization
#[test]
fn l2_regularization_shrinks_weight_norm() {
    // Dataset where gradients are well-defined
    let x = array![
        [-4.0, -3.0],
        [-3.0, -4.0],
        [-2.0, -1.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut model_unreg = LogisticRegression::new(true, 0.1, 2000, 1e-8).expect("valid params");
    model_unreg.fit(&x, &y).expect("fit should succeed");

    let mut model_l2 = LogisticRegression::new(true, 0.1, 2000, 1e-8)
        .expect("valid params")
        .with_regularization(RegularizationType::L2(5.0))
        .unwrap();
    model_l2.fit(&x, &y).expect("fit should succeed");

    let w_unreg = model_unreg.get_weights().expect("weights present");
    let w_l2 = model_l2.get_weights().expect("weights present");

    // Squared norm of the feature weights only (skip index 0 = bias)
    let norm_sq_unreg: f64 = w_unreg.iter().skip(1).map(|&w| w * w).sum();
    let norm_sq_l2: f64 = w_l2.iter().skip(1).map(|&w| w * w).sum();

    assert!(
        norm_sq_l2 < norm_sq_unreg,
        "L2 regularized weight norm ({norm_sq_l2}) should be smaller than unregularized ({norm_sq_unreg})"
    );
}

/// L1 regularization produces a smaller feature-weight norm than no regularization
#[test]
fn l1_regularization_shrinks_weight_norm() {
    let x = array![
        [-4.0, -3.0],
        [-3.0, -4.0],
        [-2.0, -1.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut model_unreg = LogisticRegression::new(true, 0.1, 2000, 1e-8).expect("valid params");
    model_unreg.fit(&x, &y).expect("fit should succeed");

    let mut model_l1 = LogisticRegression::new(true, 0.1, 2000, 1e-8)
        .expect("valid params")
        .with_regularization(RegularizationType::L1(5.0))
        .unwrap();
    model_l1.fit(&x, &y).expect("fit should succeed");

    let w_unreg = model_unreg.get_weights().expect("weights present");
    let w_l1 = model_l1.get_weights().expect("weights present");

    let norm_sq_unreg: f64 = w_unreg.iter().skip(1).map(|&w| w * w).sum();
    let norm_sq_l1: f64 = w_l1.iter().skip(1).map(|&w| w * w).sum();

    assert!(
        norm_sq_l1 < norm_sq_unreg,
        "L1 regularized weight norm ({norm_sq_l1}) should be smaller than unregularized ({norm_sq_unreg})"
    );
}

// Save / load round-trip

/// Save + load produces a model whose predictions are bit-identical to the original
#[test]
fn save_load_round_trip_identical_predictions() {
    let x_train = array![[-5.0, 1.0], [-4.0, -1.0], [4.0, 1.0], [5.0, -1.0],];
    let y_train = array![0.0, 0.0, 1.0, 1.0];

    let mut model = LogisticRegression::new(true, 0.05, 1000, 1e-7).expect("valid params");
    model.fit(&x_train, &y_train).expect("fit should succeed");

    let path = "/tmp/rustyml_test_logistic_regression.json";
    model.save_to_path(path).expect("save should succeed");

    let loaded = LogisticRegression::load_from_path(path).expect("load should succeed");

    let preds_original = model.predict(&x_train).expect("predict should succeed");
    let preds_loaded = loaded
        .predict(&x_train)
        .expect("predict on loaded should succeed");

    assert_eq!(
        preds_original, preds_loaded,
        "predictions from original and loaded model must be identical"
    );

    let probs_original = model
        .predict_proba(&x_train)
        .expect("predict_proba should succeed");
    let probs_loaded = loaded
        .predict_proba(&x_train)
        .expect("predict_proba on loaded should succeed");

    assert_allclose(&probs_original, &probs_loaded, 1e-15);
}

/// After loading, the hyperparameters match the original model
#[test]
fn save_load_preserves_hyperparameters() {
    let x = array![[-1.0], [1.0]];
    let y = array![0.0, 1.0];

    let mut model = LogisticRegression::new(false, 0.05, 300, 1e-5)
        .expect("valid params")
        .with_regularization(RegularizationType::L1(0.2))
        .unwrap();
    model.fit(&x, &y).expect("fit should succeed");

    let path = "/tmp/rustyml_test_logistic_regression_hparams.json";
    model.save_to_path(path).expect("save should succeed");
    let loaded = LogisticRegression::load_from_path(path).expect("load should succeed");

    assert!(!loaded.get_fit_intercept());
    assert_abs_diff_eq!(loaded.get_learning_rate(), 0.05, epsilon = 1e-12);
    assert_eq!(loaded.get_max_iterations(), 300);
    assert_abs_diff_eq!(loaded.get_tolerance(), 1e-5, epsilon = 1e-12);
    assert_eq!(
        loaded.get_regularization_type(),
        Some(RegularizationType::L1(0.2))
    );
}

// generate_polynomial_features

/// degree=1 on n features returns x unchanged in shape and values
#[test]
fn poly_features_degree1_returns_x_unchanged() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = generate_polynomial_features(&x, 1);

    assert_eq!(
        result.shape(),
        &[3, 2],
        "shape must match input for degree=1"
    );
    assert_allclose(&result, &x, 1e-14);
}

/// 1 feature, degree=2: output is [x, x^2] - 2 columns
#[test]
fn poly_features_one_feature_degree2_gives_x_and_x_squared() {
    let x = array![[2.0]];
    let result = generate_polynomial_features(&x, 2);

    assert_eq!(
        result.shape(),
        &[1, 2],
        "1 feature degree 2 → 2 output cols"
    );
    assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-14); // x
    assert_abs_diff_eq!(result[[0, 1]], 4.0, epsilon = 1e-14); // x^2
}

/// 2 features, degree=2: output has 5 columns [x1, x2, x1^2, x1*x2, x2^2]
#[test]
fn poly_features_two_features_degree2_gives_five_columns() {
    let x = array![[3.0, 2.0]];
    let result = generate_polynomial_features(&x, 2);

    assert_eq!(
        result.shape(),
        &[1, 5],
        "2 features degree 2 → 5 output cols"
    );
    assert_abs_diff_eq!(result[[0, 0]], 3.0, epsilon = 1e-14); // x1
    assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-14); // x2
    assert_abs_diff_eq!(result[[0, 2]], 9.0, epsilon = 1e-14); // x1^2
    assert_abs_diff_eq!(result[[0, 3]], 6.0, epsilon = 1e-14); // x1*x2
    assert_abs_diff_eq!(result[[0, 4]], 4.0, epsilon = 1e-14); // x2^2
}

/// 3 features, degree=3: output has 19 columns
#[test]
fn poly_features_three_features_degree3_gives_nineteen_columns() {
    let x = array![[1.0, 2.0, 3.0]];
    let result = generate_polynomial_features(&x, 3);

    assert_eq!(
        result.shape(),
        &[1, 19],
        "3 features degree 3 → 19 output cols"
    );
}

/// Multiple samples are transformed independently: each row is expanded correctly
#[test]
fn poly_features_multiple_samples_each_row_correct() {
    let x = array![[3.0], [5.0]];
    let result = generate_polynomial_features(&x, 2);

    assert_eq!(result.shape(), &[2, 2]);
    assert_abs_diff_eq!(result[[0, 0]], 3.0, epsilon = 1e-14);
    assert_abs_diff_eq!(result[[0, 1]], 9.0, epsilon = 1e-14);
    assert_abs_diff_eq!(result[[1, 0]], 5.0, epsilon = 1e-14);
    assert_abs_diff_eq!(result[[1, 1]], 25.0, epsilon = 1e-14);
}

/// Degree-2 expansion makes circular data (classes split by x1^2 + x2^2) linearly separable, classifying all training points correctly
#[test]
fn poly_features_pipeline_makes_circular_data_separable() {
    // Class 0: on circle of radius ~1; class 1: on circle of radius ~5
    let x = array![
        [1.0, 0.0],  // radius 1, class 0
        [0.0, 1.0],  // radius 1, class 0
        [-1.0, 0.0], // radius 1, class 0
        [5.0, 0.0],  // radius 5, class 1
        [0.0, 5.0],  // radius 5, class 1
        [-5.0, 0.0], // radius 5, class 1
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let x_poly = generate_polynomial_features(&x, 2);
    // 2 features, degree 2 -> 5 columns
    assert_eq!(x_poly.ncols(), 5);

    let mut model = LogisticRegression::new(true, 0.01, 3000, 1e-7).expect("valid params");
    model
        .fit(&x_poly, &y)
        .expect("fit on polynomial features should succeed");

    let preds = model.predict(&x_poly).expect("predict should succeed");

    // All training points must be classified correctly
    for (i, (&p, &truth)) in preds.iter().zip(y.iter()).enumerate() {
        let expected = truth as i32;
        assert_eq!(p, expected, "sample {i}: expected {expected}, got {p}");
    }
}

// Determinism (same seed = identical weights)

/// Two fits with the same hyperparameters on the same data produce bit-identical weights (gradient descent has no randomness)
#[test]
fn deterministic_fit_produces_identical_weights() {
    let x = array![[-3.0, 1.0], [-2.0, -1.0], [2.0, 1.0], [3.0, -1.0],];
    let y = array![0.0, 0.0, 1.0, 1.0];

    let mut model_a = LogisticRegression::new(true, 0.1, 500, 1e-8).expect("valid params");
    model_a.fit(&x, &y).expect("fit should succeed");

    let mut model_b = LogisticRegression::new(true, 0.1, 500, 1e-8).expect("valid params");
    model_b.fit(&x, &y).expect("fit should succeed");

    let w_a = model_a.get_weights().expect("weights present");
    let w_b = model_b.get_weights().expect("weights present");

    assert_allclose(w_a, w_b, 0.0); // bit-identical
}

/// Two fits produce the same predictions on unseen data
#[test]
fn deterministic_fit_produces_identical_predictions_on_unseen_data() {
    let x_train = array![[-3.0, 1.0], [-2.0, -1.0], [2.0, 1.0], [3.0, -1.0],];
    let y_train = array![0.0, 0.0, 1.0, 1.0];

    let x_test = array![[-1.0, 0.0], [1.0, 0.0]];

    let mut model_a = LogisticRegression::new(true, 0.1, 500, 1e-8).expect("valid params");
    model_a.fit(&x_train, &y_train).expect("fit should succeed");

    let mut model_b = LogisticRegression::new(true, 0.1, 500, 1e-8).expect("valid params");
    model_b.fit(&x_train, &y_train).expect("fit should succeed");

    let preds_a = model_a.predict(&x_test).expect("predict should succeed");
    let preds_b = model_b.predict(&x_test).expect("predict should succeed");

    assert_eq!(
        preds_a, preds_b,
        "both runs must produce identical predictions"
    );
}
// In-loop NonFinite guards (huge learning rate on finite data)

/// A huge but finite learning rate on finite, large-magnitude data trips an in-loop numerical guard in `fit` and returns `Error::NonFinite`
#[test]
fn fit_huge_learning_rate_on_finite_data_returns_non_finite() {
    // Large-magnitude finite features so an overflowing update is guaranteed
    let x = array![[-1.0e9, 0.0], [-1.0e9, 1.0], [1.0e9, 0.0], [1.0e9, -1.0],];
    let y = array![0.0, 0.0, 1.0, 1.0];

    let mut model = LogisticRegression::new(true, f64::MAX, 100, 1e-7).expect("valid params");

    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "huge learning_rate must trip an in-loop NonFinite guard, got {:?}",
        result
    );
}

// regularization scaling convention (penalty NOT divided by n_samples)

/// The L2-regularized objective (1/n)Σlogloss + (α/2)||w||² is invariant to replicating
/// every sample (mean log-loss and penalty are both unchanged), so the regularized optimum
/// must be too. With a penalty erroneously divided by n_samples, replicating the data weakens
/// the effective penalty and inflates the weights. Full-batch GD makes the replicated run an
/// exact copy of the original once the penalty no longer depends on n.
#[test]
fn l2_regularization_strength_is_sample_count_invariant() {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.5],
        [0.5, 1.0],
        [2.0, 1.5],
        [-1.0, -0.5],
        [-0.5, -1.0],
        [-2.0, -1.5],
        [1.5, 2.0]
    ];
    let y = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // 3x row-replicated copy
    let mut x3 = Array2::<f64>::zeros((x.nrows() * 3, x.ncols()));
    let mut y3 = Array1::<f64>::zeros(y.len() * 3);
    for r in 0..3 {
        for i in 0..x.nrows() {
            x3.row_mut(r * x.nrows() + i).assign(&x.row(i));
            y3[r * y.len() + i] = y[i];
        }
    }

    let train = |xx: &Array2<f64>, yy: &Array1<f64>| {
        let mut m = LogisticRegression::new(true, 0.1, 20_000, 1e-12)
            .unwrap()
            .with_regularization(RegularizationType::L2(2.0))
            .unwrap();
        m.fit(xx, yy).unwrap();
        m.get_weights().unwrap().clone()
    };

    let w1 = train(&x, &y);
    let w3 = train(&x3, &y3);
    for i in 0..w1.len() {
        assert_abs_diff_eq!(w1[i], w3[i], epsilon = 1e-6);
    }
}
