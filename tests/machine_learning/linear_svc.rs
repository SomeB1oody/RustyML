//! Integration tests for LinearSVC: constructor validation, fit/predict error
//! paths, label domain, sign consistency, separable-data correctness, getters,
//! penalties, save/load round-trip, and random_state reproducibility

use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::{LinearSVC, RegularizationType};

// helper

/// Return (x, y) for a perfectly linearly-separable 2-feature problem split by
/// the boundary x1=0 (class 0 at negative x1, class 1 at positive x1)
fn make_separable() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            -5.0, 0.0, -6.0, 0.0, -7.0, 0.0, -4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 4.0, 0.0,
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    (x, y)
}

/// Fit a default (L2) LinearSVC on the separable dataset with enough iterations
/// that convergence is expected, returning the trained model
fn fit_separable_model() -> LinearSVC {
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), true, 1e-5).unwrap();
    model.fit(&x, &y).unwrap();
    model
}

// constructor validation

#[test]
fn new_rejects_max_iter_zero() {
    let result = LinearSVC::new(0, 0.01, RegularizationType::L2(1.0), true, 1e-4);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "max_iter=0 must return InvalidParameter"
    );
}

#[test]
fn new_rejects_invalid_learning_rate() {
    // Each invalid learning_rate (zero / negative / NaN / +Inf) hits the same
    // guard and must return InvalidParameter.
    for lr in [0.0, -0.001, f64::NAN, f64::INFINITY] {
        let result = LinearSVC::new(100, lr, RegularizationType::L2(1.0), true, 1e-4);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for learning_rate={lr:?}"
        );
    }
}

#[test]
fn new_rejects_invalid_tol() {
    // Each invalid tol (zero / negative / NaN) hits the same guard and must
    // return InvalidParameter.
    for tol in [0.0, -1e-4, f64::NAN] {
        let result = LinearSVC::new(100, 0.01, RegularizationType::L2(1.0), true, tol);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for tol={tol:?}"
        );
    }
}

#[test]
fn new_rejects_invalid_penalty_lambda() {
    // Each invalid penalty lambda (negative L2, negative L1, L2(NaN), L1(+Inf))
    // hits the same guard and must return InvalidParameter.
    for penalty in [
        RegularizationType::L2(-0.5),
        RegularizationType::L1(-1.0),
        RegularizationType::L2(f64::NAN),
        RegularizationType::L1(f64::INFINITY),
    ] {
        let result = LinearSVC::new(100, 0.01, penalty, true, 1e-4);
        assert!(
            matches!(result, Err(Error::InvalidParameter { .. })),
            "expected InvalidParameter for penalty={penalty:?}"
        );
    }
}

/// Lambda = 0.0 is explicitly allowed (no regularization)
#[test]
fn new_accepts_zero_penalty_lambda() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L2(0.0), true, 1e-4);
    assert!(result.is_ok(), "L2(0.0) must be accepted");
}

/// All defaults match documented values
#[test]
fn default_constructor_has_documented_defaults() {
    let model = LinearSVC::default();
    assert_eq!(model.get_max_iterations(), 1000);
    assert_eq!(model.get_learning_rate(), 0.001);
    assert_eq!(model.get_tolerance(), 1e-4);
    assert!(model.get_fit_intercept());
    assert_eq!(model.get_penalty(), RegularizationType::L2(1.0));
    assert!(
        model.get_weights().is_none(),
        "weights should be None before fit"
    );
    assert!(model.get_bias().is_none(), "bias should be None before fit");
    assert!(
        model.get_actual_iterations().is_none(),
        "n_iter should be None before fit"
    );
}

/// Constructor stores the passed parameters exactly
#[test]
fn new_stores_parameters() {
    let model = LinearSVC::new(500, 0.005, RegularizationType::L1(0.1), false, 1e-6).unwrap();
    assert_eq!(model.get_max_iterations(), 500);
    assert_eq!(model.get_learning_rate(), 0.005);
    assert_eq!(model.get_penalty(), RegularizationType::L1(0.1));
    assert!(!model.get_fit_intercept());
    assert_eq!(model.get_tolerance(), 1e-6);
}

// fit error paths

#[test]
fn fit_rejects_empty_x() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<f64> = Array1::zeros(0);
    let mut model = LinearSVC::default();
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "0-row input must return EmptyInput"
    );
}

#[test]
fn fit_rejects_empty_y() {
    let x: Array2<f64> = Array2::zeros((3, 2));
    let y: Array1<f64> = Array1::zeros(0);
    let mut model = LinearSVC::default();
    let result = model.fit(&x, &y);
    // Either DimensionMismatch (y.len()=0 != x.nrows()=3) or EmptyInput is valid,
    // depending on which guard fires first
    assert!(result.is_err(), "mismatched empty y must return an error");
}

#[test]
fn fit_rejects_dimension_mismatch_xy() {
    // x has 4 rows but y has 3 elements
    let x = Array2::zeros((4, 2));
    let y = Array1::zeros(3);
    let mut model = LinearSVC::default();
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "x/y row count mismatch must return DimensionMismatch"
    );
}

#[test]
fn fit_rejects_non_finite_in_x() {
    // Each non-finite sentinel (NaN / +Inf) in x hits the same finiteness guard
    // and must return NonFinite.
    for sentinel in [f64::NAN, f64::INFINITY] {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, sentinel, 4.0]).unwrap();
        let y = array![0.0, 1.0];
        let mut model = LinearSVC::default();
        let result = model.fit(&x, &y);
        assert!(
            matches!(result, Err(Error::NonFinite(_))),
            "expected NonFinite for sentinel={sentinel:?} in x"
        );
    }
}

// predict / decision_function error paths

#[test]
fn predict_not_fitted_returns_not_fitted_error() {
    let model = LinearSVC::default();
    let x = Array2::zeros((2, 2));
    let result = model.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted("LinearSVC"))),
        "predict before fit must return NotFitted(\"LinearSVC\")"
    );
}

#[test]
fn decision_function_not_fitted_returns_not_fitted_error() {
    let model = LinearSVC::default();
    let x = Array2::zeros((2, 2));
    let result = model.decision_function(&x);
    assert!(
        matches!(result, Err(Error::NotFitted("LinearSVC"))),
        "decision_function before fit must return NotFitted(\"LinearSVC\")"
    );
}

#[test]
fn predict_wrong_feature_count_returns_dimension_mismatch() {
    // Fit on 2-feature data, then predict with 3 features
    let model = fit_separable_model();
    let x_wrong = Array2::zeros((2, 3));
    let result = model.predict(&x_wrong);
    assert!(
        matches!(
            result,
            Err(Error::DimensionMismatch {
                expected: 2,
                found: 3
            })
        ),
        "wrong feature count must return DimensionMismatch(expected=2, found=3)"
    );
}

#[test]
fn decision_function_wrong_feature_count_returns_dimension_mismatch() {
    let model = fit_separable_model();
    let x_wrong = Array2::zeros((2, 1));
    let result = model.decision_function(&x_wrong);
    assert!(
        matches!(
            result,
            Err(Error::DimensionMismatch {
                expected: 2,
                found: 1
            })
        ),
        "wrong feature count must return DimensionMismatch(expected=2, found=1)"
    );
}

#[test]
fn predict_empty_input_after_fit_returns_error() {
    let model = fit_separable_model();
    // 0-row input with correct ncols
    let x_empty: Array2<f64> = Array2::zeros((0, 2));
    let result = model.predict(&x_empty);
    assert!(
        result.is_err(),
        "empty input to predict after fit must return an error"
    );
}

// output label domain: must be exactly 0.0 or 1.0

#[test]
fn predict_labels_are_in_zero_one_domain() {
    let model = fit_separable_model();
    let (x, _y) = make_separable();
    let predictions = model.predict(&x).unwrap();
    for (i, &pred) in predictions.iter().enumerate() {
        assert!(
            pred == 0.0 || pred == 1.0,
            "prediction[{i}] = {pred} is not in {{0.0, 1.0}}"
        );
    }
}

// sign-consistency invariant
// Contract: predict(x)==1.0 iff decision_function(x)>0.0, else predict(x)==0.0

#[test]
fn sign_consistency_on_training_data() {
    let model = fit_separable_model();
    let (x, _y) = make_separable();
    let preds = model.predict(&x).unwrap();
    let scores = model.decision_function(&x).unwrap();
    for i in 0..preds.len() {
        let expected_pred = if scores[i] > 0.0 { 1.0 } else { 0.0 };
        assert_eq!(
            preds[i], expected_pred,
            "sign inconsistency at sample {i}: score={:.6} but predict={}",
            scores[i], preds[i]
        );
    }
}

#[test]
fn sign_consistency_on_new_points() {
    // [+10, 0] should score > 0 (class 1), [-10, 0] should score < 0 (class 0);
    // assert the invariant rather than a hardcoded class, since the sign axis may flip
    let model = fit_separable_model();
    let x_new = Array2::from_shape_vec((2, 2), vec![10.0, 0.0, -10.0, 0.0]).unwrap();
    let preds = model.predict(&x_new).unwrap();
    let scores = model.decision_function(&x_new).unwrap();
    for i in 0..2 {
        let expected_pred = if scores[i] > 0.0 { 1.0 } else { 0.0 };
        assert_eq!(
            preds[i], expected_pred,
            "sign consistency violated at sample {i}: score={:.6}, predict={}",
            scores[i], preds[i]
        );
    }
}

// correctness on separable data

/// On perfectly linearly-separable data the classifier predicts the true class
/// for every training sample
#[test]
fn predicts_all_training_samples_correctly_on_separable_data() {
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6).unwrap();
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&x).unwrap();
    // Ground truth: samples 0-3 are class 0, samples 4-7 are class 1
    let ground_truth = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (i, (&pred, &truth)) in preds.iter().zip(ground_truth.iter()).enumerate() {
        assert_eq!(
            pred, truth,
            "sample {i}: expected class {truth} but got {pred}"
        );
    }
}

/// fit_predict agrees with fit then predict on the same data
#[test]
fn fit_predict_agrees_with_fit_then_predict() {
    let (x, y) = make_separable();
    let mut model_a = LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), true, 1e-5).unwrap();
    let preds_a = model_a.fit_predict(&x, &y).unwrap();
    let preds_b = model_a.predict(&x).unwrap();
    assert_eq!(preds_a, preds_b, "fit_predict and fit→predict must agree");
}

// getters after fit

#[test]
fn getters_are_some_after_fit() {
    let model = fit_separable_model();
    assert!(
        model.get_weights().is_some(),
        "weights must be Some after fit"
    );
    assert!(model.get_bias().is_some(), "bias must be Some after fit");
    assert!(
        model.get_actual_iterations().is_some(),
        "n_iter must be Some after fit"
    );
}

#[test]
fn n_iter_is_in_valid_range() {
    let max_iter = 5000usize;
    let (x, y) = make_separable();
    let mut model =
        LinearSVC::new(max_iter, 0.01, RegularizationType::L2(0.1), true, 1e-5).unwrap();
    model.fit(&x, &y).unwrap();
    let n = model.get_actual_iterations().unwrap();
    assert!(n >= 1, "n_iter must be at least 1");
    assert!(
        n <= max_iter,
        "n_iter {n} must not exceed max_iter {max_iter}"
    );
}

/// Early stopping fires before max_iter on well-separated data with tight tolerance
#[test]
fn convergence_stops_before_max_iter_on_separable_data() {
    let max_iter = 50_000usize;
    let (x, y) = make_separable();
    let mut model =
        LinearSVC::new(max_iter, 0.01, RegularizationType::L2(0.01), true, 1e-3).unwrap();
    model.fit(&x, &y).unwrap();
    let n = model.get_actual_iterations().unwrap();
    assert!(
        n < max_iter,
        "expected early stopping (n_iter={n} < max_iter={max_iter})"
    );
}

#[test]
fn weights_have_correct_dimensionality() {
    let model = fit_separable_model();
    let weights = model.get_weights().unwrap();
    assert_eq!(
        weights.len(),
        2,
        "weight vector must match number of training features"
    );
}

// fit_intercept = false

/// When fit_intercept=false the bias is never updated and stays 0.0
#[test]
fn fit_intercept_false_bias_stays_zero() {
    // Data centred at origin lets the zero-bias model solve the problem
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), false, 1e-5).unwrap();
    model.fit(&x, &y).unwrap();
    let bias = model.get_bias().unwrap();
    assert_eq!(
        bias, 0.0,
        "bias must stay exactly 0.0 when fit_intercept=false"
    );
}

/// When fit_intercept=false, decision_function(x) = x . weights exactly (bias is 0.0)
#[test]
fn fit_intercept_false_decision_function_equals_dot_product() {
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), false, 1e-5).unwrap();
    model.fit(&x, &y).unwrap();

    let weights = model.get_weights().unwrap().clone();
    let bias = model.get_bias().unwrap();
    assert_eq!(bias, 0.0, "bias must be 0 for this test to be meaningful");

    let scores = model.decision_function(&x).unwrap();
    for i in 0..x.nrows() {
        let manual = x.row(i).dot(&weights) + bias;
        assert!(
            (scores[i] - manual).abs() < 1e-12,
            "decision_function[{i}]={:.10} != dot-product={:.10}",
            scores[i],
            manual
        );
    }
}

// L1 vs L2 penalty produce different weight vectors

/// L1 and L2 penalties with the same lambda on the same data yield different
/// weight vectors
#[test]
fn l1_and_l2_penalties_produce_different_weights() {
    let (x, y) = make_separable();

    let mut model_l2 = LinearSVC::new(3000, 0.01, RegularizationType::L2(1.0), true, 1e-6).unwrap();
    model_l2.fit(&x, &y).unwrap();
    let w_l2 = model_l2.get_weights().unwrap().clone();

    let mut model_l1 = LinearSVC::new(3000, 0.01, RegularizationType::L1(1.0), true, 1e-6).unwrap();
    model_l1.fit(&x, &y).unwrap();
    let w_l1 = model_l1.get_weights().unwrap().clone();

    assert_ne!(
        w_l1, w_l2,
        "L1 and L2 regularization must yield different weight vectors"
    );
}

/// Strong L1 keeps the irrelevant feature (column 1, all zeros) smaller in
/// magnitude than the relevant feature: |w[1]| < |w[0]|
#[test]
fn l1_sparsity_irrelevant_feature_closer_to_zero() {
    let (x, y) = make_separable();
    // x column 1 is all zeros, so a strong L1 penalty shrinks w[1] toward zero
    let mut model =
        LinearSVC::new(10_000, 0.001, RegularizationType::L1(10.0), true, 1e-7).unwrap();
    model.fit(&x, &y).unwrap();
    let w = model.get_weights().unwrap();
    assert!(
        w[1].abs() < w[0].abs(),
        "under strong L1 the irrelevant feature (w[1]={:.6}) should have smaller \
         magnitude than the relevant feature (w[0]={:.6})",
        w[1],
        w[0]
    );
}

// save / load round-trip

#[test]
fn save_load_round_trip_yields_identical_predictions() {
    let path = "/tmp/rustyml_linear_svc_roundtrip_test.json";

    let model = fit_separable_model();
    model.save_to_path(path).unwrap();

    let loaded = LinearSVC::load_from_path(path).unwrap();

    let (x, _) = make_separable();
    let preds_orig = model.predict(&x).unwrap();
    let preds_loaded = loaded.predict(&x).unwrap();

    assert_eq!(
        preds_orig, preds_loaded,
        "loaded model must produce identical predictions"
    );

    // Decision scores are also bit-identical after round-trip
    let scores_orig = model.decision_function(&x).unwrap();
    let scores_loaded = loaded.decision_function(&x).unwrap();
    for i in 0..scores_orig.len() {
        assert!(
            (scores_orig[i] - scores_loaded[i]).abs() < 1e-12,
            "decision score[{i}] changed after save/load: {:.10} vs {:.10}",
            scores_orig[i],
            scores_loaded[i]
        );
    }

    let _ = std::fs::remove_file(path);
}

#[test]
fn load_from_nonexistent_path_returns_io_error() {
    let result = LinearSVC::load_from_path("/tmp/this_file_does_not_exist_rustyml_svc.json");
    assert!(
        matches!(result, Err(Error::Io(_))),
        "loading from non-existent path must return Err(Io(...))"
    );
}

#[test]
fn save_load_preserves_hyperparameters() {
    let path = "/tmp/rustyml_linear_svc_hparams_test.json";

    let model = fit_separable_model();
    model.save_to_path(path).unwrap();
    let loaded = LinearSVC::load_from_path(path).unwrap();

    assert_eq!(loaded.get_max_iterations(), model.get_max_iterations());
    assert_eq!(loaded.get_learning_rate(), model.get_learning_rate());
    assert_eq!(loaded.get_tolerance(), model.get_tolerance());
    assert_eq!(loaded.get_fit_intercept(), model.get_fit_intercept());
    assert_eq!(loaded.get_penalty(), model.get_penalty());

    let _ = std::fs::remove_file(path);
}

// correctness invariant: class-1 must be on positive side

/// On separable data, all class-1 samples score positive and all class-0 samples
/// score negative
#[test]
fn class_one_samples_score_positive_class_zero_samples_score_negative() {
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6).unwrap();
    model.fit(&x, &y).unwrap();

    let scores = model.decision_function(&x).unwrap();

    // First 4 samples are class 0, last 4 are class 1; the large margin means all
    // land on their respective sides of the hyperplane
    let mut all_correct = true;
    for i in 0..4 {
        if scores[i] >= 0.0 {
            all_correct = false;
        }
    }
    for i in 4..8 {
        if scores[i] <= 0.0 {
            all_correct = false;
        }
    }
    assert!(
        all_correct,
        "class-0 samples must score <0 and class-1 samples must score >0 on separable data; \
         scores = {:?}",
        scores
    );
}

// decision_function returns finite values

#[test]
fn decision_function_returns_finite_values() {
    let model = fit_separable_model();
    let (x, _) = make_separable();
    let scores = model.decision_function(&x).unwrap();
    for (i, &s) in scores.iter().enumerate() {
        assert!(s.is_finite(), "decision_function[{i}] = {s} is not finite");
    }
}

// reproducibility via random_state

// Training shuffles sample indices each epoch to form minibatches (batch size 32
// here), so the shuffle order affects the trained weights and the seed matters

/// 200 deterministic, roughly-separable samples
fn make_many() -> (Array2<f64>, Array1<f64>) {
    let n = 200;
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let class = (i % 2) as f64;
        x[[i, 0]] = if class == 1.0 { 5.0 } else { -5.0 } + (i as f64 * 0.017).sin();
        x[[i, 1]] = (i as f64 * 0.013).cos();
        y[i] = class;
    }
    (x, y)
}

/// Two models trained with the same seed produce bit-identical weights and bias
#[test]
fn same_random_state_is_reproducible() {
    let (x, y) = make_many();
    let train = |seed| {
        let mut m = LinearSVC::new(300, 0.01, RegularizationType::L2(0.1), true, 1e-9)
            .unwrap()
            .with_random_state(seed);
        m.fit(&x, &y).unwrap();
        m
    };
    let a = train(42);
    let b = train(42);
    assert_eq!(
        a.get_weights().unwrap(),
        b.get_weights().unwrap(),
        "identical seed must yield identical weights"
    );
    assert_eq!(
        a.get_bias(),
        b.get_bias(),
        "identical seed must yield identical bias"
    );
}

/// Different seeds drive different minibatch orders, so the trained weights differ
#[test]
fn different_random_state_changes_result() {
    let (x, y) = make_many();
    let train = |seed| {
        let mut m = LinearSVC::new(300, 0.01, RegularizationType::L2(0.1), true, 1e-9)
            .unwrap()
            .with_random_state(seed);
        m.fit(&x, &y).unwrap();
        m.get_weights().unwrap().clone()
    };
    assert_ne!(
        train(1),
        train(2),
        "different seeds should produce different weights on this dataset"
    );
}
// in-training weight-explosion guard (check_weights_validity)

/// A huge but finite learning rate on large-magnitude finite features trips the
/// in-training weight-explosion guard and surfaces Error::NonFinite
#[test]
fn fit_huge_learning_rate_on_large_finite_data_returns_non_finite() {
    // Balanced, separable, large-magnitude finite data
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            -1.0e8, 0.0, -1.0e8, 1.0, -1.0e8, -1.0, -1.0e8, 2.0, 1.0e8, 0.0, 1.0e8, 1.0, 1.0e8,
            -1.0, 1.0e8, 2.0,
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut model = LinearSVC::new(100, f64::MAX, RegularizationType::L2(1.0), true, 1e-6)
        .unwrap()
        .with_random_state(0);

    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "huge learning_rate must trip check_weights_validity, got {:?}",
        result
    );
}

// decision_function applies a non-zero fitted bias (fit_intercept=true)

/// Origin-shifted separable data forces a non-zero trained bias, so
/// decision_function(x)[i] must equal x.row(i).weights + bias from the getters
#[test]
fn decision_function_applies_nonzero_fitted_bias() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0, 0.0, // class 0
            14.0, 0.0, 15.0, 0.0, 16.0, 0.0, 17.0, 0.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut model =
        LinearSVC::new(10_000, 0.001, RegularizationType::L2(0.01), true, 1e-7).unwrap();
    model.fit(&x, &y).unwrap();

    let weights = model.get_weights().unwrap().clone();
    let bias = model.get_bias().unwrap();

    // The shifted boundary forces a clearly non-zero intercept, which makes the
    // identity below a real test that the bias is applied
    assert!(
        bias.abs() > 1e-6,
        "origin-shifted separable data must yield a non-zero bias, got {bias}"
    );

    let scores = model.decision_function(&x).unwrap();
    for i in 0..x.nrows() {
        let manual = x.row(i).dot(&weights) + bias;
        assert!(
            (scores[i] - manual).abs() < 1e-9,
            "decision_function[{i}]={:.10} != x·w+b={:.10} (bias={:.10})",
            scores[i],
            manual,
            bias
        );
    }
}

// squared-hinge loss option

use rustyml::machine_learning::Loss;

/// 120 deterministic, linearly-separable samples (enough rows that training spans several
/// minibatches per epoch)
fn make_separable_n(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let class = (i % 2) as f64;
        let s = if class == 1.0 { 1.5 } else { -1.5 };
        x[[i, 0]] = s + (i as f64 * 0.01).sin() * 0.2;
        x[[i, 1]] = (i as f64 * 0.017).cos() * 0.5;
        y[i] = class;
    }
    (x, y)
}

/// Mean squared-hinge loss + (λ/2)||w||² for y ∈ {-1,+1}
fn squared_hinge_objective(
    x: &Array2<f64>,
    y_pm1: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
    lambda: f64,
) -> f64 {
    let n = x.nrows() as f64;
    let mut loss = 0.0;
    for i in 0..x.nrows() {
        let margin = x.row(i).dot(w) + b;
        let s = (1.0 - y_pm1[i] * margin).max(0.0);
        loss += s * s;
    }
    loss / n + 0.5 * lambda * w.dot(w)
}

/// Reference full-batch GD that minimizes (1/n)Σ max(0,1-y·m)² + (λ/2)||w||², using the
/// independently-derived squared-hinge gradient d/dw = -2·max(0,1-y·m)·y·x
fn reference_squared_hinge_svm(
    x: &Array2<f64>,
    y_pm1: &Array1<f64>,
    lambda: f64,
    lr: f64,
    iters: usize,
) -> (Array1<f64>, f64) {
    let n = x.nrows();
    let d = x.ncols();
    let mut w = Array1::<f64>::zeros(d);
    let mut b = 0.0;
    for _ in 0..iters {
        let mut gw = Array1::<f64>::zeros(d);
        let mut gb = 0.0;
        for i in 0..n {
            let margin = x.row(i).dot(&w) + b;
            let s = 1.0 - y_pm1[i] * margin;
            if s > 0.0 {
                gw.scaled_add(-2.0 * s * y_pm1[i], &x.row(i));
                gb -= 2.0 * s * y_pm1[i];
            }
        }
        gw /= n as f64;
        gb /= n as f64;
        gw.scaled_add(lambda, &w);
        w.scaled_add(-lr, &gw);
        b -= lr * gb;
    }
    (w, b)
}

/// Default loss is Hinge; with_loss sets SquaredHinge
#[test]
fn loss_default_is_hinge_and_builder_sets_squared() {
    assert_eq!(LinearSVC::default().get_loss(), Loss::Hinge);
    let m = LinearSVC::default().with_loss(Loss::SquaredHinge);
    assert_eq!(m.get_loss(), Loss::SquaredHinge);
}

/// SquaredHinge still classifies perfectly separable data correctly
#[test]
fn squared_hinge_classifies_separable_data() {
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6)
        .unwrap()
        .with_loss(Loss::SquaredHinge);
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&x).unwrap();
    let truth = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (i, (&p, &t)) in preds.iter().zip(truth.iter()).enumerate() {
        assert_eq!(p, t, "sample {i}: expected {t}, got {p}");
    }
}

/// Hinge and squared-hinge are different objectives, so they yield different weights
#[test]
fn squared_hinge_differs_from_hinge() {
    let (x, y) = make_separable_n(120);
    let train = |loss| {
        let mut m = LinearSVC::new(20_000, 0.02, RegularizationType::L2(0.1), true, 1e-10)
            .unwrap()
            .with_loss(loss)
            .with_random_state(0);
        m.fit(&x, &y).unwrap();
        m.get_weights().unwrap().clone()
    };
    let w_hinge = train(Loss::Hinge);
    let w_sq = train(Loss::SquaredHinge);
    assert_ne!(w_hinge, w_sq, "the two losses must yield different weights");
}

/// Ground truth: with SquaredHinge, LinearSVC minimizes the squared-hinge objective,
/// reaching essentially the same value as an independent reference solver. A wrong
/// gradient would converge elsewhere and miss the reference optimum.
#[test]
fn squared_hinge_minimizes_its_objective() {
    let (x, y01) = make_separable_n(120);
    let y_pm1 = y01.mapv(|v| if v <= 0.0 { -1.0 } else { 1.0 });
    let lambda = 0.5;

    let (w_ref, b_ref) = reference_squared_hinge_svm(&x, &y_pm1, lambda, 0.02, 60_000);
    let j_ref = squared_hinge_objective(&x, &y_pm1, &w_ref, b_ref, lambda);

    let mut model = LinearSVC::new(60_000, 0.02, RegularizationType::L2(lambda), true, 1e-12)
        .unwrap()
        .with_loss(Loss::SquaredHinge)
        .with_random_state(0);
    model.fit(&x, &y01).unwrap();
    let j_svc = squared_hinge_objective(
        &x,
        &y_pm1,
        model.get_weights().unwrap(),
        model.get_bias().unwrap(),
        lambda,
    );

    assert!(
        j_svc <= j_ref + 0.05,
        "squared-hinge objective {j_svc:.6} is far from the reference optimum {j_ref:.6}; \
         the squared-hinge gradient is likely wrong"
    );
}
