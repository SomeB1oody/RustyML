use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::linear_svc::{LinearSVC, RegularizationType};

// ─── helper ────────────────────────────────────────────────────────────────

/// Return (x, y) for a perfectly linearly-separable 2-feature problem.
///
/// Class 0 (y=0.0): points at x1 = -5, -6, -7, -4  (far left)
/// Class 1 (y=1.0): points at x1 = +5, +6, +7, +4  (far right)
///
/// By construction, a linear classifier need only learn that w[0] > 0 separates
/// the two classes (the boundary x1=0 separates them with margin 4). We KNOW:
///   - All class-1 training samples satisfy x·w + b > 0  → predict = 1.0
///   - All class-0 training samples satisfy x·w + b < 0  → predict = 0.0
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

/// Fit a default (L2) LinearSVC on the separable dataset and return the trained
/// model, using enough iterations that convergence is expected.
fn fit_separable_model() -> LinearSVC {
    let (x, y) = make_separable();
    let mut model =
        LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), true, 1e-5, None).unwrap();
    model.fit(&x, &y).unwrap();
    model
}

// ─── constructor validation ────────────────────────────────────────────────

#[test]
fn new_rejects_max_iter_zero() {
    let result = LinearSVC::new(0, 0.01, RegularizationType::L2(1.0), true, 1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "max_iter=0 must return InvalidParameter"
    );
}

#[test]
fn new_rejects_learning_rate_zero() {
    let result = LinearSVC::new(100, 0.0, RegularizationType::L2(1.0), true, 1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "learning_rate=0.0 must return InvalidParameter"
    );
}

#[test]
fn new_rejects_learning_rate_negative() {
    let result = LinearSVC::new(100, -0.001, RegularizationType::L2(1.0), true, 1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "learning_rate=-0.001 must return InvalidParameter"
    );
}

#[test]
fn new_rejects_learning_rate_nan() {
    let result = LinearSVC::new(100, f64::NAN, RegularizationType::L2(1.0), true, 1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "learning_rate=NaN must return InvalidParameter"
    );
}

#[test]
fn new_rejects_learning_rate_infinity() {
    let result = LinearSVC::new(
        100,
        f64::INFINITY,
        RegularizationType::L2(1.0),
        true,
        1e-4,
        None,
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "learning_rate=+Inf must return InvalidParameter"
    );
}

#[test]
fn new_rejects_tol_zero() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L2(1.0), true, 0.0, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol=0.0 must return InvalidParameter"
    );
}

#[test]
fn new_rejects_tol_negative() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L2(1.0), true, -1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol=-1e-4 must return InvalidParameter"
    );
}

#[test]
fn new_rejects_tol_nan() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L2(1.0), true, f64::NAN, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "tol=NaN must return InvalidParameter"
    );
}

#[test]
fn new_rejects_penalty_lambda_negative_l2() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L2(-0.5), true, 1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "L2 with negative lambda must return InvalidParameter"
    );
}

#[test]
fn new_rejects_penalty_lambda_negative_l1() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L1(-1.0), true, 1e-4, None);
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "L1 with negative lambda must return InvalidParameter"
    );
}

#[test]
fn new_rejects_penalty_lambda_nan() {
    let result = LinearSVC::new(
        100,
        0.01,
        RegularizationType::L2(f64::NAN),
        true,
        1e-4,
        None,
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "L2(NaN) must return InvalidParameter"
    );
}

#[test]
fn new_rejects_penalty_lambda_infinity() {
    let result = LinearSVC::new(
        100,
        0.01,
        RegularizationType::L1(f64::INFINITY),
        true,
        1e-4,
        None,
    );
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "L1(+Inf) must return InvalidParameter"
    );
}

/// Lambda = 0.0 is explicitly allowed (no regularization).
#[test]
fn new_accepts_zero_penalty_lambda() {
    let result = LinearSVC::new(100, 0.01, RegularizationType::L2(0.0), true, 1e-4, None);
    assert!(result.is_ok(), "L2(0.0) must be accepted");
}

/// Verify all defaults match documented values.
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

/// Constructor stores the passed parameters exactly.
#[test]
fn new_stores_parameters() {
    let model = LinearSVC::new(500, 0.005, RegularizationType::L1(0.1), false, 1e-6, None).unwrap();
    assert_eq!(model.get_max_iterations(), 500);
    assert_eq!(model.get_learning_rate(), 0.005);
    assert_eq!(model.get_penalty(), RegularizationType::L1(0.1));
    assert!(!model.get_fit_intercept());
    assert_eq!(model.get_tolerance(), 1e-6);
}

// ─── fit error paths ───────────────────────────────────────────────────────

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
    // preliminary_check fires dimension mismatch (y.len()=0 != x.nrows()=3)
    // or EmptyInput depending on which guard fires first — either is valid
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
fn fit_rejects_nan_in_x() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, f64::NAN, 4.0]).unwrap();
    let y = array![0.0, 1.0];
    let mut model = LinearSVC::default();
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "NaN in x must return NonFinite"
    );
}

#[test]
fn fit_rejects_infinite_in_x() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, f64::INFINITY, 4.0]).unwrap();
    let y = array![0.0, 1.0];
    let mut model = LinearSVC::default();
    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "Inf in x must return NonFinite"
    );
}

// ─── predict / decision_function error paths ──────────────────────────────

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

// ─── output label domain: must be exactly 0.0 or 1.0 ─────────────────────

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

// ─── sign-consistency invariant ───────────────────────────────────────────
//
// CONTRACT: predict(x)==1.0  iff  decision_function(x)>0.0
//           predict(x)==0.0  iff  decision_function(x)<=0.0
//
// This is enforced by the source:
//   Ok(decision.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))

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
    // Craft points we know the sign for independently:
    //   [+10, 0] → should score > 0 → class 1
    //   [-10, 0] → should score < 0 → class 0
    // We verify the invariant holds, not a hardcoded class value, because the
    // classifier might flip the sign axis — but both samples must be consistent.
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

// ─── correctness on separable data ────────────────────────────────────────

/// On perfectly linearly-separable data the classifier must predict the true
/// class for EVERY training sample.  We know the true classes by construction,
/// and we use a generous tolerance + iteration budget so convergence is
/// reliable.
#[test]
fn predicts_all_training_samples_correctly_on_separable_data() {
    let (x, y) = make_separable();
    let mut model =
        LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6, None).unwrap();
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

/// Fit-predict convenience method must agree with fit+predict on same data.
#[test]
fn fit_predict_agrees_with_fit_then_predict() {
    let (x, y) = make_separable();
    // model A: fit_predict
    let mut model_a =
        LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), true, 1e-5, None).unwrap();
    let preds_a = model_a.fit_predict(&x, &y).unwrap();
    // model B: fit then predict on same data
    let preds_b = model_a.predict(&x).unwrap();
    assert_eq!(preds_a, preds_b, "fit_predict and fit→predict must agree");
}

// ─── getters after fit ────────────────────────────────────────────────────

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
    let mut model = LinearSVC::new(
        max_iter,
        0.01,
        RegularizationType::L2(0.1),
        true,
        1e-5,
        None,
    )
    .unwrap();
    model.fit(&x, &y).unwrap();
    let n = model.get_actual_iterations().unwrap();
    assert!(n >= 1, "n_iter must be at least 1");
    assert!(
        n <= max_iter,
        "n_iter {n} must not exceed max_iter {max_iter}"
    );
}

/// With very tight tolerance and lots of iterations on well-separated data,
/// early stopping should fire before max_iter is exhausted.
#[test]
fn convergence_stops_before_max_iter_on_separable_data() {
    let max_iter = 50_000usize;
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(
        max_iter,
        0.01,
        RegularizationType::L2(0.01),
        true,
        1e-3,
        None,
    )
    .unwrap();
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
    // Training data has 2 features → weight vector must have length 2
    assert_eq!(
        weights.len(),
        2,
        "weight vector must match number of training features"
    );
}

// ─── fit_intercept = false ────────────────────────────────────────────────

/// When fit_intercept=false the bias is never updated and must remain 0.0.
#[test]
fn fit_intercept_false_bias_stays_zero() {
    // Use data centred at origin so the zero-bias model can still solve the
    // problem: class 0 at negative x1, class 1 at positive x1.
    let (x, y) = make_separable();
    let mut model = LinearSVC::new(
        5000,
        0.01,
        RegularizationType::L2(0.1),
        false, // no intercept
        1e-5,
        None,
    )
    .unwrap();
    model.fit(&x, &y).unwrap();
    let bias = model.get_bias().unwrap();
    assert_eq!(
        bias, 0.0,
        "bias must stay exactly 0.0 when fit_intercept=false"
    );
}

/// When fit_intercept=false, decision_function(x) = x · weights exactly
/// (bias is 0.0 and therefore adds nothing).
#[test]
fn fit_intercept_false_decision_function_equals_dot_product() {
    let (x, y) = make_separable();
    let mut model =
        LinearSVC::new(5000, 0.01, RegularizationType::L2(0.1), false, 1e-5, None).unwrap();
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

// ─── L1 vs L2 penalty produce different weight vectors ────────────────────

/// L1 and L2 penalties with the same lambda on the same data must yield
/// different weight vectors (the regularization gradient is different for each).
#[test]
fn l1_and_l2_penalties_produce_different_weights() {
    let (x, y) = make_separable();

    let mut model_l2 =
        LinearSVC::new(3000, 0.01, RegularizationType::L2(1.0), true, 1e-6, None).unwrap();
    model_l2.fit(&x, &y).unwrap();
    let w_l2 = model_l2.get_weights().unwrap().clone();

    let mut model_l1 =
        LinearSVC::new(3000, 0.01, RegularizationType::L1(1.0), true, 1e-6, None).unwrap();
    model_l1.fit(&x, &y).unwrap();
    let w_l1 = model_l1.get_weights().unwrap().clone();

    assert_ne!(
        w_l1, w_l2,
        "L1 and L2 regularization must yield different weight vectors"
    );
}

/// Strong L1 regularization on this 2-feature dataset (only feature 0 matters)
/// should drive the irrelevant feature (feature 1, all zeros) toward zero.
/// We verify that |w_l1[1]| < |w_l1[0]| — the relevant feature dominates.
/// (We cannot assert exact zero since SGD may not fully converge, but the
/// irrelevant feature must be closer to zero than the relevant one.)
#[test]
fn l1_sparsity_irrelevant_feature_closer_to_zero() {
    let (x, y) = make_separable();
    // x column 1 is all zeros — it carries no information.
    // A strong L1 penalty should shrink w[1] toward zero.
    let mut model = LinearSVC::new(
        10_000,
        0.001,
        RegularizationType::L1(10.0),
        true,
        1e-7,
        None,
    )
    .unwrap();
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

// ─── save / load round-trip ───────────────────────────────────────────────

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

    // Verify decision scores are also bit-identical after round-trip.
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

    // Clean up
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

// ─── correctness invariant: class-1 must be on positive side ──────────────

/// After training on fully separable data with the designed dataset, we verify
/// that all class-1 samples score positively and all class-0 samples score
/// negatively (the margin is large enough that a converged linear SVC must
/// separate them).
#[test]
fn class_one_samples_score_positive_class_zero_samples_score_negative() {
    let (x, y) = make_separable();
    let mut model =
        LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6, None).unwrap();
    model.fit(&x, &y).unwrap();

    let scores = model.decision_function(&x).unwrap();

    // First 4 samples: true class 0 (y=0.0), last 4: true class 1 (y=1.0)
    // Since the data is fully separable with large margin, all should be
    // correctly placed on their respective sides of the hyperplane.
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

// ─── decision_function returns finite values ──────────────────────────────

#[test]
fn decision_function_returns_finite_values() {
    let model = fit_separable_model();
    let (x, _) = make_separable();
    let scores = model.decision_function(&x).unwrap();
    for (i, &s) in scores.iter().enumerate() {
        assert!(s.is_finite(), "decision_function[{i}] = {s} is not finite");
    }
}

// ─── reproducibility via random_state ──────────────────────────────────────
//
// Training shuffles the sample indices once per epoch to form minibatches. With
// 200 samples the batch size is clamp(200/10, 32, 512) = 32, so ~7 minibatches
// form and the shuffle order genuinely affects the trained weights — which makes
// this a real test of the random_state seeding rather than a no-op.

/// 200 deterministic, roughly-separable samples (no RNG needed in the test).
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

/// Two models trained with the SAME seed must produce bit-identical weights/bias.
#[test]
fn same_random_state_is_reproducible() {
    let (x, y) = make_many();
    let train = |seed| {
        let mut m = LinearSVC::new(
            300,
            0.01,
            RegularizationType::L2(0.1),
            true,
            1e-9,
            Some(seed),
        )
        .unwrap();
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

/// Different seeds drive different minibatch orders, so on this data the trained
/// weights must differ — proving the shuffle (and therefore the seed) is wired in.
#[test]
fn different_random_state_changes_result() {
    let (x, y) = make_many();
    let train = |seed| {
        let mut m = LinearSVC::new(
            300,
            0.01,
            RegularizationType::L2(0.1),
            true,
            1e-9,
            Some(seed),
        )
        .unwrap();
        m.fit(&x, &y).unwrap();
        m.get_weights().unwrap().clone()
    };
    assert_ne!(
        train(1),
        train(2),
        "different seeds should produce different weights on this dataset"
    );
}
// ─── in-training weight-explosion guard (check_weights_validity) ───────────

/// A huge but finite learning rate on large-magnitude finite features must trip
/// the in-training weight-explosion guard `check_weights_validity` and surface
/// `Error::NonFinite`.
///
/// Derivation (not by running the model):
/// - The data is finite (so `preliminary_check` passes) and `learning_rate =
///   f64::MAX` is positive and finite (so the constructor passes). Any
///   `NonFinite` therefore comes only from the in-training guard at the end of
///   the batch loop (`Self::check_weights_validity(&weights, bias)?`).
/// - On the very first batch, `weight_grad_sum` has magnitude on the order of
///   |x| ≈ 1e8 summed over the batch. The update
///   `w ← 0 + weight_grad_sum * (f64::MAX / batch_len)` overflows to ±∞, and the
///   subsequent L2 step `w * (1 - f64::MAX * lambda)` keeps it non-finite. The
///   guard then returns `NonFinite`.
#[test]
fn fit_huge_learning_rate_on_large_finite_data_returns_non_finite() {
    // Balanced, separable, *large-magnitude* finite data.
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            -1.0e8, 0.0, -1.0e8, 1.0, -1.0e8, -1.0, -1.0e8, 2.0, 1.0e8, 0.0, 1.0e8, 1.0, 1.0e8,
            -1.0, 1.0e8, 2.0,
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut model = LinearSVC::new(
        100,
        f64::MAX,
        RegularizationType::L2(1.0),
        true,
        1e-6,
        Some(0),
    )
    .unwrap();

    let result = model.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NonFinite(_))),
        "huge learning_rate must trip check_weights_validity, got {:?}",
        result
    );
}

// ─── decision_function applies a NON-zero fitted bias (fit_intercept=true) ──

/// Asymmetric, separable, origin-shifted data forces a NON-zero trained bias;
/// `decision_function(x)[i]` must then equal `x.row(i)·weights + bias` using the
/// model's OWN getters — proving the intercept is actually applied (the existing
/// numeric df test forces bias==0 with fit_intercept=false).
///
/// Design / ground truth:
/// - All points sit at positive x1 (class 0 at x1∈{8,9,10,11}, class 1 at
///   x1∈{14,15,16,17}), x2=0 carries no information. A hyperplane through the
///   origin `w0*x1` would give EVERY point the same sign (all x1>0), so it
///   cannot separate the classes: a correct boundary at x1≈12.5 requires
///   `w0*12.5 + b = 0` ⇒ `b ≈ -12.5*w0 ≠ 0`. The bias is unregularized (the L2
///   step scales only `weights`), so with 10_000 iterations it grows to order
///   ~10 — far above the 1e-6 floor we assert. This is a geometric necessity,
///   not an impl-traced number.
/// - The decision function is defined as `X·w + b`; the test recomputes that
///   per-row from the model's getters and checks equality, which only passes
///   if the source adds the (non-zero) intercept.
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

    let mut model = LinearSVC::new(
        10_000,
        0.001,
        RegularizationType::L2(0.01),
        true,
        1e-7,
        None,
    )
    .unwrap();
    model.fit(&x, &y).unwrap();

    let weights = model.get_weights().unwrap().clone();
    let bias = model.get_bias().unwrap();

    // The shifted boundary forces a clearly non-zero intercept — this is what
    // makes the identity below a real test of "the bias is applied".
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
