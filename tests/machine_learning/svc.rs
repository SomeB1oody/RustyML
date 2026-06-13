//! Integration tests for SVC (Support Vector Classifier)
//!
//! Label-domain contract: fit requires labels in {+1.0, -1.0}, predict emits
//! labels in {+1.0, -1.0}, and sign(decision_function[i]) >= 0 <-> predict[i] == +1.0

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::{KernelType, SVC};

// helpers

/// Linearly separable 2-feature dataset in the +1/-1 label domain
///
/// Class +1 sits in the upper-right quadrant, class -1 in the lower-left, so a
/// large-margin linear-kernel SVC classifies every point with zero error
fn linearly_separable_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, // class +1
            -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -3.0, -3.0, // class -1
        ],
    )
    .unwrap();
    let y = array![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
    (x, y)
}

/// Concentric-rings dataset (inner ring class +1, outer ring class -1) that a
/// linear kernel cannot separate but an RBF kernel with gamma=0.5 can
fn concentric_rings_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, // inner ring (radius 1), class +1
            // Radius 5 (not 3) keeps every inner-inner distance strictly below every inner-outer
            // distance, so RBF can separate the rings (at radius 3 some points are equidistant)
            5.0, 0.0, -5.0, 0.0, 0.0, 5.0, 0.0, -5.0, // outer ring (radius 5), class -1
        ],
    )
    .unwrap();
    let y = array![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
    (x, y)
}

// constructor validation

/// C = 0.0 is non-positive -> InvalidParameter
#[test]
fn new_rejects_zero_c() {
    let result = SVC::new(KernelType::Linear, 0.0, 1e-3, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// C = -1.0 is negative -> InvalidParameter
#[test]
fn new_rejects_negative_c() {
    let result = SVC::new(KernelType::Linear, -1.0, 1e-3, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// C = NaN -> InvalidParameter (non-finite check)
#[test]
fn new_rejects_nan_c() {
    let result = SVC::new(KernelType::Linear, f64::NAN, 1e-3, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// C = +Inf -> InvalidParameter (non-finite check)
#[test]
fn new_rejects_inf_c() {
    let result = SVC::new(KernelType::Linear, f64::INFINITY, 1e-3, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tol = 0.0 is non-positive -> InvalidParameter
#[test]
fn new_rejects_zero_tol() {
    let result = SVC::new(KernelType::Linear, 1.0, 0.0, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tol = -1e-3 is negative -> InvalidParameter
#[test]
fn new_rejects_negative_tol() {
    let result = SVC::new(KernelType::Linear, 1.0, -1e-3, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// tol = NaN -> InvalidParameter
#[test]
fn new_rejects_nan_tol() {
    let result = SVC::new(KernelType::Linear, 1.0, f64::NAN, 100, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// max_iter = 0 -> InvalidParameter
#[test]
fn new_rejects_zero_max_iter() {
    let result = SVC::new(KernelType::Linear, 1.0, 1e-3, 0, Some(42));
    assert!(
        matches!(result, Err(Error::InvalidParameter { .. })),
        "expected InvalidParameter, got {:?}",
        result
    );
}

/// Valid parameters succeed and getters echo them back
#[test]
fn new_valid_parameters_round_trip() {
    let svc = SVC::new(KernelType::RBF { gamma: 0.5 }, 2.0, 1e-3, 500, Some(7))
        .expect("valid params must succeed");

    assert_abs_diff_eq!(svc.get_regularization_parameter(), 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(svc.get_tolerance(), 1e-3, epsilon = 1e-12);
    assert_eq!(svc.get_max_iterations(), 500);
    assert_eq!(svc.get_random_state(), Some(7));
}

/// Default constructor exposes the documented field values
#[test]
fn default_has_expected_params() {
    let svc = SVC::default();
    assert_abs_diff_eq!(svc.get_regularization_parameter(), 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(svc.get_tolerance(), 0.001, epsilon = 1e-12);
    assert_eq!(svc.get_max_iterations(), 1000);
    // not yet fitted
    assert!(svc.get_support_vectors().is_none());
    assert!(svc.get_alphas().is_none());
    assert!(svc.get_bias().is_none());
    assert!(svc.get_actual_iterations().is_none());
}

// fit label-domain validation

/// fit rejects labels that are not exactly +/-1.0 (e.g. 0.0/1.0 domain)
#[test]
fn fit_rejects_labels_not_plus_minus_one() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]).unwrap();
    // 0-based binary labels, not valid for SVC
    let y = array![0.0, 0.0, 1.0, 1.0];
    let mut svc = SVC::new(KernelType::Linear, 1.0, 1e-3, 100, Some(42)).unwrap();
    let result = svc.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for non-±1 labels, got {:?}",
        result
    );
}

/// fit rejects labels containing fractional values (e.g. 0.5)
#[test]
fn fit_rejects_fractional_labels() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]).unwrap();
    let y = array![1.0, -1.0, 0.5, -0.5];
    let mut svc = SVC::new(KernelType::Linear, 1.0, 1e-3, 100, Some(42)).unwrap();
    let result = svc.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::InvalidInput(_))),
        "expected InvalidInput for fractional labels, got {:?}",
        result
    );
}

// predict / decision_function before fit

/// predict before fit -> NotFitted
#[test]
fn predict_before_fit_returns_not_fitted() {
    let svc = SVC::new(KernelType::Linear, 1.0, 1e-3, 100, Some(42)).unwrap();
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, -1.0, 0.0]).unwrap();
    let result = svc.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {:?}",
        result
    );
}

/// decision_function before fit -> NotFitted
#[test]
fn decision_function_before_fit_returns_not_fitted() {
    let svc = SVC::new(KernelType::Linear, 1.0, 1e-3, 100, Some(42)).unwrap();
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, -1.0, 0.0]).unwrap();
    let result = svc.decision_function(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {:?}",
        result
    );
}

// predict / decision_function dimension checks

/// predict with wrong feature dimension -> DimensionMismatch
#[test]
fn predict_wrong_feature_dim_returns_dimension_mismatch() {
    let (x_train, y_train) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 500, Some(42)).unwrap();
    svc.fit(&x_train, &y_train).expect("fit must succeed");

    // training used 2 features; pass 3-feature input
    let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]).unwrap();
    let result = svc.predict(&x_bad);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {:?}",
        result
    );
}

/// decision_function with wrong feature dimension -> DimensionMismatch
#[test]
fn decision_function_wrong_feature_dim_returns_dimension_mismatch() {
    let (x_train, y_train) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 500, Some(42)).unwrap();
    svc.fit(&x_train, &y_train).expect("fit must succeed");

    let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]).unwrap();
    let result = svc.decision_function(&x_bad);
    assert!(
        matches!(result, Err(Error::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {:?}",
        result
    );
}

// linear kernel: linearly separable data

/// A linear-kernel SVC on well-separated data classifies every training point
/// correctly against the by-construction labels
#[test]
fn linear_kernel_classifies_separable_data_perfectly() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y)
        .expect("fit must succeed on linearly separable data");

    let preds = svc.predict(&x).expect("predict must succeed");

    // by construction the correct labels are exactly y
    for (i, (&pred, &true_label)) in preds.iter().zip(y.iter()).enumerate() {
        assert_eq!(
            pred, true_label,
            "sample {i}: predicted {pred} but true label is {true_label}"
        );
    }
}

/// predict output is strictly within the label domain {+1.0, -1.0}
#[test]
fn predict_output_domain_is_plus_minus_one() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let preds = svc.predict(&x).expect("predict must succeed");
    for &p in preds.iter() {
        assert!(
            p == 1.0 || p == -1.0,
            "predict returned {p}, which is not in {{+1.0, -1.0}}"
        );
    }
}

/// sign(decision_function[i]) >= 0 <-> predict[i] == +1.0 for the linear kernel
#[test]
fn sign_consistency_linear_kernel() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let df = svc
        .decision_function(&x)
        .expect("decision_function must succeed");
    let preds = svc.predict(&x).expect("predict must succeed");

    for (i, (&dv, &p)) in df.iter().zip(preds.iter()).enumerate() {
        let expected_pred = if dv >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        assert_eq!(
            p, expected_pred,
            "sample {i}: decision_value={dv}, predict={p} disagrees with sign"
        );
    }
}

/// After fit, getters for support vectors, alphas, and bias are Some
#[test]
fn linear_kernel_fit_populates_state() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    assert!(
        svc.get_support_vectors().is_some(),
        "support_vectors must be Some after fit"
    );
    assert!(svc.get_alphas().is_some(), "alphas must be Some after fit");
    assert!(svc.get_bias().is_some(), "bias must be Some after fit");
    assert!(
        svc.get_actual_iterations().is_some(),
        "n_iter must be Some after fit"
    );
}

/// n_iter after fit is in [1, max_iter]
#[test]
fn actual_iterations_in_valid_range() {
    let (x, y) = linearly_separable_data();
    let max_iter = 1000_usize;
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, max_iter, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let n_iter = svc
        .get_actual_iterations()
        .as_ref()
        .copied()
        .expect("n_iter must be Some after fit");
    assert!(
        n_iter >= 1 && n_iter <= max_iter,
        "n_iter={n_iter} is outside [1, {max_iter}]"
    );
}

// RBF kernel: non-linearly separable concentric rings

/// An RBF-kernel SVC classifies every concentric-ring point into its known ring
/// (class +1 inner, class -1 outer), which a linear kernel cannot
#[test]
fn rbf_kernel_classifies_concentric_rings_perfectly() {
    let (x, y) = concentric_rings_data();
    let mut svc = SVC::new(KernelType::RBF { gamma: 0.5 }, 10.0, 1e-3, 5000, Some(42)).unwrap();
    svc.fit(&x, &y)
        .expect("fit must succeed on concentric rings");

    let preds = svc.predict(&x).expect("predict must succeed");
    for (i, (&pred, &true_label)) in preds.iter().zip(y.iter()).enumerate() {
        assert_eq!(
            pred, true_label,
            "sample {i}: RBF SVC predicted {pred} but true label is {true_label}"
        );
    }
}

/// sign consistency holds for the RBF kernel too
#[test]
fn sign_consistency_rbf_kernel() {
    let (x, y) = concentric_rings_data();
    let mut svc = SVC::new(KernelType::RBF { gamma: 0.5 }, 10.0, 1e-3, 2000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let df = svc
        .decision_function(&x)
        .expect("decision_function must succeed");
    let preds = svc.predict(&x).expect("predict must succeed");

    for (i, (&dv, &p)) in df.iter().zip(preds.iter()).enumerate() {
        let expected_pred = if dv >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        assert_eq!(
            p, expected_pred,
            "sample {i}: RBF decision_value={dv}, predict={p} disagrees with sign"
        );
    }
}

// all five kernels: construct, fit, predict

/// All five kernel variants construct, train, and predict on the
/// linearly-separable data without error
#[test]
fn all_kernels_fit_and_predict_without_error() {
    let kernels: &[KernelType] = &[
        KernelType::Linear,
        KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 1.0,
        },
        KernelType::RBF { gamma: 0.5 },
        KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        },
        KernelType::Cosine,
    ];

    let (x, y) = linearly_separable_data();

    for kernel in kernels {
        let mut svc = SVC::new(*kernel, 5.0, 1e-3, 1000, Some(42))
            .expect("constructor must succeed for all kernel variants");
        svc.fit(&x, &y)
            .unwrap_or_else(|e| panic!("fit failed for kernel {:?}: {e}", kernel));
        let preds = svc
            .predict(&x)
            .unwrap_or_else(|e| panic!("predict failed for kernel {:?}: {e}", kernel));
        // predictions must be in the correct domain
        for &p in preds.iter() {
            assert!(
                p == 1.0 || p == -1.0,
                "kernel {:?} returned label {p} outside {{±1.0}}",
                kernel
            );
        }
    }
}

// Polynomial kernel correctness

/// A degree-2 polynomial kernel separates the linearly-separable data and
/// classifies every training point into its known true class
#[test]
fn poly_kernel_classifies_separable_data_correctly() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(
        KernelType::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 1.0,
        },
        10.0,
        1e-3,
        1000,
        Some(42),
    )
    .unwrap();
    svc.fit(&x, &y).expect("fit must succeed");
    let preds = svc.predict(&x).expect("predict must succeed");
    for (i, (&pred, &true_label)) in preds.iter().zip(y.iter()).enumerate() {
        assert_eq!(
            pred, true_label,
            "Poly sample {i}: predicted {pred} but true label is {true_label}"
        );
    }
}

// Cosine kernel: zero-vector robustness

/// The Cosine kernel tolerates a zero vector in the data without panic or
/// non-finite values; no specific prediction value is asserted
#[test]
fn cosine_kernel_zero_vector_does_not_panic() {
    // zero vector in the training set, labelled +1 (arbitrary but must be +/-1)
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            2.0, 2.0, 3.0, 3.0, // class +1
            0.0, 0.0, // class +1, zero vector
            -2.0, -2.0, -3.0, -3.0, -4.0, -4.0, // class -1
        ],
    )
    .unwrap();
    let y = array![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

    let mut svc = SVC::new(KernelType::Cosine, 5.0, 1e-3, 1000, Some(42)).unwrap();
    // fit may converge or not; what matters is that it does not panic
    let _ = svc.fit(&x, &y);
}

// Reproducibility with fixed random_state

/// Two runs with the same seed produce identical predictions, decision values,
/// and bias
#[test]
fn same_seed_produces_identical_results() {
    let (x, y) = linearly_separable_data();

    let mut svc1 = SVC::new(KernelType::RBF { gamma: 0.5 }, 5.0, 1e-3, 1000, Some(42)).unwrap();
    svc1.fit(&x, &y).expect("first fit must succeed");

    let mut svc2 = SVC::new(KernelType::RBF { gamma: 0.5 }, 5.0, 1e-3, 1000, Some(42)).unwrap();
    svc2.fit(&x, &y).expect("second fit must succeed");

    // predictions must be identical
    let preds1 = svc1.predict(&x).expect("predict1 must succeed");
    let preds2 = svc2.predict(&x).expect("predict2 must succeed");
    assert_eq!(preds1, preds2, "same seed must yield identical predictions");

    // decision function values must be identical
    let df1 = svc1.decision_function(&x).expect("df1 must succeed");
    let df2 = svc2.decision_function(&x).expect("df2 must succeed");
    for (a, b) in df1.iter().zip(df2.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }

    // bias must be identical
    assert_abs_diff_eq!(
        svc1.get_bias().unwrap(),
        svc2.get_bias().unwrap(),
        epsilon = 1e-12
    );
}

// fit_predict convenience method

/// fit_predict returns the same array as fit then predict on the same data
#[test]
fn fit_predict_agrees_with_fit_then_predict() {
    let (x, y) = linearly_separable_data();

    let mut svc_a = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    let fp_preds = svc_a.fit_predict(&x, &y).expect("fit_predict must succeed");

    let mut svc_b = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc_b.fit(&x, &y).expect("fit must succeed");
    let preds = svc_b.predict(&x).expect("predict must succeed");

    assert_eq!(fp_preds, preds, "fit_predict must match fit+predict");
}

// save / load round-trip

/// Saving a fitted model and loading it back yields identical predictions and
/// decision-function values
#[test]
fn save_load_round_trip_yields_identical_predictions() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::RBF { gamma: 0.5 }, 5.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let original_preds = svc.predict(&x).expect("predict must succeed before save");
    let original_df = svc
        .decision_function(&x)
        .expect("df must succeed before save");

    // write to a temporary path
    let path = "/tmp/rustyml_svc_test_roundtrip.json";
    svc.save_to_path(path).expect("save_to_path must succeed");

    let loaded = SVC::load_from_path(path).expect("load_from_path must succeed");

    let loaded_preds = loaded.predict(&x).expect("predict must succeed after load");
    let loaded_df = loaded
        .decision_function(&x)
        .expect("df must succeed after load");

    assert_eq!(
        original_preds, loaded_preds,
        "predictions must match after round-trip"
    );
    for (a, b) in original_df.iter().zip(loaded_df.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-10);
    }

    // hyperparameters must survive round-trip
    assert_abs_diff_eq!(
        svc.get_regularization_parameter(),
        loaded.get_regularization_parameter(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        svc.get_bias().unwrap(),
        loaded.get_bias().unwrap(),
        epsilon = 1e-12
    );

    // clean up the temp file; failure to delete is non-fatal for test correctness
    let _ = std::fs::remove_file(path);
}

/// A linear-kernel model round-trips identically through save/load, exercising
/// kernel serialization
#[test]
fn save_load_round_trip_linear_kernel() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let path = "/tmp/rustyml_svc_test_linear_roundtrip.json";
    svc.save_to_path(path).expect("save_to_path must succeed");
    let loaded = SVC::load_from_path(path).expect("load_from_path must succeed");

    let preds_orig = svc.predict(&x).expect("original predict");
    let preds_load = loaded.predict(&x).expect("loaded predict");
    assert_eq!(
        preds_orig, preds_load,
        "linear kernel round-trip must match"
    );

    let _ = std::fs::remove_file(path);
}

// EmptyInput guard

/// predict on an empty input matrix -> EmptyInput
#[test]
fn predict_empty_input_returns_error() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let x_empty = Array2::<f64>::zeros((0, 2));
    let result = svc.predict(&x_empty);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput for zero-row input, got {:?}",
        result
    );
}

/// decision_function on an empty input matrix -> EmptyInput
#[test]
fn decision_function_empty_input_returns_error() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let x_empty = Array2::<f64>::zeros((0, 2));
    let result = svc.decision_function(&x_empty);
    assert!(
        matches!(result, Err(Error::EmptyInput(_))),
        "expected EmptyInput for zero-row input, got {:?}",
        result
    );
}

// Sigmoid kernel: construct, fit, predict

/// Sigmoid-kernel SVC on linearly-separable data: each predicted label agrees
/// with the sign of its decision value
#[test]
fn sigmoid_kernel_sign_consistency() {
    let (x, y) = linearly_separable_data();
    let mut svc = SVC::new(
        KernelType::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        },
        5.0,
        1e-3,
        1000,
        Some(42),
    )
    .unwrap();
    svc.fit(&x, &y).expect("fit must succeed");

    let df = svc
        .decision_function(&x)
        .expect("decision_function must succeed");
    let preds = svc.predict(&x).expect("predict must succeed");

    for (i, (&dv, &p)) in df.iter().zip(preds.iter()).enumerate() {
        let expected_pred = if dv >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        assert_eq!(
            p, expected_pred,
            "sample {i}: Sigmoid decision_value={dv}, predict={p} disagrees"
        );
    }
}

// fit: no support vectors (single-class data) -> NotConverged

/// Single-class training data yields an empty support set, so fit surfaces
/// NotConverged rather than a degenerate all-zero model
#[test]
fn fit_single_class_data_returns_not_converged() {
    let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0]];
    let y = array![1.0, 1.0, 1.0, 1.0]; // one class only
    let mut svc = SVC::new(KernelType::Linear, 1.0, 1e-3, 100, Some(42)).unwrap();
    let result = svc.fit(&x, &y);
    assert!(
        matches!(result, Err(Error::NotConverged(_))),
        "single-class data must yield no support vectors → NotConverged, got {result:?}"
    );
}

// decision_function / bias: closed-form ground truth (linear kernel)

/// On a 1-D separable set whose max-margin solution is f(x) = x - 1, bias and
/// decision values match the independently derived non-zero ground truth
#[test]
fn decision_function_and_bias_match_closed_form_linear_kernel() {
    let x = array![[0.0], [-1.0], [2.0], [3.0]];
    let y = array![-1.0, -1.0, 1.0, 1.0];
    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-5, 5000, Some(7)).unwrap();
    svc.fit(&x, &y).expect("separable data must fit");

    // asymmetric placement forces a non-zero intercept: b = -1
    assert_abs_diff_eq!(svc.get_bias().unwrap(), -1.0, epsilon = 1e-2);

    // f(x) = x - 1 at the two support vectors (the +/-1 margins) and the boundary
    let probe = array![[0.0], [2.0], [1.0]];
    let df = svc
        .decision_function(&probe)
        .expect("decision_function must succeed");
    assert_abs_diff_eq!(df[0], -1.0, epsilon = 1e-2); // SV on the -1 margin
    assert_abs_diff_eq!(df[1], 1.0, epsilon = 1e-2); // SV on the +1 margin
    assert_abs_diff_eq!(df[2], 0.0, epsilon = 1e-2); // decision boundary
}
