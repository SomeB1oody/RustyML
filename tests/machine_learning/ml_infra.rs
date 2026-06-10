//! Integration tests for cross-cutting ML infrastructure.
//!
//! Covers:
//! (1) Error smart constructors — verify each constructor builds the right variant with the
//!     right fields.
//! (2) KernelType::compute — closed-form expected values for every variant.
//! (3) DistanceCalculationMetric::distance — closed-form expected values.
//! (4) Fit / Predict traits used generically — small generic helper exercised with a concrete
//!     model.
//! (5) save_to_path + load_from_path round-trip for LinearRegression and KMeans.
//! (6) predict() before fit() → NotFitted for LinearRegression (supervised), KMeans
//!     (unsupervised), and IsolationForest (anomaly).
//!
//! This file is a SUBMODULE of tests/machine_learning/main.rs.
//! No #![cfg(...)], no fn main.
//!
//! All expected values are derived from mathematical definitions or closed-form arithmetic;
//! derivations are shown in per-test comments.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::isolation_forest::IsolationForest;
use rustyml::machine_learning::kmeans::KMeans;
use rustyml::machine_learning::linear_regression::LinearRegression;
use rustyml::machine_learning::traits::{Fit, Predict};
use rustyml::types::{DistanceCalculationMetric, KernelType};

use crate::common::assert_allclose;
use rustyml::machine_learning::DistanceCalculationMetric as Metric;
use rustyml::machine_learning::dbscan::DBSCAN;
use rustyml::machine_learning::decision_tree::{Algorithm, DecisionTree};
use rustyml::machine_learning::knn::{KNN, WeightingStrategy};
use rustyml::machine_learning::lda::LDA;
use rustyml::machine_learning::linear_svc::{LinearSVC, RegularizationType};
use rustyml::machine_learning::mean_shift::MeanShift;
use rustyml::machine_learning::svc::SVC;

// ═══════════════════════════════════════════════════════════════════════════════
// (1) Error smart constructors
// ═══════════════════════════════════════════════════════════════════════════════

/// Error::dimension_mismatch(3, 2) → DimensionMismatch { expected: 3, found: 2 }
///
/// Derivation: the smart constructor stores its two arguments directly into the
/// `expected` and `found` fields of the DimensionMismatch variant.
#[test]
fn error_dimension_mismatch_constructor_fields() {
    let err = Error::dimension_mismatch(3, 2);
    match err {
        Error::DimensionMismatch { expected, found } => {
            assert_eq!(expected, 3, "expected field should be 3");
            assert_eq!(found, 2, "found field should be 2");
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

/// Error::dimension_mismatch(0, 5) — zero expected is still a valid variant
#[test]
fn error_dimension_mismatch_zero_expected() {
    let err = Error::dimension_mismatch(0, 5);
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 0,
                found: 5
            }
        ),
        "expected DimensionMismatch{{0,5}}, got {err:?}"
    );
}

/// Error::invalid_parameter("lr", "must be positive") → InvalidParameter with matching fields
///
/// Derivation: the smart constructor stores name and reason as Strings.
#[test]
fn error_invalid_parameter_constructor_fields() {
    let err = Error::invalid_parameter("lr", "must be positive");
    match err {
        Error::InvalidParameter {
            ref name,
            ref reason,
        } => {
            assert_eq!(name, "lr");
            assert!(
                reason.contains("must be positive"),
                "reason should contain 'must be positive', got: {reason}"
            );
        }
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

/// Error::non_finite("weights") → NonFinite carrying the supplied context string
///
/// Derivation: Error::non_finite wraps its argument in a NonFinite(String).
#[test]
fn error_non_finite_constructor_carries_context() {
    let err = Error::non_finite("weights");
    match err {
        Error::NonFinite(ref ctx) => {
            assert!(
                ctx.contains("weights"),
                "context should mention 'weights', got: {ctx}"
            );
        }
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// Error::empty_input("dataset") → EmptyInput("dataset")
///
/// Derivation: smart constructor wraps its argument into EmptyInput(String).
#[test]
fn error_empty_input_constructor_carries_what() {
    let err = Error::empty_input("dataset");
    match err {
        Error::EmptyInput(ref what) => {
            assert!(
                what.contains("dataset"),
                "payload should mention 'dataset', got: {what}"
            );
        }
        other => panic!("expected EmptyInput, got {other:?}"),
    }
}

/// Error::not_fitted("KMeans") → NotFitted("KMeans")
///
/// Derivation: Error::not_fitted stores the &'static str directly into the variant.
#[test]
fn error_not_fitted_constructor_carries_model_name() {
    let err = Error::not_fitted("KMeans");
    match err {
        Error::NotFitted(name) => {
            assert_eq!(name, "KMeans", "model name should be 'KMeans'");
        }
        other => panic!("expected NotFitted, got {other:?}"),
    }
}

/// RustymlResult<T> is Result<T, Error> — the alias is usable exactly like Result
///
/// Derivation: type alias identity — assigning an Ok(42i32) to a RustymlResult<i32>
/// must compile and unwrap to 42.
#[test]
fn rustyml_result_alias_is_result_t_error() {
    use rustyml::error::RustymlResult;
    let r: RustymlResult<i32> = Ok(42);
    assert!(matches!(r, Ok(42)));

    let e: RustymlResult<i32> = Err(Error::empty_input("test"));
    assert!(matches!(e, Err(Error::EmptyInput(_))));
}

/// Error::computation("overflow") → Computation { context: "overflow", source: None }
///
/// Derivation: Error::computation wraps its argument and sets source to None.
#[test]
fn error_computation_constructor_no_source() {
    let err = Error::computation("overflow");
    match err {
        Error::Computation {
            ref context,
            ref source,
        } => {
            assert!(
                context.contains("overflow"),
                "context should contain 'overflow'"
            );
            assert!(
                source.is_none(),
                "source should be None for Error::computation"
            );
        }
        other => panic!("expected Computation, got {other:?}"),
    }
}

/// Error::shape_mismatch([2,3], [2,4]) → ShapeMismatch with correct expected/found shapes
///
/// Derivation: shape_mismatch converts both arguments to Vec<usize> and stores them.
#[test]
fn error_shape_mismatch_constructor_fields() {
    let err = Error::shape_mismatch(vec![2usize, 3], vec![2usize, 4]);
    match err {
        Error::ShapeMismatch {
            ref expected,
            ref found,
        } => {
            assert_eq!(expected, &vec![2usize, 3]);
            assert_eq!(found, &vec![2usize, 4]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

/// Error::invalid_input("unexpected rank") → InvalidInput carrying the message
///
/// Derivation: Error::invalid_input is a fallback variant that wraps a String.
#[test]
fn error_invalid_input_constructor() {
    let err = Error::invalid_input("unexpected rank");
    match err {
        Error::InvalidInput(ref msg) => {
            assert!(
                msg.contains("unexpected rank"),
                "message should mention the supplied text"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// (2) KernelType::compute — closed-form values for each variant
// ═══════════════════════════════════════════════════════════════════════════════

// Linear kernel: K(x, y) = x · y  (dot product)
//
// [1,0] · [0,1] = 1*0 + 0*1 = 0
#[test]
fn kernel_linear_orthogonal_is_zero() {
    let k = KernelType::Linear;
    let x1 = array![1.0_f64, 0.0];
    let x2 = array![0.0_f64, 1.0];
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-12);
}

// [1,2] · [3,4] = 1*3 + 2*4 = 3 + 8 = 11
#[test]
fn kernel_linear_general_is_eleven() {
    let k = KernelType::Linear;
    let x1 = array![1.0_f64, 2.0];
    let x2 = array![3.0_f64, 4.0];
    // K = 1*3 + 2*4 = 11
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 11.0, epsilon = 1e-12);
}

// RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
//
// Same vector: diff = [0,0], ||diff||^2 = 0, exp(-1*0) = exp(0) = 1
#[test]
fn kernel_rbf_same_vector_is_one() {
    let k = KernelType::RBF { gamma: 1.0 };
    let x = array![1.0_f64, 0.0];
    // K([1,0],[1,0]) = exp(-1 * 0) = 1.0
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-12);
}

// Orthonormal standard-basis vectors [1,0] and [0,1]:
// diff = [1,-1], ||diff||^2 = 1^2 + (-1)^2 = 2, exp(-1*2) = exp(-2) ≈ 0.13533528
#[test]
fn kernel_rbf_orthonormal_pair_is_exp_minus_two() {
    let k = KernelType::RBF { gamma: 1.0 };
    let x1 = array![1.0_f64, 0.0];
    let x2 = array![0.0_f64, 1.0];
    // diff = [1,-1], ||diff||^2 = 2, K = exp(-2)
    let expected = (-2.0_f64).exp();
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), expected, epsilon = 1e-12);
}

// Polynomial kernel: K(x, y) = (gamma * x·y + coef0)^degree
//
// degree=2, gamma=1, coef0=0; [1,0]·[0,1] = 0 → (1*0 + 0)^2 = 0
#[test]
fn kernel_poly_degree2_orthogonal_is_zero() {
    let k = KernelType::Poly {
        degree: 2,
        gamma: 1.0,
        coef0: 0.0,
    };
    let x1 = array![1.0_f64, 0.0];
    let x2 = array![0.0_f64, 1.0];
    // (1*0 + 0)^2 = 0
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-12);
}

// degree=2, gamma=1, coef0=1; [1,1]·[1,1] = 2 → (1*2 + 1)^2 = 3^2 = 9
#[test]
fn kernel_poly_degree2_with_coef0_one() {
    let k = KernelType::Poly {
        degree: 2,
        gamma: 1.0,
        coef0: 1.0,
    };
    let x = array![1.0_f64, 1.0];
    // x·x = 2, (1*2 + 1)^2 = 9
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), 9.0, epsilon = 1e-12);
}

// Sigmoid kernel: K(x, y) = tanh(gamma * x·y + coef0)
//
// gamma=1, coef0=0; [1,0]·[1,0] = 1 → tanh(1*1 + 0) = tanh(1) ≈ 0.7615941559...
#[test]
fn kernel_sigmoid_unit_vector_is_tanh_one() {
    let k = KernelType::Sigmoid {
        gamma: 1.0,
        coef0: 0.0,
    };
    let x = array![1.0_f64, 0.0];
    // K = tanh(1*1 + 0) = tanh(1)
    let expected = 1.0_f64.tanh();
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), expected, epsilon = 1e-12);
}

// Cosine kernel: K(x, y) = (x · y) / (||x|| * ||y||)
//
// Zero vector: norm_product = sqrt(0 * ||y||^2) = 0; guard returns 0.0
#[test]
fn kernel_cosine_zero_vector_is_zero() {
    let k = KernelType::Cosine;
    let zero = array![0.0_f64, 0.0];
    let other = array![1.0_f64, 2.0];
    // norm_product = 0 → epsilon guard fires → 0.0
    assert_abs_diff_eq!(k.compute(zero.view(), other.view()), 0.0, epsilon = 1e-12);
}

// Identical non-zero vector: K(x, x) = (x·x) / (||x|| * ||x||) = ||x||^2 / ||x||^2 = 1
#[test]
fn kernel_cosine_identical_vector_is_one() {
    let k = KernelType::Cosine;
    let x = array![3.0_f64, 4.0]; // any non-zero vector
    // K = (3^2+4^2) / (5 * 5) = 25/25 = 1
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// (3) DistanceCalculationMetric::distance — closed-form values
// ═══════════════════════════════════════════════════════════════════════════════

// Euclidean: sqrt((3-0)^2 + (4-0)^2) = sqrt(9 + 16) = sqrt(25) = 5
#[test]
fn distance_euclidean_3_4_triangle_is_5() {
    let metric = DistanceCalculationMetric::Euclidean;
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    // sqrt(3^2 + 4^2) = sqrt(9+16) = 5
    assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 5.0, epsilon = 1e-12);
}

// Euclidean is symmetric: distance(a,b) == distance(b,a)
#[test]
fn distance_euclidean_is_symmetric() {
    let metric = DistanceCalculationMetric::Euclidean;
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(
        metric.distance(a.view(), b.view()),
        metric.distance(b.view(), a.view()),
        epsilon = 1e-12
    );
}

// Manhattan: |3-0| + |4-0| = 3 + 4 = 7
#[test]
fn distance_manhattan_3_4_is_7() {
    let metric = DistanceCalculationMetric::Manhattan;
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    // |3| + |4| = 7
    assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 7.0, epsilon = 1e-12);
}

// Minkowski(p=3): (|3|^3 + |4|^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)
// 91^(1/3) ≈ 4.497941445275415
#[test]
fn distance_minkowski_p3_is_91_cbrt() {
    let metric = DistanceCalculationMetric::Minkowski(3.0);
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    // (|3|^3 + |4|^3)^(1/3) = (27+64)^(1/3) = 91^(1/3)
    let expected = 91.0_f64.powf(1.0 / 3.0);
    assert_abs_diff_eq!(
        metric.distance(a.view(), b.view()),
        expected,
        epsilon = 1e-9
    );
}

// Euclidean self-distance: distance(x, x) = 0 for any x
#[test]
fn distance_euclidean_self_is_zero() {
    let metric = DistanceCalculationMetric::Euclidean;
    let a = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(metric.distance(a.view(), a.view()), 0.0, epsilon = 1e-12);
}

// Manhattan self-distance: distance(x, x) = 0
#[test]
fn distance_manhattan_self_is_zero() {
    let metric = DistanceCalculationMetric::Manhattan;
    let a = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(metric.distance(a.view(), a.view()), 0.0, epsilon = 1e-12);
}

// Minkowski(p=2) reduces to Euclidean:
// [0,0]->[3,4]: (|3|^2 + |4|^2)^(1/2) = (9+16)^(1/2) = 25^(1/2) = 5
#[test]
fn distance_minkowski_p2_equals_euclidean() {
    let euclidean = DistanceCalculationMetric::Euclidean;
    let mink2 = DistanceCalculationMetric::Minkowski(2.0);
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    // Both must give 5.0
    assert_abs_diff_eq!(mink2.distance(a.view(), b.view()), 5.0, epsilon = 1e-9);
    assert_abs_diff_eq!(
        mink2.distance(a.view(), b.view()),
        euclidean.distance(a.view(), b.view()),
        epsilon = 1e-9
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// (4) Fit / Predict traits used generically
// ═══════════════════════════════════════════════════════════════════════════════

/// A generic helper that trains any supervised estimator through the Fit trait and then
/// predicts through the Predict trait, returning the number of predictions.
///
/// This exercises that the trait implementations dispatch correctly to the concrete model.
fn train_and_predict_count<M>(
    model: &mut M,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    x_test: &Array2<f64>,
) -> usize
where
    M: for<'a> Fit<(&'a Array2<f64>, &'a Array1<f64>)>
        + for<'a> Predict<&'a Array2<f64>, Output = Array1<f64>>,
{
    Fit::fit(model, (x_train, y_train)).expect("fit through trait should succeed");
    let preds = Predict::predict(model, x_test).expect("predict through trait should succeed");
    preds.len()
}

/// Exercises the generic helper with LinearRegression, confirming that the Fit and Predict
/// trait implementations forward correctly to the inherent methods.
///
/// Data: y = 2x + 1 on x = [1,2,3,4,5].
/// Test inputs: 3 new points → expect 3 predictions returned.
#[test]
fn generic_fit_predict_with_linear_regression() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x_test = Array2::from_shape_vec((3, 1), vec![6.0, 7.0, 8.0]).unwrap();

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let n = train_and_predict_count(&mut model, &x_train, &y_train, &x_test);
    assert_eq!(n, 3, "expected 3 predictions for 3 test points");
}

/// Exercises the Fit trait on KMeans (unsupervised variant: `Fit<&Array2<f64>>`).
/// After fitting, predict should return labels of the same length as the input.
///
/// This tests the trait dispatch path independently from the inherent method path
/// tested in kmeans.rs.
#[test]
fn generic_fit_trait_with_kmeans_unsupervised() {
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // blob A near origin
            10.0, 0.0, 10.1, 0.0, 10.0, 0.1, // blob B near (10,0)
        ],
    )
    .unwrap();

    let mut km = KMeans::new(2, 200, 1e-4, Some(42)).unwrap();
    // Invoke Fit trait explicitly — Fit<&Array2<f64>>
    Fit::fit(&mut km, &data).expect("fit via Fit trait should succeed");

    // Invoke Predict trait explicitly
    let labels = Predict::predict(&km, &data).expect("predict via Predict trait should succeed");
    assert_eq!(labels.len(), 6, "should produce one label per sample");
}

/// Calling Fit::fit and then Predict::predict through the trait produces the same predictions
/// as calling the inherent methods directly. This confirms the traits do not recurse.
///
/// y = 2x + 1 on x = [1,2,3,4,5]; predict at x = 6 → 2*6+1 = 13.
#[test]
fn trait_predictions_match_inherent_method_predictions() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x_test = Array2::from_shape_vec((1, 1), vec![6.0]).unwrap();

    // Via trait
    let mut model_trait = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    Fit::fit(&mut model_trait, (&x_train, &y_train)).unwrap();
    let preds_trait = Predict::predict(&model_trait, &x_test).unwrap();

    // Via inherent method
    let mut model_direct = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model_direct.fit(&x_train, &y_train).unwrap();
    let preds_direct = model_direct.predict(&x_test).unwrap();

    // Both must agree exactly (same deterministic gradient descent, same starting state)
    assert_abs_diff_eq!(preds_trait[0], preds_direct[0], epsilon = 0.0);

    // Also verify closed-form value: y = 2*6+1 = 13
    assert_abs_diff_eq!(preds_trait[0], 13.0, epsilon = 5e-3);
}

// ═══════════════════════════════════════════════════════════════════════════════
// (5) save_to_path + load_from_path round-trip
// ═══════════════════════════════════════════════════════════════════════════════

/// LinearRegression: save and reload produces a model that returns identical predictions.
///
/// Data: y = 2x + 1 on x = [1..5].
/// Test points: x = 6 → 13, x = 0 → 1 (closed-form OLS).
/// Round-trip assertion: predictions must be bit-identical before and after serialisation.
#[test]
fn linear_regression_save_load_round_trip_predictions_identical() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x_test = Array2::from_shape_vec((2, 1), vec![6.0, 0.0]).unwrap();

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model.fit(&x_train, &y_train).unwrap();
    let preds_before = model.predict(&x_test).unwrap();

    // Closed-form sanity: y(6) = 2*6+1 = 13, y(0) = 2*0+1 = 1
    assert_abs_diff_eq!(preds_before[0], 13.0, epsilon = 5e-3);
    assert_abs_diff_eq!(preds_before[1], 1.0, epsilon = 5e-3);

    let path = "/tmp/rustyml_ml_infra_linreg_round_trip.json";
    model
        .save_to_path(path)
        .expect("save_to_path should succeed");

    let loaded = LinearRegression::load_from_path(path).expect("load_from_path should succeed");
    let preds_after = loaded.predict(&x_test).unwrap();

    // Predictions must be bit-identical — no lossy serialisation
    assert_allclose(&preds_before, &preds_after, 0.0);

    let _ = std::fs::remove_file(path);
}

/// LinearRegression: loaded model has same fit_intercept, learning_rate, and coefficients
/// as the original.
#[test]
fn linear_regression_save_load_preserves_hyperparameters() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

    let mut model = LinearRegression::new(false, 0.005, 8_000, 1e-9, None).unwrap();
    model.fit(&x_train, &y_train).unwrap();

    let path = "/tmp/rustyml_ml_infra_linreg_hyperparams.json";
    model.save_to_path(path).unwrap();
    let loaded = LinearRegression::load_from_path(path).unwrap();

    assert_eq!(loaded.get_fit_intercept(), model.get_fit_intercept());
    assert_abs_diff_eq!(
        loaded.get_learning_rate(),
        model.get_learning_rate(),
        epsilon = 1e-15
    );
    assert_eq!(loaded.get_max_iterations(), model.get_max_iterations());

    // Coefficients must be exactly preserved (f64 round-trip through JSON)
    let orig = model.get_coefficients().unwrap();
    let load = loaded.get_coefficients().unwrap();
    assert_allclose(orig, load, 0.0);

    let _ = std::fs::remove_file(path);
}

/// Build three tight, well-separated blobs for KMeans save/load round-trip.
/// Centres at (0,0), (100,0), (50,100) so the test doesn't accidentally depend on
/// seeds in a fragile way.
fn three_blob_data_for_round_trip() -> Array2<f64> {
    Array2::from_shape_vec(
        (15, 2),
        vec![
            // blob A around (0,0)
            -0.05, 0.03, 0.04, -0.02, 0.01, 0.05, -0.03, -0.04, 0.02, 0.01,
            // blob B around (100,0)
            99.95, 0.03, 100.04, -0.02, 100.01, 0.05, 99.97, -0.04, 100.02, 0.01,
            // blob C around (50,100)
            49.95, 100.03, 50.04, 99.98, 50.01, 100.05, 49.97, 99.96, 50.02, 100.01,
        ],
    )
    .unwrap()
}

/// KMeans: save and reload produces a model that returns identical predictions.
///
/// Derivation: predict is a pure function of the stored centroids; serialisation must
/// preserve centroids exactly, so predictions must be bit-identical.
#[test]
fn kmeans_save_load_round_trip_predictions_identical() {
    let data = three_blob_data_for_round_trip();

    let mut km = KMeans::new(3, 300, 1e-4, Some(42)).unwrap();
    km.fit(&data).unwrap();
    let preds_before = km.predict(&data).unwrap();

    let path = "/tmp/rustyml_ml_infra_kmeans_round_trip.json";
    km.save_to_path(path).expect("save_to_path should succeed");

    let km_loaded = KMeans::load_from_path(path).expect("load_from_path should succeed");
    let preds_after = km_loaded.predict(&data).unwrap();

    assert_eq!(preds_before.len(), preds_after.len());
    for (o, l) in preds_before.iter().zip(preds_after.iter()) {
        assert_eq!(o, l, "prediction mismatch after save/load round-trip");
    }

    // Centroids must be preserved exactly
    let c_orig = km.get_centroids().unwrap();
    let c_load = km_loaded.get_centroids().unwrap();
    assert_allclose(c_orig, c_load, 0.0_f64);

    let _ = std::fs::remove_file(path);
}

/// KMeans: loaded model hyperparameters match the original.
#[test]
fn kmeans_save_load_preserves_hyperparameters() {
    let data = three_blob_data_for_round_trip();

    let mut km = KMeans::new(3, 200, 1e-5, Some(7)).unwrap();
    km.fit(&data).unwrap();

    let path = "/tmp/rustyml_ml_infra_kmeans_hyperparams.json";
    km.save_to_path(path).unwrap();
    let loaded = KMeans::load_from_path(path).unwrap();

    assert_eq!(loaded.get_n_clusters(), km.get_n_clusters());
    assert_eq!(loaded.get_max_iterations(), km.get_max_iterations());
    assert_abs_diff_eq!(loaded.get_tolerance(), km.get_tolerance(), epsilon = 1e-15);
    assert_eq!(loaded.get_random_state(), km.get_random_state());

    let _ = std::fs::remove_file(path);
}

// ═══════════════════════════════════════════════════════════════════════════════
// (6) predict() before fit() → NotFitted
// ═══════════════════════════════════════════════════════════════════════════════

/// LinearRegression (supervised): predict before fit → NotFitted
///
/// Derivation: the predict method calls check_is_fitted(self.coefficients.is_some(), "LinearRegression")
/// which returns Err(NotFitted) when coefficients is None.
#[test]
fn linear_regression_predict_before_fit_is_not_fitted() {
    let model = LinearRegression::new(true, 0.01, 100, 1e-6, None).unwrap();
    let x = array![[1.0, 2.0]];
    let result = model.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

/// KMeans (unsupervised): predict before fit → NotFitted
///
/// Derivation: predict checks self.centroids.is_none() and returns Error::not_fitted("KMeans").
#[test]
fn kmeans_predict_before_fit_is_not_fitted() {
    let km = KMeans::new(3, 100, 1e-4, Some(42)).unwrap();
    let x = array![[1.0, 2.0]];
    let result = km.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

/// IsolationForest (anomaly detection): predict before fit → NotFitted
///
/// Derivation: predict calls check_is_fitted(self.trees.is_some(), "IsolationForest")
/// which returns Err(NotFitted) when trees is None.
#[test]
fn isolation_forest_predict_before_fit_is_not_fitted() {
    let forest = IsolationForest::new(10, 32, None, Some(42)).unwrap();
    let x = array![[1.0, 2.0]];
    let result = forest.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

/// LinearRegression default: predict before fit also returns NotFitted (default constructor path)
#[test]
fn linear_regression_default_predict_before_fit_is_not_fitted() {
    let model = LinearRegression::default();
    let x = array![[1.0]];
    let result = model.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

/// KMeans default: predict before fit also returns NotFitted
#[test]
fn kmeans_default_predict_before_fit_is_not_fitted() {
    let km = KMeans::default();
    let x = array![[1.0, 2.0]];
    let result = km.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

/// IsolationForest default: predict before fit also returns NotFitted
#[test]
fn isolation_forest_default_predict_before_fit_is_not_fitted() {
    let forest = IsolationForest::default();
    let x = array![[1.0, 2.0]];
    let result = forest.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// (4b) Fit / Predict trait forwarding for the remaining estimators
// ═══════════════════════════════════════════════════════════════════════════════
//
// Each test invokes Fit::fit and Predict::predict EXPLICITLY (via the trait, not the
// inherent method) on a distinct estimator, confirming the trait impls compile and
// dispatch for every distinct Predict::Output type. The data is small and well-
// separated so the only thing under test is trait forwarding, not learning quality;
// assertions check the output length / element domain, which are known by construction.
//
// Note: KernelType is already imported at the top of this file (rustyml::types::KernelType),
// which is the same type SVC::new takes; we do NOT re-import it from the svc module.

/// IsolationForest (unsupervised, Fit<&Array2<f64>>, Predict::Output = Array1<f64>).
#[test]
fn generic_fit_predict_isolation_forest_outputs_f64_scores() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 50.0, 50.0],
    )
    .unwrap();

    let mut forest = IsolationForest::new(20, 32, None, Some(42)).unwrap();
    Fit::fit(&mut forest, &data).expect("fit via Fit trait should succeed");
    let scores: Array1<f64> =
        Predict::predict(&forest, &data).expect("predict via Predict trait should succeed");

    assert_eq!(scores.len(), 5, "one anomaly score per sample");
    // Anomaly scores are 2^(-E/c) with E,c > 0, hence strictly in (0, 1].
    for (i, &s) in scores.iter().enumerate() {
        assert!((0.0..=1.0).contains(&s), "score[{i}] = {s} not in [0,1]");
    }
}

/// DBSCAN (unsupervised, Fit<&Array2<f64>>, Predict::Output = Array1<isize>).
/// Two tight, well-separated blobs → each blob's points must share one non-negative
/// cluster id (the -1 sentinel is reserved for noise, which these dense blobs are not).
#[test]
fn generic_fit_predict_dbscan_outputs_isize_labels() {
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // blob A near origin
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, // blob B near (10,10)
        ],
    )
    .unwrap();

    let mut db = DBSCAN::new(0.5, 2, Metric::Euclidean).unwrap();
    Fit::fit(&mut db, &data).expect("fit via Fit trait should succeed");
    // predict returns labels for the points passed in (DBSCAN assigns each to a fitted cluster).
    let labels: Array1<isize> =
        Predict::predict(&db, &data).expect("predict via Predict trait should succeed");

    assert_eq!(labels.len(), 6, "one label per sample");
    // Both dense blobs are clusters, so no point is noise; ids are non-negative.
    for (i, &l) in labels.iter().enumerate() {
        assert!(
            l >= 0,
            "label[{i}] = {l} should be a non-negative cluster id"
        );
    }
    // The two blobs are far apart → they must land in different clusters.
    assert_ne!(labels[0], labels[3], "the two separated blobs must differ");
}

/// KNN<i32> (supervised, Fit<(&Array2<f64>, &Array1<i32>)>, Predict::Output = Array1<i32>).
/// k=1 nearest neighbour on two labelled anchor points: a query next to anchor 0 must be
/// labelled 0 and a query next to anchor 1 must be labelled 1 — known by construction.
#[test]
fn generic_fit_predict_knn_outputs_generic_labels() {
    let x_train = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 0.0]).unwrap();
    let y_train = Array1::from_vec(vec![0_i32, 1]);

    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    Fit::fit(&mut knn, (&x_train, &y_train)).expect("fit via Fit trait should succeed");

    let x_test = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 9.5, 0.0]).unwrap();
    let preds: Array1<i32> =
        Predict::predict(&knn, &x_test).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 2, "one label per test point");
    // (0.5,0) is nearest to anchor 0 (label 0); (9.5,0) is nearest to anchor 1 (label 1).
    assert_eq!(preds[0], 0, "point near anchor 0 must get label 0");
    assert_eq!(preds[1], 1, "point near anchor 1 must get label 1");
}

/// LDA (supervised, Fit<(&Array2<f64>, &Array1<i32>)>, Predict::Output = Array1<i32>).
/// Two well-separated 1-D classes; LDA must label each training point with its own class.
/// 6 samples, 2 classes (3 each), n_components=1 == (n_classes-1).min(n_features) = 1.
#[test]
fn generic_fit_predict_lda_outputs_i32_labels() {
    let x_train = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]).unwrap();
    let y_train = Array1::from_vec(vec![0_i32, 0, 0, 1, 1, 1]);

    let mut lda = LDA::new(1, None, None).unwrap();
    Fit::fit(&mut lda, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<i32> =
        Predict::predict(&lda, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 6, "one label per sample");
    // Classes are separated by a wide margin (3 vs 7), so training accuracy must be perfect.
    for (i, (&p, &t)) in preds.iter().zip(y_train.iter()).enumerate() {
        assert_eq!(
            p, t,
            "sample {i}: LDA via trait predicted {p}, expected {t}"
        );
    }
}

/// DecisionTree (supervised, Fit<(&Array2<f64>, &Array1<f64>)>, Predict::Output = Array1<f64>).
/// Linearly separable binary data; an unbounded CART tree memorizes every label exactly.
#[test]
fn generic_fit_predict_decision_tree_outputs_f64_labels() {
    let x_train = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2]).unwrap();
    let y_train = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    Fit::fit(&mut tree, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<f64> =
        Predict::predict(&tree, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 6, "one label per sample");
    // Separable by a single threshold on feature 0 → zero training error by construction.
    for (i, (&p, &t)) in preds.iter().zip(y_train.iter()).enumerate() {
        assert_abs_diff_eq!(p, t, epsilon = 1e-9);
        let _ = i;
    }
}

/// LinearSVC (supervised, Fit<(&Array2<f64>, &Array1<f64>)>, Predict::Output = Array1<f64>).
/// Confirms trait forwarding and the {0,1} label domain on separable data.
#[test]
fn generic_fit_predict_linear_svc_outputs_f64_labels() {
    let x_train = Array2::from_shape_vec(
        (8, 2),
        vec![
            -5.0, 0.0, -6.0, 0.0, -7.0, 0.0, -4.0, 0.0, // class 0
            5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 4.0, 0.0, // class 1
        ],
    )
    .unwrap();
    let y_train = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

    let mut svc =
        LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6, None).unwrap();
    Fit::fit(&mut svc, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<f64> =
        Predict::predict(&svc, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 8, "one label per sample");
    // Widely separated (x<0 vs x>0) → perfect training classification into {0,1}.
    let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (i, (&p, &t)) in preds.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            p, t,
            "sample {i}: LinearSVC via trait predicted {p}, expected {t}"
        );
    }
}

/// MeanShift (unsupervised, Fit<&Array2<f64>>, Predict::Output = Array1<usize>).
/// Two tight blobs at (0,0) and (20,20); MeanShift must place each blob's samples in the
/// same cluster and the two blobs in different clusters.
#[test]
fn generic_fit_predict_mean_shift_outputs_usize_labels() {
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            -0.1, 0.0, 0.1, 0.0, 0.0, 0.0, // blob A near (0,0)
            19.9, 20.0, 20.1, 20.0, 20.0, 20.0, // blob B near (20,20)
        ],
    )
    .unwrap();

    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true), None).unwrap();
    Fit::fit(&mut ms, &data).expect("fit via Fit trait should succeed");
    let labels: Array1<usize> =
        Predict::predict(&ms, &data).expect("predict via Predict trait should succeed");

    assert_eq!(labels.len(), 6, "one label per sample");
    // Within-blob agreement and across-blob separation are guaranteed by the geometry.
    assert_eq!(labels[0], labels[1], "blob A samples must share a cluster");
    assert_eq!(labels[3], labels[4], "blob B samples must share a cluster");
    assert_ne!(
        labels[0], labels[3],
        "the two far-apart blobs must be different clusters"
    );
}

/// SVC (supervised, Fit<(&Array2<f64>, &Array1<f64>)>, Predict::Output = Array1<f64>).
/// Labels are in the SVC ±1 domain; a linear SVC on separable data classifies all correctly.
#[test]
fn generic_fit_predict_svc_outputs_pm1_labels() {
    let x_train = Array2::from_shape_vec(
        (8, 2),
        vec![
            2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, // class +1
            -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -3.0, -3.0, // class -1
        ],
    )
    .unwrap();
    let y_train = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);

    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000, Some(42)).unwrap();
    Fit::fit(&mut svc, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<f64> =
        Predict::predict(&svc, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 8, "one label per sample");
    for (i, (&p, &t)) in preds.iter().zip(y_train.iter()).enumerate() {
        assert_eq!(
            p, t,
            "sample {i}: SVC via trait predicted {p}, expected {t}"
        );
    }
}
