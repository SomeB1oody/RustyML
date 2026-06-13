//! Integration tests for cross-cutting ML infrastructure: Error smart constructors,
//! KernelType::compute, DistanceCalculationMetric::distance, generic Fit/Predict trait
//! forwarding, save/load round-trips, and predict-before-fit NotFitted errors

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::IsolationForest;
use rustyml::machine_learning::KMeans;
use rustyml::machine_learning::LinearRegression;
use rustyml::machine_learning::traits::{Fit, Predict};
use rustyml::types::{DistanceCalculationMetric, KernelType};

use crate::common::assert_allclose;
use rustyml::machine_learning::DBSCAN;
use rustyml::machine_learning::DistanceCalculationMetric as Metric;
use rustyml::machine_learning::LDA;
use rustyml::machine_learning::MeanShift;
use rustyml::machine_learning::SVC;
use rustyml::machine_learning::{Algorithm, DecisionTree};
use rustyml::machine_learning::{KNN, WeightingStrategy};
use rustyml::machine_learning::{LinearSVC, RegularizationType};

// (1) Error smart constructors

/// dimension_mismatch builds DimensionMismatch with the expected and found fields set
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

/// dimension_mismatch with zero expected still builds a valid DimensionMismatch
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

/// invalid_parameter builds InvalidParameter with name and reason set
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

/// non_finite builds NonFinite carrying the supplied context string
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

/// empty_input builds EmptyInput carrying the supplied string
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

/// not_fitted builds NotFitted carrying the supplied model name
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

/// RustymlResult<T> aliases Result<T, Error> and is usable exactly like Result
#[test]
fn rustyml_result_alias_is_result_t_error() {
    use rustyml::error::RustymlResult;
    let r: RustymlResult<i32> = Ok(42);
    assert!(matches!(r, Ok(42)));

    let e: RustymlResult<i32> = Err(Error::empty_input("test"));
    assert!(matches!(e, Err(Error::EmptyInput(_))));
}

/// computation builds Computation with the context set and source None
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

/// shape_mismatch builds ShapeMismatch with the expected and found shapes set
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

/// invalid_input builds InvalidInput carrying the supplied message
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

// (2) KernelType::compute - closed-form values for each variant

// Linear kernel K(x, y) = x . y; orthogonal [1,0] and [0,1] give 0
#[test]
fn kernel_linear_orthogonal_is_zero() {
    let k = KernelType::Linear;
    let x1 = array![1.0_f64, 0.0];
    let x2 = array![0.0_f64, 1.0];
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-12);
}

// Linear kernel [1,2] . [3,4] = 11
#[test]
fn kernel_linear_general_is_eleven() {
    let k = KernelType::Linear;
    let x1 = array![1.0_f64, 2.0];
    let x2 = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 11.0, epsilon = 1e-12);
}

// RBF kernel K(x, y) = exp(-gamma * ||x - y||^2); same vector gives exp(0) = 1
#[test]
fn kernel_rbf_same_vector_is_one() {
    let k = KernelType::RBF { gamma: 1.0 };
    let x = array![1.0_f64, 0.0];
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-12);
}

// RBF kernel on orthonormal [1,0] and [0,1]: ||diff||^2 = 2, so K = exp(-2)
#[test]
fn kernel_rbf_orthonormal_pair_is_exp_minus_two() {
    let k = KernelType::RBF { gamma: 1.0 };
    let x1 = array![1.0_f64, 0.0];
    let x2 = array![0.0_f64, 1.0];
    let expected = (-2.0_f64).exp();
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), expected, epsilon = 1e-12);
}

// Polynomial kernel K(x, y) = (gamma * x.y + coef0)^degree
// degree=2, gamma=1, coef0=0 with orthogonal inputs gives (0)^2 = 0
#[test]
fn kernel_poly_degree2_orthogonal_is_zero() {
    let k = KernelType::Poly {
        degree: 2,
        gamma: 1.0,
        coef0: 0.0,
    };
    let x1 = array![1.0_f64, 0.0];
    let x2 = array![0.0_f64, 1.0];
    assert_abs_diff_eq!(k.compute(x1.view(), x2.view()), 0.0, epsilon = 1e-12);
}

// Polynomial kernel degree=2, gamma=1, coef0=1 on [1,1]: (1*2 + 1)^2 = 9
#[test]
fn kernel_poly_degree2_with_coef0_one() {
    let k = KernelType::Poly {
        degree: 2,
        gamma: 1.0,
        coef0: 1.0,
    };
    let x = array![1.0_f64, 1.0];
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), 9.0, epsilon = 1e-12);
}

// Sigmoid kernel K(x, y) = tanh(gamma * x.y + coef0); gamma=1, coef0=0 on [1,0] gives tanh(1)
#[test]
fn kernel_sigmoid_unit_vector_is_tanh_one() {
    let k = KernelType::Sigmoid {
        gamma: 1.0,
        coef0: 0.0,
    };
    let x = array![1.0_f64, 0.0];
    let expected = 1.0_f64.tanh();
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), expected, epsilon = 1e-12);
}

// Cosine kernel K(x, y) = (x . y) / (||x|| * ||y||); a zero vector trips the guard and returns 0
#[test]
fn kernel_cosine_zero_vector_is_zero() {
    let k = KernelType::Cosine;
    let zero = array![0.0_f64, 0.0];
    let other = array![1.0_f64, 2.0];
    assert_abs_diff_eq!(k.compute(zero.view(), other.view()), 0.0, epsilon = 1e-12);
}

// Cosine kernel on an identical non-zero vector: K(x, x) = ||x||^2 / ||x||^2 = 1
#[test]
fn kernel_cosine_identical_vector_is_one() {
    let k = KernelType::Cosine;
    let x = array![3.0_f64, 4.0]; // any non-zero vector
    assert_abs_diff_eq!(k.compute(x.view(), x.view()), 1.0, epsilon = 1e-12);
}

// (3) DistanceCalculationMetric::distance - closed-form values

// Euclidean distance on the 3-4-5 triangle: sqrt(3^2 + 4^2) = 5
#[test]
fn distance_euclidean_3_4_triangle_is_5() {
    let metric = DistanceCalculationMetric::Euclidean;
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 5.0, epsilon = 1e-12);
}

// Euclidean distance is symmetric: distance(a,b) == distance(b,a)
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

// Manhattan distance: |3| + |4| = 7
#[test]
fn distance_manhattan_3_4_is_7() {
    let metric = DistanceCalculationMetric::Manhattan;
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(metric.distance(a.view(), b.view()), 7.0, epsilon = 1e-12);
}

// Minkowski(p=3): (|3|^3 + |4|^3)^(1/3) = 91^(1/3)
#[test]
fn distance_minkowski_p3_is_91_cbrt() {
    let metric = DistanceCalculationMetric::Minkowski(3.0);
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    let expected = 91.0_f64.powf(1.0 / 3.0);
    assert_abs_diff_eq!(
        metric.distance(a.view(), b.view()),
        expected,
        epsilon = 1e-9
    );
}

// Euclidean self-distance: distance(x, x) = 0
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

// Minkowski(p=2) reduces to Euclidean: [0,0] -> [3,4] gives 5
#[test]
fn distance_minkowski_p2_equals_euclidean() {
    let euclidean = DistanceCalculationMetric::Euclidean;
    let mink2 = DistanceCalculationMetric::Minkowski(2.0);
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    assert_abs_diff_eq!(mink2.distance(a.view(), b.view()), 5.0, epsilon = 1e-9);
    assert_abs_diff_eq!(
        mink2.distance(a.view(), b.view()),
        euclidean.distance(a.view(), b.view()),
        epsilon = 1e-9
    );
}

// (4) Fit / Predict traits used generically

/// Train a supervised estimator through the Fit trait, predict through the Predict trait,
/// and return the number of predictions
///
/// # Parameters
///
/// - `model` - estimator to train and query
/// - `x_train` - training features
/// - `y_train` - training targets
/// - `x_test` - inputs to predict
///
/// # Returns
///
/// - Number of predictions produced for `x_test`
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

/// Generic Fit/Predict helper on LinearRegression returns one prediction per test point
#[test]
fn generic_fit_predict_with_linear_regression() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x_test = Array2::from_shape_vec((3, 1), vec![6.0, 7.0, 8.0]).unwrap();

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    let n = train_and_predict_count(&mut model, &x_train, &y_train, &x_test);
    assert_eq!(n, 3, "expected 3 predictions for 3 test points");
}

/// Fit/Predict traits on unsupervised KMeans return one label per input sample
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
    // Invoke the Fit and Predict traits explicitly
    Fit::fit(&mut km, &data).expect("fit via Fit trait should succeed");
    let labels = Predict::predict(&km, &data).expect("predict via Predict trait should succeed");
    assert_eq!(labels.len(), 6, "should produce one label per sample");
}

/// Predictions via the Fit/Predict traits match the inherent methods exactly
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

    // Both paths must agree exactly, and match the closed-form y = 2*6+1 = 13
    assert_abs_diff_eq!(preds_trait[0], preds_direct[0], epsilon = 0.0);
    assert_abs_diff_eq!(preds_trait[0], 13.0, epsilon = 5e-3);
}

// (5) save_to_path + load_from_path round-trip

/// LinearRegression save then reload yields identical predictions
#[test]
fn linear_regression_save_load_round_trip_predictions_identical() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x_test = Array2::from_shape_vec((2, 1), vec![6.0, 0.0]).unwrap();

    let mut model = LinearRegression::new(true, 0.01, 10_000, 1e-10, None).unwrap();
    model.fit(&x_train, &y_train).unwrap();
    let preds_before = model.predict(&x_test).unwrap();

    // Closed-form sanity: y(6) = 13, y(0) = 1
    assert_abs_diff_eq!(preds_before[0], 13.0, epsilon = 5e-3);
    assert_abs_diff_eq!(preds_before[1], 1.0, epsilon = 5e-3);

    let path = "/tmp/rustyml_ml_infra_linreg_round_trip.json";
    model
        .save_to_path(path)
        .expect("save_to_path should succeed");

    let loaded = LinearRegression::load_from_path(path).expect("load_from_path should succeed");
    let preds_after = loaded.predict(&x_test).unwrap();

    // Predictions must be bit-identical - no lossy serialisation
    assert_allclose(&preds_before, &preds_after, 0.0);

    let _ = std::fs::remove_file(path);
}

/// LinearRegression reload preserves fit_intercept, learning_rate, and coefficients
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

    // Coefficients must be exactly preserved through the JSON round-trip
    let orig = model.get_coefficients().unwrap();
    let load = loaded.get_coefficients().unwrap();
    assert_allclose(orig, load, 0.0);

    let _ = std::fs::remove_file(path);
}

/// Build three tight, well-separated blobs centred at (0,0), (100,0), (50,100) for the
/// KMeans save/load round-trip
///
/// # Returns
///
/// - 15x2 matrix of points, five per blob
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

/// KMeans save then reload yields identical predictions and preserves centroids exactly
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

/// KMeans reload preserves all hyperparameters
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

// (6) predict() before fit() -> NotFitted

/// LinearRegression (supervised): predict before fit returns NotFitted
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

/// KMeans (unsupervised): predict before fit returns NotFitted
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

/// IsolationForest (anomaly detection): predict before fit returns NotFitted
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

/// LinearRegression::default: predict before fit returns NotFitted
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

/// KMeans::default: predict before fit returns NotFitted
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

/// IsolationForest::default: predict before fit returns NotFitted
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

// (4b) Fit / Predict trait forwarding for the remaining estimators: each test invokes the
// traits explicitly to confirm dispatch for every distinct Predict::Output type

/// IsolationForest (unsupervised, Fit<&Array2<f64>>, Predict::Output = Array1<f64>)
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
    // Anomaly scores are 2^(-E/c) with E,c > 0, hence strictly in (0, 1]
    for (i, &s) in scores.iter().enumerate() {
        assert!((0.0..=1.0).contains(&s), "score[{i}] = {s} not in [0,1]");
    }
}

/// DBSCAN (unsupervised, Predict::Output = Array1<isize>): two dense blobs get distinct
/// non-negative cluster ids
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
    // predict assigns each passed-in point to a fitted cluster
    let labels: Array1<isize> =
        Predict::predict(&db, &data).expect("predict via Predict trait should succeed");

    assert_eq!(labels.len(), 6, "one label per sample");
    // Both dense blobs are clusters, so no point is noise and ids are non-negative
    for (i, &l) in labels.iter().enumerate() {
        assert!(
            l >= 0,
            "label[{i}] = {l} should be a non-negative cluster id"
        );
    }
    // The two far-apart blobs must land in different clusters
    assert_ne!(labels[0], labels[3], "the two separated blobs must differ");
}

/// KNN<i32> (supervised, Predict::Output = Array1<i32>): k=1 labels each query with its
/// nearest anchor's class
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
    // (0.5,0) is nearest to anchor 0 (label 0); (9.5,0) is nearest to anchor 1 (label 1)
    assert_eq!(preds[0], 0, "point near anchor 0 must get label 0");
    assert_eq!(preds[1], 1, "point near anchor 1 must get label 1");
}

/// LDA (supervised, Predict::Output = Array1<i32>): two well-separated 1-D classes are
/// each labelled correctly on the training set
#[test]
fn generic_fit_predict_lda_outputs_i32_labels() {
    let x_train = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]).unwrap();
    let y_train = Array1::from_vec(vec![0_i32, 0, 0, 1, 1, 1]);

    let mut lda = LDA::new(1, None, None).unwrap();
    Fit::fit(&mut lda, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<i32> =
        Predict::predict(&lda, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 6, "one label per sample");
    // Classes are separated by a wide margin (3 vs 7), so training accuracy must be perfect
    for (i, (&p, &t)) in preds.iter().zip(y_train.iter()).enumerate() {
        assert_eq!(
            p, t,
            "sample {i}: LDA via trait predicted {p}, expected {t}"
        );
    }
}

/// DecisionTree (supervised, Predict::Output = Array1<f64>): an unbounded CART tree
/// memorizes separable binary labels exactly
#[test]
fn generic_fit_predict_decision_tree_outputs_f64_labels() {
    let x_train = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2]).unwrap();
    let y_train = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut tree = DecisionTree::new(Algorithm::CART, true, None).unwrap();
    Fit::fit(&mut tree, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<f64> =
        Predict::predict(&tree, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 6, "one label per sample");
    // Separable by a single threshold on feature 0, so training error is zero
    for (i, (&p, &t)) in preds.iter().zip(y_train.iter()).enumerate() {
        assert_abs_diff_eq!(p, t, epsilon = 1e-9);
        let _ = i;
    }
}

/// LinearSVC (supervised, Predict::Output = Array1<f64>): separable data is classified
/// into the {0,1} label domain
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
    // Widely separated (x<0 vs x>0), so training classification into {0,1} is perfect
    let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (i, (&p, &t)) in preds.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            p, t,
            "sample {i}: LinearSVC via trait predicted {p}, expected {t}"
        );
    }
}

/// MeanShift (unsupervised, Predict::Output = Array1<usize>): two tight blobs each land in
/// one cluster, and the blobs land in different clusters
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

    let mut ms = MeanShift::new(2.0, Some(300), Some(1e-5), Some(true), Some(true)).unwrap();
    Fit::fit(&mut ms, &data).expect("fit via Fit trait should succeed");
    let labels: Array1<usize> =
        Predict::predict(&ms, &data).expect("predict via Predict trait should succeed");

    assert_eq!(labels.len(), 6, "one label per sample");
    // Within-blob agreement and across-blob separation follow from the geometry
    assert_eq!(labels[0], labels[1], "blob A samples must share a cluster");
    assert_eq!(labels[3], labels[4], "blob B samples must share a cluster");
    assert_ne!(
        labels[0], labels[3],
        "the two far-apart blobs must be different clusters"
    );
}

/// SVC (supervised, Predict::Output = Array1<f64>): a linear SVC on separable data labels
/// every sample correctly in the +/-1 domain
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
