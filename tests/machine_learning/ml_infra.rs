//! Integration tests for cross-cutting ML infrastructure.
//!
//! Covers generic Fit/Predict trait forwarding across every estimator and a save/load
//! round-trip. Table-driven tests check the shared validation of every estimator: NotFitted
//! before fit, empty and non-finite input, and a missing file on load.
//!
//! Per-type kernel and distance math has unit tests in `src/machine_learning/types.rs`.
//! The `Error` smart constructors have unit tests in `src/error.rs`. This file does not
//! repeat those. It covers only the cross-cutting wiring that the per-estimator files
//! cannot exercise alone.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::{Error, IoError};
use rustyml::machine_learning::DistanceCalculationMetric as Metric;
use rustyml::machine_learning::linear_model::LeastSquaresSolver;
use rustyml::machine_learning::{
    Algorithm, DBSCAN, DecisionTree, EigenSolver, Gamma, Init, IsolationForest, KMeans, KNN,
    KernelPCA, KernelType, LDA, LinearRegression, LinearSVC, LogisticRegression, MeanShift, PCA,
    RegularizationType, SVC, SVDSolver, TSNE, TSNEMethod, WeightingStrategy,
};
use rustyml::traits::{Fit, Predict};

// Fit and Predict traits used generically

/// Trains a supervised estimator through the `Fit` trait, predicts through the `Predict`
/// trait, and returns the number of predictions produced.
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

/// Generic Fit/Predict helper on LinearRegression returns 1 prediction per test point
#[test]
fn generic_fit_predict_with_linear_regression() {
    let x_train = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_train = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let x_test = Array2::from_shape_vec((3, 1), vec![6.0, 7.0, 8.0]).unwrap();

    let mut model = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent {
            learning_rate: 0.01,
            max_iter: 10_000,
            tol: 1e-10,
        })
        .unwrap();
    let n = train_and_predict_count(&mut model, &x_train, &y_train, &x_test);
    assert_eq!(n, 3, "expected 3 predictions for 3 test points");
}

/// Fit/Predict traits on unsupervised KMeans return 1 label per input sample
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

    let mut km = KMeans::new(2, 200, 1e-4).unwrap().with_random_state(42);
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
    let mut model_trait = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent {
            learning_rate: 0.01,
            max_iter: 10_000,
            tol: 1e-10,
        })
        .unwrap();
    Fit::fit(&mut model_trait, (&x_train, &y_train)).unwrap();
    let preds_trait = Predict::predict(&model_trait, &x_test).unwrap();

    // Via inherent method
    let mut model_direct = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent {
            learning_rate: 0.01,
            max_iter: 10_000,
            tol: 1e-10,
        })
        .unwrap();
    model_direct.fit(&x_train, &y_train).unwrap();
    let preds_direct = model_direct.predict(&x_test).unwrap();

    // Both paths must agree exactly, and match the closed-form y = 2*6+1 = 13
    assert_abs_diff_eq!(preds_trait[0], preds_direct[0], epsilon = 0.0);
    assert_abs_diff_eq!(preds_trait[0], 13.0, epsilon = 5e-3);
}

// save_to_path + load_from_path round-trip

/// Builds 3 tight, well-separated blobs centered at (0,0), (100,0), and (50,100) for the
/// KMeans save/load round-trip.
///
/// # Returns
///
/// - `Array2<f64>` - 15x2 matrix of points, 5 per blob
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

/// KMeans reload preserves all hyperparameters
#[test]
fn kmeans_save_load_preserves_hyperparameters() {
    let data = three_blob_data_for_round_trip();

    let mut km = KMeans::new(3, 200, 1e-5).unwrap().with_random_state(7);
    km.fit(&data).unwrap();

    let path = "/tmp/rustyml_ml_infra_kmeans_hyperparams.bin";
    km.save_to_path(path).unwrap();
    let loaded = KMeans::load_from_path(path).unwrap();

    assert_eq!(loaded.get_n_clusters(), km.get_n_clusters());
    assert_eq!(loaded.get_max_iterations(), km.get_max_iterations());
    assert_abs_diff_eq!(loaded.get_tolerance(), km.get_tolerance(), epsilon = 1e-15);
    assert_eq!(loaded.get_random_state(), km.get_random_state());

    let _ = std::fs::remove_file(path);
}

// Shared validation helpers: one table of models and methods

/// Holds 1 instance of each estimator that keeps a fitted state.
///
/// `TSNE` has no fitted state, so only `fit_calls` uses it.
struct Models {
    dbscan: DBSCAN,
    decision_tree: DecisionTree,
    isolation_forest: IsolationForest,
    kernel_pca: KernelPCA,
    kmeans: KMeans,
    knn: KNN<i32>,
    lda: LDA,
    linear_regression: LinearRegression,
    linear_svc: LinearSVC,
    logistic_regression: LogisticRegression,
    mean_shift: MeanShift,
    pca: PCA,
    svc: SVC,
}

/// Builds a `TSNE` with exact optimization and a seeded PCA start.
fn new_tsne() -> TSNE {
    TSNE::new(2, 2.0, 200.0, 100)
        .unwrap()
        .with_random_state(42)
        .with_init(Init::PCA)
        .with_method(TSNEMethod::Exact)
        .unwrap()
}

impl Models {
    /// Builds every estimator with valid hyperparameters and no fitted state.
    fn unfitted() -> Self {
        Self {
            dbscan: DBSCAN::new(1.0, 2).unwrap(),
            decision_tree: DecisionTree::new(Algorithm::CART, true).unwrap(),
            isolation_forest: IsolationForest::new(10, 32).unwrap().with_random_state(42),
            kernel_pca: KernelPCA::new(
                KernelType::RBF {
                    gamma: Gamma::Value(0.5),
                },
                2,
            )
            .unwrap()
            .with_eigen_solver(EigenSolver::Dense),
            kmeans: KMeans::new(2, 100, 1e-4).unwrap().with_random_state(42),
            knn: KNN::<i32>::new(1)
                .unwrap()
                .with_weighting_strategy(WeightingStrategy::Uniform)
                .with_metric(Metric::Euclidean)
                .unwrap(),
            lda: LDA::new(1).unwrap(),
            linear_regression: LinearRegression::new(true)
                .with_solver(LeastSquaresSolver::GradientDescent {
                    learning_rate: 0.01,
                    max_iter: 100,
                    tol: 1e-6,
                })
                .unwrap(),
            linear_svc: LinearSVC::default(),
            logistic_regression: LogisticRegression::default(),
            mean_shift: MeanShift::new(2.0).unwrap(),
            pca: PCA::new(2).unwrap().with_svd_solver(SVDSolver::Full),
            svc: SVC::new(KernelType::Linear, 1.0, 1e-3, 100)
                .unwrap()
                .with_random_state(42),
        }
    }

    /// Builds every estimator and fits it on `two_class_data`.
    fn fitted() -> Self {
        let (x, y) = two_class_data();
        let mut models = Self::unfitted();
        for (model, fit) in fit_calls() {
            if model != "TSNE" {
                fit(&mut models, &x, &y).unwrap_or_else(|e| panic!("{model}::fit: {e:?}"));
            }
        }
        models
    }
}

/// Builds 2 separated classes of 3 samples each, with 2 features.
///
/// # Returns
///
/// - `(Array2<f64>, Array1<f64>)` - the 6x2 features and the labels 0.0 and 1.0
fn two_class_data() -> (Array2<f64>, Array1<f64>) {
    let x = array![
        [0.0, 0.0],
        [0.4, 0.1],
        [0.1, 0.5],
        [5.0, 5.0],
        [5.3, 5.1],
        [5.1, 5.4],
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    (x, y)
}

/// Calls a fit method on 1 estimator in `Models` with the features and the `f64` labels.
type FitCall = fn(&mut Models, &Array2<f64>, &Array1<f64>) -> Result<(), Error>;

/// Calls a method on 1 estimator in `Models` and returns the number of output rows.
type MethodCall = fn(&Models, &Array2<f64>) -> Result<usize, Error>;

/// Lists the fit method of each estimator, with the model name.
///
/// `KNN` and `LDA` take `i32` labels, so their calls convert the labels.
fn fit_calls() -> Vec<(&'static str, FitCall)> {
    vec![
        ("DBSCAN", |m, x, _| m.dbscan.fit(x).map(|_| ())),
        ("DecisionTree", |m, x, y| {
            m.decision_tree.fit(x, y).map(|_| ())
        }),
        ("IsolationForest", |m, x, _| {
            m.isolation_forest.fit(x).map(|_| ())
        }),
        ("KernelPCA", |m, x, _| m.kernel_pca.fit(x).map(|_| ())),
        ("KMeans", |m, x, _| m.kmeans.fit(x).map(|_| ())),
        ("KNN", |m, x, y| {
            m.knn.fit(x, &y.mapv(|v| v as i32)).map(|_| ())
        }),
        ("LDA", |m, x, y| {
            m.lda.fit(x, &y.mapv(|v| v as i32)).map(|_| ())
        }),
        ("LinearRegression", |m, x, y| {
            m.linear_regression.fit(x, y).map(|_| ())
        }),
        ("LinearSVC", |m, x, y| m.linear_svc.fit(x, y).map(|_| ())),
        ("LogisticRegression", |m, x, y| {
            m.logistic_regression.fit(x, y).map(|_| ())
        }),
        ("MeanShift", |m, x, _| m.mean_shift.fit(x).map(|_| ())),
        ("PCA", |m, x, _| m.pca.fit(x).map(|_| ())),
        ("SVC", |m, x, y| m.svc.fit(x, y).map(|_| ())),
        ("TSNE", |_, x, _| new_tsne().fit_transform(x).map(|_| ())),
    ]
}

/// Lists each method that takes a feature matrix after fit, with the model and method names.
fn matrix_method_calls() -> Vec<(&'static str, &'static str, MethodCall)> {
    vec![
        ("DBSCAN", "predict", |m, x| {
            m.dbscan.predict(x).map(|p| p.len())
        }),
        ("DecisionTree", "predict", |m, x| {
            m.decision_tree.predict(x).map(|p| p.len())
        }),
        ("DecisionTree", "predict_proba", |m, x| {
            m.decision_tree.predict_proba(x).map(|p| p.nrows())
        }),
        ("IsolationForest", "predict", |m, x| {
            m.isolation_forest.predict(x).map(|p| p.len())
        }),
        ("IsolationForest", "score_samples", |m, x| {
            m.isolation_forest.score_samples(x).map(|p| p.len())
        }),
        ("IsolationForest", "decision_function", |m, x| {
            m.isolation_forest.decision_function(x).map(|p| p.len())
        }),
        ("KernelPCA", "transform", |m, x| {
            m.kernel_pca.transform(x).map(|p| p.nrows())
        }),
        ("KMeans", "predict", |m, x| {
            m.kmeans.predict(x).map(|p| p.len())
        }),
        ("KNN", "predict", |m, x| m.knn.predict(x).map(|p| p.len())),
        ("KNN", "predict_parallel", |m, x| {
            m.knn.predict_parallel(x).map(|p| p.len())
        }),
        ("LDA", "predict", |m, x| m.lda.predict(x).map(|p| p.len())),
        ("LDA", "transform", |m, x| {
            m.lda.transform(x).map(|p| p.nrows())
        }),
        ("LDA", "decision_function", |m, x| {
            m.lda.decision_function(x).map(|p| p.nrows())
        }),
        ("LDA", "predict_proba", |m, x| {
            m.lda.predict_proba(x).map(|p| p.nrows())
        }),
        ("LinearRegression", "predict", |m, x| {
            m.linear_regression.predict(x).map(|p| p.len())
        }),
        ("LinearRegression", "score", |m, x| {
            m.linear_regression
                .score(x, &Array1::zeros(x.nrows()))
                .map(|_| 1)
        }),
        ("LinearSVC", "predict", |m, x| {
            m.linear_svc.predict(x).map(|p| p.len())
        }),
        ("LinearSVC", "decision_function", |m, x| {
            m.linear_svc.decision_function(x).map(|p| p.len())
        }),
        ("LogisticRegression", "predict", |m, x| {
            m.logistic_regression.predict(x).map(|p| p.len())
        }),
        ("LogisticRegression", "predict_proba", |m, x| {
            m.logistic_regression.predict_proba(x).map(|p| p.len())
        }),
        ("MeanShift", "predict", |m, x| {
            m.mean_shift.predict(x).map(|p| p.len())
        }),
        ("PCA", "transform", |m, x| {
            m.pca.transform(x).map(|p| p.nrows())
        }),
        ("PCA", "inverse_transform", |m, x| {
            m.pca.inverse_transform(x).map(|p| p.nrows())
        }),
        ("SVC", "predict", |m, x| m.svc.predict(x).map(|p| p.len())),
        ("SVC", "decision_function", |m, x| {
            m.svc.decision_function(x).map(|p| p.len())
        }),
    ]
}

/// Records the result of 1 call in a form that `assert_eq!` can compare.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// The call succeeded with this number of output rows
    Rows(usize),
    /// `Error::NotFitted` with this model name
    NotFitted(&'static str),
    /// `Error::EmptyInput`
    EmptyInput,
    /// `Error::NonFinite`
    NonFinite,
    /// A different error, kept as its debug text
    Other(String),
}

impl From<Result<usize, Error>> for Outcome {
    fn from(result: Result<usize, Error>) -> Self {
        match result {
            Ok(rows) => Outcome::Rows(rows),
            Err(Error::NotFitted(model)) => Outcome::NotFitted(model),
            Err(Error::EmptyInput(_)) => Outcome::EmptyInput,
            Err(Error::NonFinite(_)) => Outcome::NonFinite,
            Err(other) => Outcome::Other(format!("{other:?}")),
        }
    }
}

/// Every method that needs a fitted model returns `NotFitted` with its own model name
/// before fit.
#[test]
fn methods_before_fit_return_not_fitted() {
    let models = Models::unfitted();
    let x = array![[1.0, 2.0]];
    let row = [1.0, 2.0];

    let mut calls: Vec<(&str, &str, Result<usize, Error>)> = matrix_method_calls()
        .into_iter()
        .map(|(model, method, call)| (model, method, call(&models, &x)))
        .collect();
    calls.extend([
        (
            "DecisionTree",
            "predict_one",
            models.decision_tree.predict_one(&row).map(|_| 1),
        ),
        (
            "DecisionTree",
            "predict_proba_one",
            models.decision_tree.predict_proba_one(&row).map(|_| 1),
        ),
        (
            "DecisionTree",
            "generate_tree_structure",
            models.decision_tree.generate_tree_structure().map(|_| 1),
        ),
        (
            "IsolationForest",
            "score_sample",
            models.isolation_forest.score_sample(&row).map(|_| 1),
        ),
    ]);

    for (model, method, result) in calls {
        assert_eq!(
            Outcome::from(result),
            Outcome::NotFitted(model),
            "{model}::{method} before fit"
        );
    }
}

/// Every fit method returns `EmptyInput` for a matrix with 0 rows.
#[test]
fn fit_on_empty_input_returns_empty_input() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<f64> = Array1::zeros(0);
    for (model, fit) in fit_calls() {
        let mut models = Models::unfitted();
        let result = fit(&mut models, &x, &y).map(|_| 0);
        assert_eq!(
            Outcome::from(result),
            Outcome::EmptyInput,
            "{model}::fit on a 0-row matrix"
        );
    }
}

/// Every fit method returns `NonFinite` when the features contain NaN, +inf, or -inf.
#[test]
fn fit_on_non_finite_input_returns_non_finite() {
    let (clean_x, y) = two_class_data();
    for sentinel in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut x = clean_x.clone();
        x[[0, 1]] = sentinel;
        for (model, fit) in fit_calls() {
            let mut models = Models::unfitted();
            let result = fit(&mut models, &x, &y).map(|_| 0);
            assert_eq!(
                Outcome::from(result),
                Outcome::NonFinite,
                "{model}::fit with {sentinel:?} in the features"
            );
        }
    }
}

/// After fit, each method returns `EmptyInput` for a matrix with 0 rows.
///
/// `DBSCAN::predict` is the exception: it returns an empty array.
#[test]
fn fitted_methods_on_empty_input() {
    let models = Models::fitted();
    let x: Array2<f64> = Array2::zeros((0, 2));
    for (model, method, call) in matrix_method_calls() {
        let expected = match (model, method) {
            ("DBSCAN", "predict") => Outcome::Rows(0),
            _ => Outcome::EmptyInput,
        };
        assert_eq!(
            Outcome::from(call(&models, &x)),
            expected,
            "{model}::{method} on a 0-row matrix"
        );
    }
}

/// After fit, each method returns `NonFinite` when the input contains NaN, +inf, or -inf.
#[test]
fn fitted_methods_on_non_finite_input_return_non_finite() {
    let models = Models::fitted();
    for sentinel in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let x = array![[sentinel, 1.0], [2.0, 3.0]];
        for (model, method, call) in matrix_method_calls() {
            assert_eq!(
                Outcome::from(call(&models, &x)),
                Outcome::NonFinite,
                "{model}::{method} with {sentinel:?} in the input"
            );
        }
    }
}

/// `load_from_path` on a missing file returns `Error::Io` with the `NotFound` kind.
///
/// The `model_save_and_load_methods!` macro gives each estimator this method.
#[test]
fn load_from_nonexistent_path_returns_io_not_found() {
    type LoadCall = fn(&str) -> Result<(), Error>;
    let loads: [(&str, LoadCall); 13] = [
        ("DBSCAN", |p| DBSCAN::load_from_path(p).map(|_| ())),
        ("DecisionTree", |p| {
            DecisionTree::load_from_path(p).map(|_| ())
        }),
        ("IsolationForest", |p| {
            IsolationForest::load_from_path(p).map(|_| ())
        }),
        ("KernelPCA", |p| KernelPCA::load_from_path(p).map(|_| ())),
        ("KMeans", |p| KMeans::load_from_path(p).map(|_| ())),
        ("KNN", |p| KNN::<i32>::load_from_path(p).map(|_| ())),
        ("LDA", |p| LDA::load_from_path(p).map(|_| ())),
        ("LinearRegression", |p| {
            LinearRegression::load_from_path(p).map(|_| ())
        }),
        ("LinearSVC", |p| LinearSVC::load_from_path(p).map(|_| ())),
        ("LogisticRegression", |p| {
            LogisticRegression::load_from_path(p).map(|_| ())
        }),
        ("MeanShift", |p| MeanShift::load_from_path(p).map(|_| ())),
        ("PCA", |p| PCA::load_from_path(p).map(|_| ())),
        ("SVC", |p| SVC::load_from_path(p).map(|_| ())),
    ];
    let path = "/tmp/rustyml_ml_infra_no_such_file.bin";
    for (model, load) in loads {
        let result = load(path);
        assert!(
            matches!(
                &result,
                Err(Error::Io(IoError::Std(e))) if e.kind() == std::io::ErrorKind::NotFound
            ),
            "{model}::load_from_path on a missing file: got {result:?}"
        );
    }
}

// Fit and Predict trait forwarding for the remaining estimators. Each test confirms
// dispatch for a distinct Predict::Output type.

/// IsolationForest via the generic traits yields labels (`Array1<i32>`), not raw scores.
/// Scores stay reachable through the inherent `score_samples`.
#[test]
fn generic_fit_predict_isolation_forest_outputs_i32_labels() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 50.0, 50.0],
    )
    .unwrap();

    let mut forest = IsolationForest::new(20, 32).unwrap().with_random_state(42);
    Fit::fit(&mut forest, &data).expect("fit via Fit trait should succeed");
    let labels: Array1<i32> =
        Predict::predict(&forest, &data).expect("predict via Predict trait should succeed");

    assert_eq!(labels.len(), 5, "one label per sample");
    for (i, &l) in labels.iter().enumerate() {
        assert!(l == -1 || l == 1, "label[{i}] = {l} not in {{-1, +1}}");
    }

    // Anomaly scores follow -(2^(-E/c)) with E, c > 0, so they fall in [-1, 0).
    let scores = forest
        .score_samples(&data)
        .expect("score_samples should succeed");
    for (i, &s) in scores.iter().enumerate() {
        assert!((-1.0..0.0).contains(&s), "score[{i}] = {s} not in [-1,0)");
    }
}

/// DBSCAN (unsupervised, Predict::Output = Array1<isize>): 2 dense blobs get distinct
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

    let mut db = DBSCAN::new(0.5, 2).unwrap();
    Fit::fit(&mut db, &data).expect("fit via Fit trait should succeed");
    let labels: Array1<isize> =
        Predict::predict(&db, &data).expect("predict via Predict trait should succeed");

    assert_eq!(labels.len(), 6, "one label per sample");
    // Both dense blobs are clusters, so no point is noise and ids are non-negative.
    for (i, &l) in labels.iter().enumerate() {
        assert!(
            l >= 0,
            "label[{i}] = {l} should be a non-negative cluster id"
        );
    }
    assert_ne!(labels[0], labels[3], "the two separated blobs must differ");
}

/// KNN<i32> (supervised, Predict::Output = Array1<i32>): k=1 labels each query with its
/// nearest anchor's class
#[test]
fn generic_fit_predict_knn_outputs_generic_labels() {
    let x_train = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 0.0]).unwrap();
    let y_train = Array1::from_vec(vec![0_i32, 1]);

    let mut knn = KNN::<i32>::new(1)
        .unwrap()
        .with_weighting_strategy(WeightingStrategy::Uniform)
        .with_metric(Metric::Euclidean)
        .unwrap();
    Fit::fit(&mut knn, (&x_train, &y_train)).expect("fit via Fit trait should succeed");

    let x_test = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 9.5, 0.0]).unwrap();
    let preds: Array1<i32> =
        Predict::predict(&knn, &x_test).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 2, "one label per test point");
    // (0.5, 0) is nearest to anchor 0, label 0. (9.5, 0) is nearest to anchor 1, label 1.
    assert_eq!(preds[0], 0, "point near anchor 0 must get label 0");
    assert_eq!(preds[1], 1, "point near anchor 1 must get label 1");
}

/// LDA (supervised, Predict::Output = Array1<i32>) labels 2 well-separated 1-D classes
/// correctly on the training set.
#[test]
fn generic_fit_predict_lda_outputs_i32_labels() {
    let x_train = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]).unwrap();
    let y_train = Array1::from_vec(vec![0_i32, 0, 0, 1, 1, 1]);

    let mut lda = LDA::new(1).unwrap();
    Fit::fit(&mut lda, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<i32> =
        Predict::predict(&lda, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 6, "one label per sample");
    // A wide margin (3 vs 7) separates the classes, so training accuracy must be perfect
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

    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    Fit::fit(&mut tree, (&x_train, &y_train)).expect("fit via Fit trait should succeed");
    let preds: Array1<f64> =
        Predict::predict(&tree, &x_train).expect("predict via Predict trait should succeed");

    assert_eq!(preds.len(), 6, "one label per sample");
    // A single threshold on feature 0 separates the classes, so training error is zero
    for (i, (&p, &t)) in preds.iter().zip(y_train.iter()).enumerate() {
        assert_abs_diff_eq!(p, t, epsilon = 1e-9);
        let _ = i;
    }
}

/// LinearSVC (supervised, Predict::Output = Array1<f64>) classifies separable data into the
/// {0,1} label domain
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

    let mut svc = LinearSVC::new(10_000, 0.01, RegularizationType::L2(0.01), true, 1e-6).unwrap();
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

/// MeanShift (unsupervised, Predict::Output = Array1<isize>): 2 tight blobs each land in
/// 1 cluster, and the 2 blobs land in different clusters.
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

    let mut ms = MeanShift::new(2.0)
        .unwrap()
        .with_max_iter(300)
        .unwrap()
        .with_tolerance(1e-5)
        .unwrap()
        .with_bin_seeding(true)
        .with_cluster_all(true);
    Fit::fit(&mut ms, &data).expect("fit via Fit trait should succeed");
    let labels: Array1<isize> =
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
/// every sample correctly in the 0/1 domain
#[test]
fn generic_fit_predict_svc_outputs_zero_one_labels() {
    let x_train = Array2::from_shape_vec(
        (8, 2),
        vec![
            2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, // class 1
            -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -3.0, -3.0, // class 0
        ],
    )
    .unwrap();
    let y_train = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let mut svc = SVC::new(KernelType::Linear, 10.0, 1e-3, 1000)
        .unwrap()
        .with_random_state(42);
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

// Storage-generic relaxation: `x` and `y` may use different storage types

/// Every supervised estimator accepts an owned feature matrix paired with a borrowed label
/// view, and the reverse pairing of storage types also compiles.
#[test]
fn supervised_fit_accepts_mixed_storage_for_x_and_y() {
    let x = array![
        [-3.0, -2.0],
        [-2.0, -3.0],
        [-2.0, -2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [2.0, 2.0],
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // S1 = OwnedRepr (x), S2 = ViewRepr (y)
    LinearRegression::default()
        .fit(&x, &y.view())
        .expect("LinearRegression: owned x + view y");
    DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .fit(&x, &y.view())
        .expect("DecisionTree: owned x + view y");
    LinearSVC::default()
        .fit(&x, &y.view())
        .expect("LinearSVC: owned x + view y");
    SVC::new(KernelType::Linear, 1.0, 1e-3, 200)
        .unwrap()
        .with_random_state(42)
        .fit(&x, &y.view())
        .expect("SVC: owned x + view y");

    // S1 = ViewRepr (x), S2 = OwnedRepr (y)
    LinearRegression::default()
        .fit(&x.view(), &y)
        .expect("LinearRegression: view x + owned y");
    DecisionTree::new(Algorithm::CART, true)
        .unwrap()
        .fit(&x.view(), &y)
        .expect("DecisionTree: view x + owned y");
    LinearSVC::default()
        .fit(&x.view(), &y)
        .expect("LinearSVC: view x + owned y");
    SVC::new(KernelType::Linear, 1.0, 1e-3, 200)
        .unwrap()
        .with_random_state(42)
        .fit(&x.view(), &y)
        .expect("SVC: view x + owned y");
}

/// `LinearRegression::score` takes its targets independently of the feature matrix's storage
#[test]
fn linear_regression_score_accepts_mixed_storage() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 4.0, 6.0, 8.0];

    let mut model = LinearRegression::default();
    model.fit(&x, &y).expect("fit should succeed");

    let r2 = model
        .score(&x, &y.view())
        .expect("score with a view target");
    assert!(r2.is_finite(), "R^2 should be finite, got {r2}");
}

/// `DecisionTree::predict` and `predict_proba` both accept any storage type that only
/// implements `Data<Elem = f64>`, with no `Send + Sync` bound.
#[test]
fn decision_tree_predict_accepts_plain_data_storage() {
    fn predict_via_plain_bound<S>(tree: &DecisionTree, x: &ndarray::ArrayBase<S, ndarray::Ix2>)
    where
        S: ndarray::Data<Elem = f64>,
    {
        let preds = tree.predict(x).expect("predict should succeed");
        assert_eq!(preds.len(), x.nrows());
    }

    let x = array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let y = array![0.0, 1.0, 0.0, 1.0];

    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
    tree.fit(&x, &y).expect("fit should succeed");

    // Callable from a context that only promises `Data<Elem = f64>`
    predict_via_plain_bound(&tree, &x);
}
