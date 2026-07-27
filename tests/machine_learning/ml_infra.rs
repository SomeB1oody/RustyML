//! Integration tests for cross-cutting ML infrastructure: generic Fit/Predict trait
//! forwarding across every estimator, a save/load round-trip, and predict-before-fit
//! NotFitted errors
//!
//! Per-type kernel/distance math is unit-tested in `src/types.rs`, and the `Error` smart
//! constructors in `src/error.rs`; this file no longer re-covers those, focusing only on
//! the cross-cutting wiring that the per-estimator files cannot exercise in isolation

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::IsolationForest;
use rustyml::machine_learning::KMeans;
use rustyml::machine_learning::KernelType;
use rustyml::machine_learning::LinearRegression;
use rustyml::machine_learning::linear_model::LeastSquaresSolver;
use rustyml::traits::{Fit, Predict};

use rustyml::machine_learning::DBSCAN;
use rustyml::machine_learning::DistanceCalculationMetric as Metric;
use rustyml::machine_learning::LDA;
use rustyml::machine_learning::MeanShift;
use rustyml::machine_learning::SVC;
use rustyml::machine_learning::{Algorithm, DecisionTree};
use rustyml::machine_learning::{KNN, WeightingStrategy};
use rustyml::machine_learning::{LinearSVC, RegularizationType};

// Fit / Predict traits used generically

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

/// Build 3 tight, well-separated blobs centred at (0,0), (100,0), (50,100) for the
/// KMeans save/load round-trip
///
/// # Returns
///
/// - 15x2 matrix of points, 5 per blob
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

// predict() before fit() -> NotFitted

/// LinearRegression (supervised): predict before fit returns NotFitted
#[test]
fn linear_regression_predict_before_fit_is_not_fitted() {
    let model = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent {
            learning_rate: 0.01,
            max_iter: 100,
            tol: 1e-6,
        })
        .unwrap();
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
    let km = KMeans::new(3, 100, 1e-4).unwrap().with_random_state(42);
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
    let forest = IsolationForest::new(10, 32).unwrap().with_random_state(42);
    let x = array![[1.0, 2.0]];
    let result = forest.predict(&x);
    assert!(
        matches!(result, Err(Error::NotFitted(_))),
        "expected NotFitted, got {result:?}"
    );
}

// Fit / Predict trait forwarding for the remaining estimators: each test invokes the
// traits explicitly to confirm dispatch for every distinct Predict::Output type

/// IsolationForest (unsupervised, Fit<&Array2<f64>>, Predict::Output = Array1<i32>):
/// like every other estimator, the trait yields labels, not raw scores. Scores stay
/// reachable through the inherent `score_samples`.
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

    // Anomaly scores are -(2^(-E/c)) with E,c > 0, hence in [-1, 0)
    let scores = forest
        .score_samples(&data)
        .expect("score_samples should succeed");
    for (i, &s) in scores.iter().enumerate() {
        assert!((-1.0..0.0).contains(&s), "score[{i}] = {s} not in [-1,0)");
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

    let mut db = DBSCAN::new(0.5, 2).unwrap();
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

    let mut lda = LDA::new(1).unwrap();
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

    let mut tree = DecisionTree::new(Algorithm::CART, true).unwrap();
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

/// Every supervised estimator accepts an owned feature matrix paired with a borrowed
/// label view (and vice versa). The signatures previously bound both to one `S`, so
/// mixing owned and view storage was a compile error rather than a runtime one.
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

/// `LinearRegression::score` takes its targets independently of the feature matrix' storage
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

/// `DecisionTree::predict` no longer requires `Send + Sync` on the storage type, so a
/// non-Sync-friendly view works — `predict_proba` never had the bound to begin with
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
