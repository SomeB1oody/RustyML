//! Integration tests for the KNN (K-Nearest Neighbours) classifier: constructor
//! validation, fit/predict error paths, distance metrics, weighting, parallel
//! paths, and save/load round-trips

use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::DistanceCalculationMetric as Metric;
use rustyml::machine_learning::{KNN, WeightingStrategy};

// Constructor validation

/// k=0 must return Error::InvalidParameter
#[test]
fn constructor_k_zero_returns_invalid_parameter() {
    let err = KNN::<i32>::new(0, WeightingStrategy::Uniform, Metric::Euclidean).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// k=1 (minimum legal) must succeed
#[test]
fn constructor_k_one_succeeds() {
    KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
}

/// Minkowski p < 1 is not a valid metric and must return Error::InvalidParameter
#[test]
fn constructor_rejects_minkowski_p_below_one() {
    let err = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Minkowski(0.5)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for Minkowski(0.5), got {err:?}"
    );
}

/// Default constructor must expose k=5, Uniform, Euclidean
#[test]
fn constructor_default_values() {
    let knn = KNN::<i32>::default();
    assert_eq!(knn.get_k(), 5);
    assert_eq!(knn.get_weighting_strategy(), WeightingStrategy::Uniform);
    assert_eq!(knn.get_metric(), Metric::Euclidean);
    assert!(knn.get_x_train().is_none());
}

/// new() must store and expose the exact parameters supplied
#[test]
fn constructor_stores_parameters() {
    let knn = KNN::<i32>::new(3, WeightingStrategy::Distance, Metric::Manhattan).unwrap();
    assert_eq!(knn.get_k(), 3);
    assert_eq!(knn.get_weighting_strategy(), WeightingStrategy::Distance);
    assert_eq!(knn.get_metric(), Metric::Manhattan);
}

// fit error paths

/// Fitting with fewer samples than k returns Error::InvalidInput (k=5, 3 rows)
#[test]
fn fit_fewer_samples_than_k_returns_invalid_input() {
    let mut knn = KNN::<i32>::new(5, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3 rows < k=5
    let y = array![0, 1, 0];
    let err = knn.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// Fitting with y length != x.nrows() returns DimensionMismatch (x has 3 rows, y has 2 labels)
#[test]
fn fit_mismatched_y_length_returns_dimension_mismatch() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3 rows
    let y = array![0, 1]; // only 2 labels
    let err = knn.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

/// Fitting with NaN in x must return Error::NonFinite
#[test]
fn fit_nan_in_x_returns_non_finite() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x = array![[f64::NAN, 2.0]];
    let y = array![0];
    let err = knn.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Fitting with Inf in x must return Error::NonFinite
#[test]
fn fit_inf_in_x_returns_non_finite() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x = array![[f64::INFINITY, 2.0]];
    let y = array![0];
    let err = knn.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// Fitting with an empty x (0 rows) must return Error::EmptyInput
#[test]
fn fit_empty_x_returns_empty_input() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<i32> = Array1::zeros(0);
    let err = knn.fit(&x, &y).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

// predict error paths

/// predict before fit must return Error::NotFitted
#[test]
fn predict_before_fit_returns_not_fitted() {
    let knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_test = array![[1.0, 2.0]];
    let err = knn.predict(&x_test).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {err:?}"
    );
}

/// predict_parallel before fit must return Error::NotFitted
#[test]
fn predict_parallel_before_fit_returns_not_fitted() {
    let knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_test = array![[1.0, 2.0]];
    let err = knn.predict_parallel(&x_test).unwrap_err();
    assert!(
        matches!(err, Error::NotFitted(_)),
        "expected NotFitted, got {err:?}"
    );
}

/// predict with wrong feature count must return Error::DimensionMismatch (train 2, predict 3)
#[test]
fn predict_wrong_feature_count_returns_dimension_mismatch() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0]];
    let y_train = array![0, 1];
    knn.fit(&x_train, &y_train).unwrap();

    // supply 3 features instead of 2
    let x_test = array![[1.0, 2.0, 3.0]];
    let err = knn.predict(&x_test).unwrap_err();
    assert!(
        matches!(err, Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

/// predict with NaN in x_test must return Error::NonFinite
#[test]
fn predict_nan_in_x_test_returns_non_finite() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0]];
    let y_train = array![0, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[f64::NAN, 0.0]];
    let err = knn.predict(&x_test).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}

/// predict with empty x_test must return Error::EmptyInput
#[test]
fn predict_empty_x_test_returns_empty_input() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0]];
    let y_train = array![0, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test: Array2<f64> = Array2::zeros((0, 2));
    let err = knn.predict(&x_test).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {err:?}"
    );
}

// k=1 Euclidean correctness

/// k=1 Euclidean Uniform picks the single nearest neighbour per query
#[test]
fn predict_k1_euclidean_correctness() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [9.5, 0.0], [0.0, 9.5]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(predictions[0], 0, "Q0=(0.5,0) should map to class 0");
    assert_eq!(predictions[1], 1, "Q1=(9.5,0) should map to class 1");
    assert_eq!(predictions[2], 1, "Q2=(0,9.5) should map to class 1");
}

// k=1 Manhattan correctness

/// k=1 Manhattan Uniform picks the single nearest neighbour per query
#[test]
fn predict_k1_manhattan_correctness() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Manhattan).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [9.5, 0.0], [0.0, 9.5]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(predictions[0], 0, "Q0=(0.5,0) should map to class 0");
    assert_eq!(predictions[1], 1, "Q1=(9.5,0) should map to class 1");
    assert_eq!(predictions[2], 1, "Q2=(0,9.5) should map to class 1");
}

// Minkowski p=1 equals Manhattan

/// Minkowski(p=1) predictions must equal Manhattan predictions on the same data
#[test]
fn predict_minkowski_p1_equals_manhattan() {
    // Manhattan reference model
    let mut knn_man = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Manhattan).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn_man.fit(&x_train, &y_train).unwrap();

    // Minkowski(p=1) model
    let mut knn_mink =
        KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Minkowski(1.0)).unwrap();
    knn_mink.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [9.5, 0.0], [0.0, 9.5]];
    let preds_man = knn_man.predict(&x_test).unwrap();
    let preds_mink = knn_mink.predict(&x_test).unwrap();

    assert_eq!(
        preds_man, preds_mink,
        "Minkowski(p=1) predictions must equal Manhattan"
    );
}

// Minkowski p=2 equals Euclidean

/// Minkowski(p=2) predictions must equal Euclidean predictions on the same data
#[test]
fn predict_minkowski_p2_equals_euclidean() {
    // Euclidean reference model
    let mut knn_euc = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn_euc.fit(&x_train, &y_train).unwrap();

    // Minkowski(p=2) model
    let mut knn_mink =
        KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Minkowski(2.0)).unwrap();
    knn_mink.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [9.5, 0.0], [0.0, 9.5]];
    let preds_euc = knn_euc.predict(&x_test).unwrap();
    let preds_mink = knn_mink.predict(&x_test).unwrap();

    assert_eq!(
        preds_euc, preds_mink,
        "Minkowski(p=2) predictions must equal Euclidean"
    );
}

// k=3 majority vote

/// k=3 Euclidean Uniform resolves a clean 3-vs-0 majority per query
#[test]
fn predict_k3_majority_clean() {
    let mut knn = KNN::<i32>::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [10.0, 0.0],
        [11.0, 0.0],
        [12.0, 0.0]
    ];
    let y_train = array![0, 0, 0, 1, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [10.5, 0.0]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(predictions[0], 0, "Q0=(0.5,0) → 3 votes for class 0");
    assert_eq!(predictions[1], 1, "Q1=(10.5,0) → 3 votes for class 1");
}

/// Above 16 features KNN falls back from the kd-tree to the brute-force search; predictions
/// (and the sequential/parallel paths) must remain correct on that path
#[test]
fn predict_high_dimensional_falls_back_to_brute_force() {
    let n_features = 18; // above the kd-tree dimensionality cutoff (16)
    let mut x_train = Array2::<f64>::zeros((6, n_features));
    for i in 0..3 {
        for j in 0..n_features {
            x_train[[i, j]] = 0.1 * i as f64; // class 0 near the origin
        }
    }
    for i in 3..6 {
        for j in 0..n_features {
            x_train[[i, j]] = 10.0 + 0.1 * i as f64; // class 1 near 10
        }
    }
    let y_train = array![0, 0, 0, 1, 1, 1];

    let mut knn = KNN::<i32>::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    knn.fit(&x_train, &y_train).unwrap();

    let mut x_test = Array2::<f64>::zeros((2, n_features));
    for j in 0..n_features {
        x_test[[0, j]] = 0.15; // near class 0
        x_test[[1, j]] = 10.15; // near class 1
    }

    let seq = knn.predict(&x_test).unwrap();
    let par = knn.predict_parallel(&x_test).unwrap();
    assert_eq!(seq[0], 0, "high-dim query near class 0 must predict 0");
    assert_eq!(seq[1], 1, "high-dim query near class 1 must predict 1");
    assert_eq!(
        seq, par,
        "sequential and parallel must agree on the brute-force path"
    );
}

/// k=3 Euclidean Uniform resolves a 2:1 split (2 of 3 neighbours decide the class)
#[test]
fn predict_k3_majority_two_to_one_split() {
    let mut knn = KNN::<i32>::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    // 4 training points - exactly k+1 to keep the problem minimal
    let x_train = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
    let y_train = array![0, 0, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    // Q=(0.5,0): 3 nearest P0,P1,P2 -> 2 class 0, 1 class 1 -> 0
    // Q=(5.5,0): 3 nearest P2,P3,P1 -> 2 class 1, 1 class 0 -> 1
    let x_test = array![[0.5, 0.0], [5.5, 0.0]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(
        predictions[0], 0,
        "Q0=(0.5,0): 2 votes class 0, 1 vote class 1 → class 0"
    );
    assert_eq!(
        predictions[1], 1,
        "Q1=(5.5,0): 2 votes class 1, 1 vote class 0 → class 1"
    );
}

// Distance weighting

/// k=2 Distance weighting: the closer neighbour decides the class
#[test]
fn predict_distance_weighting_closer_wins() {
    let mut knn = KNN::<i32>::new(2, WeightingStrategy::Distance, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0]];
    let y_train = array![0, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[1.0, 0.0], [9.0, 0.0]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(
        predictions[0], 0,
        "Q=(1,0): weight(P0)=1 > weight(P1)=1/9 → class 0"
    );
    assert_eq!(
        predictions[1], 1,
        "Q=(9,0): weight(P1)=1 > weight(P0)=1/9 → class 1"
    );
}

/// Distance weighting: an exact-match query (distance 0) returns the matched point's class
#[test]
fn predict_distance_zero_exact_match_returns_immediately() {
    let mut knn = KNN::<i32>::new(2, WeightingStrategy::Distance, Metric::Euclidean).unwrap();
    // at least k=2 training samples needed to fit
    let x_train = array![[3.0, 4.0], [100.0, 100.0]];
    let y_train = array![7, 99];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[3.0, 4.0]]; // exact match for P0
    let predictions = knn.predict(&x_test).unwrap();

    // exact match wins regardless of the other neighbour
    assert_eq!(predictions[0], 7, "exact-match query must return class 7");
}

// String labels

/// KNN with T=String returns the nearest neighbour's string label
#[test]
fn predict_string_labels() {
    let mut knn = KNN::<String>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train: Array1<String> = Array1::from(vec![
        "cat".to_string(),
        "dog".to_string(),
        "dog".to_string(),
    ]);
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [9.5, 0.0]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(predictions[0], "cat");
    assert_eq!(predictions[1], "dog");
}

// predict == predict_parallel

/// predict and predict_parallel must produce identical results (Uniform, small k)
#[test]
fn predict_and_predict_parallel_agree_small_k() {
    let mut knn = KNN::<i32>::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [10.0, 0.0],
        [11.0, 0.0],
        [12.0, 0.0]
    ];
    let y_train = array![0, 0, 0, 1, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [10.5, 0.0], [1.0, 0.0], [11.0, 0.0]];
    let preds_seq = knn.predict(&x_test).unwrap();
    let preds_par = knn.predict_parallel(&x_test).unwrap();

    assert_eq!(
        preds_seq, preds_par,
        "predict and predict_parallel must agree"
    );
}

/// predict and predict_parallel must also agree with Distance weighting
#[test]
fn predict_and_predict_parallel_agree_distance_weights() {
    let mut knn = KNN::<i32>::new(2, WeightingStrategy::Distance, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[1.0, 0.0], [9.0, 0.0], [0.0, 1.0]];
    let preds_seq = knn.predict(&x_test).unwrap();
    let preds_par = knn.predict_parallel(&x_test).unwrap();

    assert_eq!(
        preds_seq, preds_par,
        "predict and predict_parallel must agree with Distance weighting"
    );
}

// k>=100 parallel voting path

/// k>=100 Uniform exercises the parallel voting branch; both entry points must agree
#[test]
fn predict_parallel_large_k_exercises_parallel_voting_branch() {
    const N: usize = 200; // 100 per class

    // 100 class-0 at x=0..99, 100 class-1 at x=200..299 (all on y=0)
    let mut x_rows: Vec<[f64; 2]> = Vec::with_capacity(N);
    let mut y_vals: Vec<i32> = Vec::with_capacity(N);
    for i in 0..100_usize {
        x_rows.push([i as f64, 0.0]);
        y_vals.push(0);
    }
    for i in 200..300_usize {
        x_rows.push([i as f64, 0.0]);
        y_vals.push(1);
    }
    let x_train =
        Array2::from_shape_vec((N, 2), x_rows.iter().flatten().copied().collect()).unwrap();
    let y_train = Array1::from_vec(y_vals);

    let mut knn = KNN::<i32>::new(100, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[50.0, 0.0], [250.0, 0.0]];

    // sequential path
    let preds_seq = knn.predict(&x_test).unwrap();
    assert_eq!(
        preds_seq[0], 0,
        "Q=(50,0): 100 class-0 neighbours → class 0"
    );
    assert_eq!(
        preds_seq[1], 1,
        "Q=(250,0): 100 class-1 neighbours → class 1"
    );

    // parallel path must match
    let preds_par = knn.predict_parallel(&x_test).unwrap();
    assert_eq!(
        preds_seq, preds_par,
        "parallel path must agree with sequential at k=100"
    );
}

/// k>=100 with Distance weighting exercises the parallel weight-aggregation branch
#[test]
fn predict_parallel_large_k_distance_weighting_exercises_parallel_weight_branch() {
    const N: usize = 200;
    let mut x_rows: Vec<[f64; 2]> = Vec::with_capacity(N);
    let mut y_vals: Vec<i32> = Vec::with_capacity(N);
    for i in 0..100_usize {
        x_rows.push([i as f64 + 1.0, 0.0]); // avoid x=0 to prevent the dist=0 edge case
        y_vals.push(0);
    }
    for i in 200..300_usize {
        x_rows.push([i as f64, 0.0]);
        y_vals.push(1);
    }
    let x_train =
        Array2::from_shape_vec((N, 2), x_rows.iter().flatten().copied().collect()).unwrap();
    let y_train = Array1::from_vec(y_vals);

    let mut knn = KNN::<i32>::new(100, WeightingStrategy::Distance, Metric::Euclidean).unwrap();
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[50.5, 0.0]];
    let preds_seq = knn.predict(&x_test).unwrap();
    let preds_par = knn.predict_parallel(&x_test).unwrap();

    // class-0 points are all within distance <= 100, class-1 points start at distance >= 149.5,
    // so class-0 aggregate weight far exceeds class-1 even with 100 neighbours
    assert_eq!(
        preds_seq[0], 0,
        "distance-weighted: class-0 should win near x=50.5"
    );
    assert_eq!(preds_seq, preds_par, "parallel must agree with sequential");
}

// fit_predict convenience method

/// fit_predict with k=1 on the training data returns every training label unchanged
#[test]
fn fit_predict_returns_training_labels() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0]];
    let y_train = array![0, 1];
    let result = knn.fit_predict(&x_train, &y_train).unwrap();

    assert_eq!(result[0], 0);
    assert_eq!(result[1], 1);
}

// Minkowski p=3 numerical check (end-to-end)

/// k=1 Minkowski(p=3) finds the correct nearest neighbour under the Lp-3 metric
#[test]
fn predict_minkowski_p3_correct_nearest_neighbour() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Minkowski(3.0)).unwrap();
    let x_train = array![[3.0, 0.0], [0.0, 4.0]];
    let y_train = array![0, 1];
    knn.fit(&x_train, &y_train).unwrap();

    // Q=(0,0): dist to P0=3, to P1=4 -> nearest P0 -> class 0
    // Q=(0,3): dist to P0=(54)^(1/3)~=3.78, to P1=1 -> nearest P1 -> class 1
    let x_test = array![[0.0, 0.0], [0.0, 3.0]];
    let predictions = knn.predict(&x_test).unwrap();

    assert_eq!(
        predictions[0], 0,
        "Q=(0,0): Minkowski-3 nearest is P0=(3,0)"
    );
    assert_eq!(
        predictions[1], 1,
        "Q=(0,3): Minkowski-3 nearest is P1=(0,4)"
    );
}

// Save / Load round-trip

/// A reloaded KNN model must produce predictions identical to the original
#[test]
fn save_and_load_produces_identical_predictions() {
    let mut knn = KNN::<i32>::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [10.0, 0.0],
        [11.0, 0.0],
        [12.0, 0.0]
    ];
    let y_train = array![0, 0, 0, 1, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[0.5, 0.0], [5.0, 0.0], [10.5, 0.0]];
    let preds_before = knn.predict(&x_test).unwrap();

    let path = "/tmp/rustyml_knn_save_load_test.json";
    knn.save_to_path(path).unwrap();

    let knn_loaded = KNN::<i32>::load_from_path(path).unwrap();
    let preds_after = knn_loaded.predict(&x_test).unwrap();

    assert_eq!(
        preds_before, preds_after,
        "predictions must be identical after save/load round-trip"
    );

    let _ = std::fs::remove_file(path);
}

/// Save/load preserves the Distance weighting strategy, metric, and k
#[test]
fn save_and_load_preserves_model_metadata() {
    let mut knn = KNN::<i32>::new(7, WeightingStrategy::Distance, Metric::Manhattan).unwrap();
    // 7 training points to satisfy nrows >= k
    let x_train = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0,
        ],
    )
    .unwrap();
    let y_train = array![0, 0, 0, 1, 1, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    let path = "/tmp/rustyml_knn_metadata_test.json";
    knn.save_to_path(path).unwrap();

    let knn_loaded = KNN::<i32>::load_from_path(path).unwrap();
    assert_eq!(knn_loaded.get_k(), 7);
    assert_eq!(
        knn_loaded.get_weighting_strategy(),
        WeightingStrategy::Distance
    );
    assert_eq!(knn_loaded.get_metric(), Metric::Manhattan);

    let _ = std::fs::remove_file(path);
}
