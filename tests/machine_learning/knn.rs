//! Integration tests for KNN (K-Nearest Neighbours) classifier.
//!
//! Every expected value is derived from the problem design or a closed-form
//! calculation — never from running the model and recording its output.
//!
//! Notation used in derivations:
//!   dist_E(a,b) = Euclidean distance
//!   dist_M(a,b) = Manhattan (L1) distance
//!   dist_K(p)(a,b) = Minkowski distance with parameter p
//!   Minkowski(1) == Manhattan, Minkowski(2) == Euclidean

use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::machine_learning::DistanceCalculationMetric as Metric;
use rustyml::machine_learning::knn::{KNN, WeightingStrategy};

// ─── Constructor validation ──────────────────────────────────────────────────

/// k=0 must return Error::InvalidParameter.
/// Contract (src/machine_learning/knn.rs:127): "if k == 0 { return Err(Error::invalid_parameter(...)) }"
#[test]
fn constructor_k_zero_returns_invalid_parameter() {
    let err = KNN::<i32>::new(0, WeightingStrategy::Uniform, Metric::Euclidean).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

/// k=1 (minimum legal) must succeed.
#[test]
fn constructor_k_one_succeeds() {
    KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
}

/// Default constructor must expose k=5, Uniform, Euclidean.
/// Contract (src/machine_learning/knn.rs:92-101): default values documented.
#[test]
fn constructor_default_values() {
    let knn = KNN::<i32>::default();
    assert_eq!(knn.get_k(), 5);
    assert_eq!(knn.get_weighting_strategy(), WeightingStrategy::Uniform);
    assert_eq!(knn.get_metric(), Metric::Euclidean);
    assert!(knn.get_x_train().is_none());
}

/// new() must store and expose the exact parameters supplied.
#[test]
fn constructor_stores_parameters() {
    let knn = KNN::<i32>::new(3, WeightingStrategy::Distance, Metric::Manhattan).unwrap();
    assert_eq!(knn.get_k(), 3);
    assert_eq!(knn.get_weighting_strategy(), WeightingStrategy::Distance);
    assert_eq!(knn.get_metric(), Metric::Manhattan);
}

// ─── fit error paths ─────────────────────────────────────────────────────────

/// Fitting with fewer samples than k returns Error::InvalidInput.
/// Contract: "if x.nrows() < self.k { return Err(Error::invalid_input(...)) }"
/// Here k=5 but we supply only 3 rows.
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

/// Fitting with y length != x.nrows() returns DimensionMismatch (instead of panicking at predict).
/// KNN labels are a generic type `T`, so `preliminary_check(x, None)` cannot validate them; `fit`
/// checks the row counts explicitly. Here x has 3 rows but y has only 2 labels.
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

/// Fitting with NaN in x must return Error::NonFinite (via preliminary_check).
/// Contract: preliminary_check checks for non-finite values in x.
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

/// Fitting with Inf in x must return Error::NonFinite.
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

/// Fitting with an empty x (0 rows) must return Error::EmptyInput.
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

// ─── predict error paths ─────────────────────────────────────────────────────

/// predict before fit must return Error::NotFitted.
/// Contract: check_is_fitted fires before any computation.
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

/// predict_parallel before fit must return Error::NotFitted.
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

/// predict with wrong number of features (dimension mismatch) must return
/// Error::DimensionMismatch.
/// Training uses 2 features; predict uses 3 features.
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

/// predict with NaN in x_test must return Error::NonFinite.
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

/// predict with empty x_test must return Error::EmptyInput.
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

// ─── k=1 Euclidean correctness ───────────────────────────────────────────────

/// k=1, Euclidean, Uniform.
///
/// Training set (2-D):
///   P0 = (0, 0)  → class 0
///   P1 = (10, 0) → class 1
///   P2 = (0, 10) → class 1
///
/// Queries:
///   Q0 = (0.5, 0):
///     dist_E(Q0, P0) = 0.5
///     dist_E(Q0, P1) = 9.5
///     dist_E(Q0, P2) = sqrt(0.25 + 100) ≈ 10.012
///     nearest = P0 → class 0 ✓
///
///   Q1 = (9.5, 0):
///     dist_E(Q1, P0) = 9.5
///     dist_E(Q1, P1) = 0.5
///     dist_E(Q1, P2) = sqrt(90.25 + 100) ≈ 13.8
///     nearest = P1 → class 1 ✓
///
///   Q2 = (0, 9.5):
///     dist_E(Q2, P0) = 9.5
///     dist_E(Q2, P1) = sqrt(100 + 90.25) ≈ 13.8
///     dist_E(Q2, P2) = 0.5
///     nearest = P2 → class 1 ✓
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

// ─── k=1 Manhattan correctness ───────────────────────────────────────────────

/// k=1, Manhattan distance, Uniform.
///
/// Training set (2-D, same as Euclidean test):
///   P0 = (0, 0)  → class 0
///   P1 = (10, 0) → class 1
///   P2 = (0, 10) → class 1
///
/// Queries:
///   Q0 = (0.5, 0):
///     dist_M(Q0, P0) = 0.5
///     dist_M(Q0, P1) = 9.5
///     dist_M(Q0, P2) = 0.5 + 10 = 10.5
///     nearest = P0 → class 0 ✓
///
///   Q1 = (9.5, 0):
///     dist_M(Q1, P0) = 9.5
///     dist_M(Q1, P1) = 0.5
///     dist_M(Q1, P2) = 9.5 + 10 = 19.5
///     nearest = P1 → class 1 ✓
///
///   Q2 = (0, 9.5):
///     dist_M(Q2, P0) = 9.5
///     dist_M(Q2, P1) = 10 + 9.5 = 19.5
///     dist_M(Q2, P2) = 0.5
///     nearest = P2 → class 1 ✓
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

// ─── Minkowski p=1 equals Manhattan ─────────────────────────────────────────

/// Minkowski(p=1) must equal Manhattan by definition:
///   dist_K(1)(a,b) = sum |a_i - b_i| = dist_M(a,b)
///
/// We use the same 3-point layout and verify predictions match the Manhattan
/// result, confirming the metric integration is correct end-to-end.
#[test]
fn predict_minkowski_p1_equals_manhattan() {
    // Build Manhattan model (reference)
    let mut knn_man = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Manhattan).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn_man.fit(&x_train, &y_train).unwrap();

    // Build Minkowski(p=1) model
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

// ─── Minkowski p=2 equals Euclidean ─────────────────────────────────────────

/// Minkowski(p=2) must equal Euclidean by definition:
///   dist_K(2)(a,b) = (sum (a_i - b_i)^2)^(1/2) = dist_E(a,b)
#[test]
fn predict_minkowski_p2_equals_euclidean() {
    // Build Euclidean model (reference)
    let mut knn_euc = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
    let y_train = array![0, 1, 1];
    knn_euc.fit(&x_train, &y_train).unwrap();

    // Build Minkowski(p=2) model
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

// ─── k=3 majority vote ───────────────────────────────────────────────────────

/// k=3, Euclidean, Uniform — clean 3-vs-0 majority.
///
/// Training set (1-D embedded in 2-D):
///   P0=(0,0)->0, P1=(1,0)->0, P2=(2,0)->0  ← cluster of class 0
///   P3=(10,0)->1, P4=(11,0)->1, P5=(12,0)->1  ← cluster of class 1
///
/// Queries (6 training points, need exactly 6 for k=3 to not exceed nrows):
///   Q0=(0.5,0): distances {P0=0.5, P1=0.5, P2=1.5, P3=9.5, ...}
///               3 nearest: P0,P1,P2 → 3 votes class 0 → class 0 ✓
///   Q1=(10.5,0): distances {P3=0.5, P4=0.5, P5=1.5, ...}
///               3 nearest: P3,P4,P5 → 3 votes class 1 → class 1 ✓
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

/// k=3, Euclidean, Uniform — 2:1 split majority.
///
/// Training set (4 points):
///   P0=(0,0)->0, P1=(1,0)->0, P2=(2,0)->0, P3=(20,0)->1
///
/// Query Q=(0.5,0):
///   dist to P0=0.5, P1=0.5, P2=1.5, P3=19.5
///   3 nearest: P0,P1,P2 → votes: class 0=3, class 1=0 → class 0 ✓
///
/// Query Q2=(1.5,0):
///   dist to P0=1.5, P1=0.5, P2=0.5, P3=18.5
///   3 nearest: P1,P2,P0 → all class 0 → class 0 ✓
///
/// This exercises the "2:1 split" scenario where 2 of 3 neighbors agree:
/// Use 3 class-0 and 1 class-1, query midpoint between class-0 and class-1:
///
/// Training: P0=(0,0)->0, P1=(1,0)->0, P2=(5,0)->1, P3=(6,0)->1 (4 points)
/// Query (0.5,0): k=3 nearest = P0(0.5), P1(0.5), P2(4.5) → 2×class0, 1×class1 → class 0
/// Query (5.5,0): k=3 nearest = P2(0.5), P3(0.5), P1(4.5) → 2×class1, 1×class0 → class 1
#[test]
fn predict_k3_majority_two_to_one_split() {
    let mut knn = KNN::<i32>::new(3, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    // 4 training points — exactly k+1 to keep the problem minimal
    let x_train = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
    let y_train = array![0, 0, 1, 1];
    knn.fit(&x_train, &y_train).unwrap();

    // Q=(0.5,0): 3 nearest = P0(dist=0.5), P1(dist=0.5), P2(dist=4.5) → 2×0, 1×1 → 0
    // Q=(5.5,0): 3 nearest = P2(dist=0.5), P3(dist=0.5), P1(dist=4.5) → 2×1, 1×0 → 1
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

// ─── Distance weighting ──────────────────────────────────────────────────────

/// k=2, Distance weighting: closer neighbour has more influence.
///
/// Training set (on x-axis):
///   P0=(0,0)->class 0, P1=(10,0)->class 1
///
/// Query Q=(1,0):
///   dist to P0 = 1.0  → weight = 1/1 = 1.0
///   dist to P1 = 9.0  → weight = 1/9 ≈ 0.111
///   class 0 total weight = 1.0
///   class 1 total weight = 0.111
///   → class 0 wins ✓
///
/// Query Q=(9,0):
///   dist to P0 = 9.0  → weight = 1/9 ≈ 0.111
///   dist to P1 = 1.0  → weight = 1/1 = 1.0
///   → class 1 wins ✓
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

/// Distance weighting: exact-match query (distance == 0) returns the
/// matched point's class immediately (early-return code path in predict_one).
///
/// Contract (src/machine_learning/knn.rs:396-399):
///   "if distance == 0.0 { return Ok(y_train_encoded[idx]); }"
///
/// Training: P0=(3,4)->class 7.
/// Query=(3,4): distance to P0 = 0 → immediate return of class 7.
#[test]
fn predict_distance_zero_exact_match_returns_immediately() {
    let mut knn = KNN::<i32>::new(2, WeightingStrategy::Distance, Metric::Euclidean).unwrap();
    // Need at least k=2 training samples to fit
    let x_train = array![[3.0, 4.0], [100.0, 100.0]];
    let y_train = array![7, 99];
    knn.fit(&x_train, &y_train).unwrap();

    let x_test = array![[3.0, 4.0]]; // exact match for P0
    let predictions = knn.predict(&x_test).unwrap();

    // Regardless of the other neighbour, the exact match must win
    assert_eq!(predictions[0], 7, "exact-match query must return class 7");
}

// ─── String labels ───────────────────────────────────────────────────────────

/// KNN is generic: T=String must work correctly.
///
/// Same spatial layout as the basic Euclidean test:
///   P0=(0,0)->  "cat"
///   P1=(10,0)-> "dog"
///   P2=(0,10)-> "dog"
///
/// Q0=(0.5,0) → nearest P0 → "cat"
/// Q1=(9.5,0) → nearest P1 → "dog"
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

// ─── predict == predict_parallel ─────────────────────────────────────────────

/// predict and predict_parallel must produce identical results on the same data
/// (both use Uniform weighting, k < 100 threshold — sequential inner path).
///
/// This is a contractual requirement: both entry points call predict_one
/// identically; any divergence would indicate a threading bug.
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

/// predict and predict_parallel must also agree with Distance weighting.
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

// ─── k>=100 parallel voting path ─────────────────────────────────────────────

/// When k >= 100 the Uniform voting branch switches to parallel aggregation
/// (VOTING_PARALLEL_THRESHOLD = 100 in predict_one).
///
/// Design: 200 training points split evenly: 100 class-0 points at x∈[0,99]
/// and 100 class-1 points at x∈[200,299].
///
/// Query Q=(50,0): 100 nearest are class-0 → majority vote = class 0.
/// Query Q=(250,0): 100 nearest are class-1 → majority vote = class 1.
///
/// This exercises the k>=100 parallel branch of predict_one AND
/// predict_parallel's outer parallelism, so both entry points are tested.
#[test]
fn predict_parallel_large_k_exercises_parallel_voting_branch() {
    const N: usize = 200; // 100 per class

    // Build training data: 100 class-0 at y=0,x=0..99 and 100 class-1 at y=0,x=200..299
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

    // parallel path — must match
    let preds_par = knn.predict_parallel(&x_test).unwrap();
    assert_eq!(
        preds_seq, preds_par,
        "parallel path must agree with sequential at k=100"
    );
}

/// k>=100 with Distance weighting exercises the parallel weight-aggregation branch.
///
/// Design: same 200-point split layout.
/// Query at (50,0): all 100 class-0 points are within [0,99]×0 and much closer
/// than any class-1 point (nearest class-1 is at (200,0), distance=150). Even
/// with inverse-distance weighting the aggregate class-0 weight enormously
/// exceeds class-1 weight.
#[test]
fn predict_parallel_large_k_distance_weighting_exercises_parallel_weight_branch() {
    const N: usize = 200;
    let mut x_rows: Vec<[f64; 2]> = Vec::with_capacity(N);
    let mut y_vals: Vec<i32> = Vec::with_capacity(N);
    for i in 0..100_usize {
        x_rows.push([i as f64 + 1.0, 0.0]); // avoid x=0 to prevent dist=0 edge case
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

    // class-0 points are all within distance ≤ 100, class-1 points start at distance ≥ 149.5
    // Even with 100 neighbours, class-0 aggregate weight >> class-1 aggregate weight.
    assert_eq!(
        preds_seq[0], 0,
        "distance-weighted: class-0 should win near x=50.5"
    );
    assert_eq!(preds_seq, preds_par, "parallel must agree with sequential");
}

// ─── fit_predict convenience method ──────────────────────────────────────────

/// fit_predict on training data: with k=1 each point's nearest neighbour is
/// itself (distance 0), so every training label is returned unchanged.
///
/// Training: P0=(0,0)->0, P1=(10,0)->1.
/// fit_predict on the same x_train → [0, 1].
#[test]
fn fit_predict_returns_training_labels() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Euclidean).unwrap();
    let x_train = array![[0.0, 0.0], [10.0, 0.0]];
    let y_train = array![0, 1];
    let result = knn.fit_predict(&x_train, &y_train).unwrap();

    assert_eq!(result[0], 0);
    assert_eq!(result[1], 1);
}

// ─── Minkowski p=3 numerical check (end-to-end) ──────────────────────────────

/// k=1, Minkowski(p=3), verify the correct nearest-neighbour is found using
/// the Lp-3 metric.
///
/// Training (axis-aligned):
///   P0=(3,0)->class 0, P1=(0,4)->class 1
///
/// Query Q=(0,0):
///   dist_K(3)(Q, P0) = (3^3)^(1/3) = 3
///   dist_K(3)(Q, P1) = (4^3)^(1/3) = 4
///   nearest = P0 → class 0 ✓
///
/// Query Q2=(0,3):
///   dist_K(3)(Q2, P0) = (3^3 + 3^3)^(1/3) = (54)^(1/3) ≈ 3.780
///   dist_K(3)(Q2, P1) = (0^3 + 1^3)^(1/3) = 1
///   nearest = P1 → class 1 ✓
#[test]
fn predict_minkowski_p3_correct_nearest_neighbour() {
    let mut knn = KNN::<i32>::new(1, WeightingStrategy::Uniform, Metric::Minkowski(3.0)).unwrap();
    let x_train = array![[3.0, 0.0], [0.0, 4.0]];
    let y_train = array![0, 1];
    knn.fit(&x_train, &y_train).unwrap();

    // Q=(0,0): dist_K3 to P0=3, to P1=4 → nearest P0 → class 0
    // Q=(0,3): dist_K3 to P0=(54)^(1/3)≈3.78, to P1=1 → nearest P1 → class 1
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

// ─── Save / Load round-trip ───────────────────────────────────────────────────

/// Serialise a fitted KNN model to disk and reload it; predictions from
/// the reloaded model must be identical to predictions from the original.
///
/// The save_to_path / load_from_path methods are generated by the
/// model_save_and_load_methods! macro and use serde_json for persistence.
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

    // Clean up
    let _ = std::fs::remove_file(path);
}

/// Save/load also preserves the Distance weighting strategy and metric.
/// Verify by checking the loaded model's getters.
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
