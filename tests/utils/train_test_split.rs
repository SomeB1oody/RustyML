use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::utils::train_test_split::train_test_split;

// ---------------------------------------------------------------------------
// Happy-path: split shapes
// ---------------------------------------------------------------------------

/// 10 samples, test_size=0.3:
///   n_test = round(10 * 0.3) = round(3.0) = 3  → 3 test, 7 train
#[test]
fn test_default_test_size_shapes() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, None, Some(42)).unwrap();

    assert_eq!(x_train.nrows(), 7, "train rows");
    assert_eq!(x_test.nrows(), 3, "test rows");
    assert_eq!(y_train.len(), 7, "train labels");
    assert_eq!(y_test.len(), 3, "test labels");
}

/// 5 samples, test_size=0.4:
///   n_test = round(5 * 0.4) = round(2.0) = 2  → 2 test, 3 train
#[test]
fn test_explicit_test_size_shapes() {
    let x = Array2::from_shape_fn((5, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_iter(0..5i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.4), Some(42)).unwrap();

    assert_eq!(x_train.nrows(), 3, "train rows");
    assert_eq!(x_test.nrows(), 2, "test rows");
    assert_eq!(y_train.len(), 3, "train labels");
    assert_eq!(y_test.len(), 2, "test labels");
}

/// Total row count is always preserved: n_train + n_test == n_samples.
#[test]
fn test_row_count_preserved() {
    let n = 10usize;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..n as i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.25), Some(7)).unwrap();

    assert_eq!(
        x_train.nrows() + x_test.nrows(),
        n,
        "x row count must be preserved"
    );
    assert_eq!(
        y_train.len() + y_test.len(),
        n,
        "y length must be preserved"
    );
}

// ---------------------------------------------------------------------------
// Default test_size = None → 0.3 behaviour
// ---------------------------------------------------------------------------

/// test_size=None must yield the same shapes as test_size=Some(0.3).
/// 10 rows: round(10 * 0.3) = 3 test, 7 train.
#[test]
fn test_none_test_size_defaults_to_0_3() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let (x_train, x_test, _y_train, _y_test) =
        train_test_split(x.clone(), y.clone(), None, Some(42)).unwrap();
    let (x_train_explicit, x_test_explicit, _y_train_e, _y_test_e) =
        train_test_split(x, y, Some(0.3), Some(42)).unwrap();

    assert_eq!(x_train.nrows(), x_train_explicit.nrows());
    assert_eq!(x_test.nrows(), x_test_explicit.nrows());
}

// ---------------------------------------------------------------------------
// Determinism: seed determinism
// ---------------------------------------------------------------------------

/// Same random_state → byte-identical splits.
#[test]
fn test_same_seed_identical_split() {
    let x = Array2::from_shape_fn((20, 3), |(i, j)| (i + j * 5) as f64);
    let y = Array1::from_iter(0..20i32);

    let (xa_train, xa_test, ya_train, ya_test) =
        train_test_split(x.clone(), y.clone(), Some(0.3), Some(42)).unwrap();
    let (xb_train, xb_test, yb_train, yb_test) =
        train_test_split(x, y, Some(0.3), Some(42)).unwrap();

    assert_eq!(
        xa_train, xb_train,
        "x_train must be identical for same seed"
    );
    assert_eq!(xa_test, xb_test, "x_test must be identical for same seed");
    assert_eq!(
        ya_train, yb_train,
        "y_train must be identical for same seed"
    );
    assert_eq!(ya_test, yb_test, "y_test must be identical for same seed");
}

/// Different seeds → different shuffles (probability of collision is negligible at n=100).
#[test]
fn test_different_seeds_different_splits() {
    let x = Array2::from_shape_fn((100, 2), |(i, j)| (i + j * 100) as f64);
    let y = Array1::from_iter(0..100i32);

    let (xa_train, _xa_test, _ya_train, _ya_test) =
        train_test_split(x.clone(), y.clone(), Some(0.3), Some(42)).unwrap();
    let (xb_train, _xb_test, _yb_train, _yb_test) =
        train_test_split(x, y, Some(0.3), Some(99)).unwrap();

    assert_ne!(
        xa_train, xb_train,
        "different seeds should produce different splits"
    );
}

// ---------------------------------------------------------------------------
// Row-value determinism: verify actual values, not just counts
// ---------------------------------------------------------------------------

/// With a known dataset and seed=42, we can derive what ends up in each partition
/// by checking that no data is invented: every row in the output was a row in the input.
///
/// We use unique row-identifying values: row i has x[i, 0] = (i*10) as f64.
/// After split, each output row must trace back to exactly one input row.
#[test]
fn test_output_rows_are_subsets_of_input() {
    // Rows: [0,1], [10,11], [20,21], [30,31], [40,41]
    let data: Vec<f64> = (0..5)
        .flat_map(|i| vec![i as f64 * 10.0, i as f64 * 10.0 + 1.0])
        .collect();
    let x = Array2::from_shape_vec((5, 2), data).unwrap();
    let y = Array1::from_iter(0..5i32);

    let (x_train, x_test, _y_train, _y_test) =
        train_test_split(x.clone(), y, Some(0.4), Some(42)).unwrap();

    // Every row in x_train must appear as a row in x (no invented rows).
    for row in x_train.rows() {
        let found = x.rows().into_iter().any(|r| r == row);
        assert!(found, "x_train row {:?} not found in original x", row);
    }
    // Every row in x_test must also appear in x.
    for row in x_test.rows() {
        let found = x.rows().into_iter().any(|r| r == row);
        assert!(found, "x_test row {:?} not found in original x", row);
    }
    // No row is duplicated across partitions (uniqueness via x[:,0] as key).
    let train_keys: Vec<i32> = x_train.column(0).iter().map(|&v| v as i32).collect();
    let test_keys: Vec<i32> = x_test.column(0).iter().map(|&v| v as i32).collect();
    for tk in &test_keys {
        assert!(
            !train_keys.contains(tk),
            "row key {} appears in both partitions",
            tk
        );
    }
}

// ---------------------------------------------------------------------------
// Row-label alignment: x[i] and y[i] must land in the same partition
// ---------------------------------------------------------------------------

/// Encode the row index in y so we can check alignment.
/// Row i: x[i, 0] = i as f64; y[i] = i * 100.
/// After split, for each row in x_train we know the original index, and y_train
/// at that position must match i * 100.
#[test]
fn test_x_y_alignment() {
    let n = 10usize;
    // x[i, 0] = i as f64 (unique identifier)
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    // y[i] = i * 100 (unique, different scale to prevent confusion)
    let y = Array1::from_iter((0..n as i32).map(|i| i * 100));

    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.3), Some(42)).unwrap();

    // For each position in the train split, x value / 1.0 should be the row index,
    // and y value / 100 should match.
    for (x_val, y_val) in x_train.column(0).iter().zip(y_train.iter()) {
        let row_from_x = *x_val as i32;
        let row_from_y = *y_val / 100;
        assert_eq!(
            row_from_x, row_from_y,
            "alignment broken: x says row {}, y says row {}",
            row_from_x, row_from_y
        );
    }

    // Same for the test partition.
    for (x_val, y_val) in x_test.column(0).iter().zip(y_test.iter()) {
        let row_from_x = *x_val as i32;
        let row_from_y = *y_val / 100;
        assert_eq!(
            row_from_x, row_from_y,
            "alignment broken in test partition: x says row {}, y says row {}",
            row_from_x, row_from_y
        );
    }
}

// ---------------------------------------------------------------------------
// n_samples == 2: always 1 test regardless of test_size
// ---------------------------------------------------------------------------

/// Special case: 2 samples → exactly 1 test row, regardless of test_size.
/// Source line 80-81: `} else if n_samples == 2 { 1 }`.
#[test]
fn test_two_samples_always_one_test() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = Array1::from(vec![10i32, 20i32]);

    // test_size=0.1 would normally give round(2*0.1)=0 → clamped to 1, but special-cased first
    let (x_train_a, x_test_a, _, _) =
        train_test_split(x.clone(), y.clone(), Some(0.1), Some(42)).unwrap();
    assert_eq!(
        x_test_a.nrows(),
        1,
        "n_samples=2, test_size=0.1 → 1 test row"
    );
    assert_eq!(
        x_train_a.nrows(),
        1,
        "n_samples=2, test_size=0.1 → 1 train row"
    );

    // test_size=0.9 would give round(2*0.9)=2 → clamped to 1, but special-cased first
    let (x_train_b, x_test_b, _, _) =
        train_test_split(x.clone(), y.clone(), Some(0.9), Some(42)).unwrap();
    assert_eq!(
        x_test_b.nrows(),
        1,
        "n_samples=2, test_size=0.9 → 1 test row"
    );
    assert_eq!(
        x_train_b.nrows(),
        1,
        "n_samples=2, test_size=0.9 → 1 train row"
    );

    // Even with default test_size
    let (x_train_c, x_test_c, _, _) = train_test_split(x, y, None, Some(42)).unwrap();
    assert_eq!(
        x_test_c.nrows(),
        1,
        "n_samples=2, test_size=None → 1 test row"
    );
    assert_eq!(
        x_train_c.nrows(),
        1,
        "n_samples=2, test_size=None → 1 train row"
    );
}

// ---------------------------------------------------------------------------
// Clamping: test_size near 1.0 or near 0.0
// ---------------------------------------------------------------------------

/// test_size=0.99, 10 samples:
///   n_test = round(10 * 0.99) = round(9.9) = 10, then .min(10-1) = 9 → 9 test, 1 train.
#[test]
fn test_large_test_size_keeps_at_least_one_train() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.99), Some(42)).unwrap();

    // round(10 * 0.99) = round(9.9) = 10, clamped via .min(n_samples-1) = 9
    assert_eq!(x_test.nrows(), 9, "9 test rows expected");
    assert_eq!(x_train.nrows(), 1, "at least 1 train row must remain");
    assert_eq!(y_test.len(), 9);
    assert_eq!(y_train.len(), 1);
}

/// test_size=0.01, 10 samples:
///   n_test = round(10 * 0.01) = round(0.1) = 0, then .max(1) = 1 → 1 test, 9 train.
#[test]
fn test_small_test_size_keeps_at_least_one_test() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.01), Some(42)).unwrap();

    // round(10 * 0.01) = round(0.1) = 0, clamped via .max(1) = 1
    assert_eq!(x_test.nrows(), 1, "at least 1 test row must exist");
    assert_eq!(x_train.nrows(), 9, "9 train rows expected");
    assert_eq!(y_test.len(), 1);
    assert_eq!(y_train.len(), 9);
}

// ---------------------------------------------------------------------------
// Generic label type: &str
// ---------------------------------------------------------------------------

/// Verify the generic A: Clone path works with string slice labels.
#[test]
fn test_str_labels() {
    let x = Array2::from_shape_fn((6, 2), |(i, j)| (i + j) as f64);
    // 6 samples, test_size=0.5: round(6 * 0.5) = round(3.0) = 3 → 3 test, 3 train
    let y = Array1::from(vec!["cat", "dog", "fish", "bird", "rat", "cow"]);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.5), Some(42)).unwrap();

    assert_eq!(x_train.nrows(), 3);
    assert_eq!(x_test.nrows(), 3);
    assert_eq!(y_train.len(), 3);
    assert_eq!(y_test.len(), 3);

    // Every label in y_train/y_test must come from the original set.
    let known = ["cat", "dog", "fish", "bird", "rat", "cow"];
    for lbl in y_train.iter().chain(y_test.iter()) {
        assert!(known.contains(lbl), "unexpected label: {}", lbl);
    }

    // No label should appear in both train and test (all labels are unique here).
    for lbl in y_test.iter() {
        assert!(
            !y_train.iter().any(|l| l == lbl),
            "label {:?} appears in both partitions",
            lbl
        );
    }
}

// ---------------------------------------------------------------------------
// None random_state: non-deterministic but valid
// ---------------------------------------------------------------------------

/// None seed must not panic and must produce a valid split with correct total row count.
#[test]
fn test_none_seed_produces_valid_split() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let result = train_test_split(x, y, Some(0.3), None);
    let (x_train, x_test, y_train, y_test) = result.expect("None seed must succeed");

    // round(10 * 0.3) = 3 test, 7 train
    assert_eq!(x_train.nrows(), 7);
    assert_eq!(x_test.nrows(), 3);
    assert_eq!(y_train.len() + y_test.len(), 10);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// Empty dataset (0 rows) → EmptyInput.
#[test]
fn test_error_empty_input() {
    let x: Array2<f64> = Array2::zeros((0, 2));
    let y: Array1<i32> = Array1::zeros(0);
    let err = train_test_split(x, y, None, None).unwrap_err();
    assert!(
        matches!(err, Error::EmptyInput(_)),
        "expected EmptyInput, got {:?}",
        err
    );
}

/// x.nrows() != y.len() → DimensionMismatch.
#[test]
fn test_error_dimension_mismatch() {
    let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..3i32); // only 3 labels for 5 rows
    let err = train_test_split(x, y, None, Some(42)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::DimensionMismatch {
                expected: 5,
                found: 3
            }
        ),
        "expected DimensionMismatch{{5, 3}}, got {:?}",
        err
    );
}

/// test_size = 0.0 (exact lower boundary, exclusive) → InvalidParameter.
#[test]
fn test_error_test_size_zero() {
    let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..5i32);
    let err = train_test_split(x, y, Some(0.0), Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for test_size=0.0, got {:?}",
        err
    );
}

/// test_size = 1.0 (exact upper boundary, exclusive) → InvalidParameter.
#[test]
fn test_error_test_size_one() {
    let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..5i32);
    let err = train_test_split(x, y, Some(1.0), Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for test_size=1.0, got {:?}",
        err
    );
}

/// test_size < 0.0 → InvalidParameter.
#[test]
fn test_error_test_size_negative() {
    let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..5i32);
    let err = train_test_split(x, y, Some(-0.1), Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for test_size=-0.1, got {:?}",
        err
    );
}

/// test_size > 1.0 → InvalidParameter.
#[test]
fn test_error_test_size_greater_than_one() {
    let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..5i32);
    let err = train_test_split(x, y, Some(1.5), Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter for test_size=1.5, got {:?}",
        err
    );
}

/// n_samples == 1 → InvalidInput (cannot form both a train and test set).
#[test]
fn test_error_single_sample() {
    let x = array![[1.0, 2.0]];
    let y = Array1::from(vec![42i32]);
    let err = train_test_split(x, y, None, Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for n_samples=1, got {:?}",
        err
    );
}
