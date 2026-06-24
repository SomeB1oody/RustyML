//! Integration tests for `train_test_split`: split shapes, determinism,
//! row/label alignment, clamping, generic label types, and error paths

use ndarray::{Array1, Array2, array};
use rustyml::error::Error;
use rustyml::utils::train_test_split::{train_test_split, train_test_split_stratified};

// Split shapes

/// Default test_size yields 3 test, 7 train rows for 10 samples
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

/// Explicit test_size=0.4 yields 2 test, 3 train rows for 5 samples
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

/// Total row count is preserved: n_train + n_test == n_samples
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

// Default test_size = None behaves as 0.3

/// test_size=None yields the same shapes as test_size=Some(0.3)
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

// Determinism: seed behaviour

/// Same random_state yields byte-identical splits
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

/// Different seeds yield different shuffles (collision negligible at n=100)
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

// Row-value determinism: verify actual values, not just counts

/// Every output row traces back to exactly one input row, with no cross-partition duplicates
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

    // Every x_train row must appear as a row in x
    for row in x_train.rows() {
        let found = x.rows().into_iter().any(|r| r == row);
        assert!(found, "x_train row {:?} not found in original x", row);
    }
    // Every x_test row must appear as a row in x
    for row in x_test.rows() {
        let found = x.rows().into_iter().any(|r| r == row);
        assert!(found, "x_test row {:?} not found in original x", row);
    }
    // No row is duplicated across partitions, keyed by x[:,0]
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

// Row-label alignment: x[i] and y[i] must land in the same partition

/// x[i] and y[i] stay paired after the split, in both partitions
#[test]
fn test_x_y_alignment() {
    let n = 10usize;
    // x[i, 0] = i as f64 (unique identifier)
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    // y[i] = i * 100 (unique, distinct scale)
    let y = Array1::from_iter((0..n as i32).map(|i| i * 100));

    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.3), Some(42)).unwrap();

    // In the train split, the x value and y/100 must agree on the row index
    for (x_val, y_val) in x_train.column(0).iter().zip(y_train.iter()) {
        let row_from_x = *x_val as i32;
        let row_from_y = *y_val / 100;
        assert_eq!(
            row_from_x, row_from_y,
            "alignment broken: x says row {}, y says row {}",
            row_from_x, row_from_y
        );
    }

    // Same check for the test partition
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

// n_samples == 2: always 1 test regardless of test_size

/// With 2 samples, the split is always 1 test and 1 train, regardless of test_size
#[test]
fn test_two_samples_always_one_test() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = Array1::from(vec![10i32, 20i32]);

    // test_size=0.1: 2-sample case is special-cased before any rounding/clamping
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

    // test_size=0.9: same special case applies
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

    // Default test_size: same special case applies
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

// Clamping: test_size near 1.0 or near 0.0

/// test_size=0.99 over 10 samples clamps to 9 test, leaving at least 1 train row
#[test]
fn test_large_test_size_keeps_at_least_one_train() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.99), Some(42)).unwrap();

    // round(10 * 0.99) = 10, clamped via .min(n_samples-1) = 9
    assert_eq!(x_test.nrows(), 9, "9 test rows expected");
    assert_eq!(x_train.nrows(), 1, "at least 1 train row must remain");
    assert_eq!(y_test.len(), 9);
    assert_eq!(y_train.len(), 1);
}

/// test_size=0.01 over 10 samples clamps to 1 test, leaving 9 train rows
#[test]
fn test_small_test_size_keeps_at_least_one_test() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter(0..10i32);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.01), Some(42)).unwrap();

    // round(10 * 0.01) = 0, clamped via .max(1) = 1
    assert_eq!(x_test.nrows(), 1, "at least 1 test row must exist");
    assert_eq!(x_train.nrows(), 9, "9 train rows expected");
    assert_eq!(y_test.len(), 1);
    assert_eq!(y_train.len(), 9);
}

// Generic label type: &str

/// The generic A: Clone path works with string-slice labels
#[test]
fn test_str_labels() {
    let x = Array2::from_shape_fn((6, 2), |(i, j)| (i + j) as f64);
    // 6 samples, test_size=0.5: round(6 * 0.5) = 3 test, 3 train
    let y = Array1::from(vec!["cat", "dog", "fish", "bird", "rat", "cow"]);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.5), Some(42)).unwrap();

    assert_eq!(x_train.nrows(), 3);
    assert_eq!(x_test.nrows(), 3);
    assert_eq!(y_train.len(), 3);
    assert_eq!(y_test.len(), 3);

    // Every label in y_train/y_test must come from the original set
    let known = ["cat", "dog", "fish", "bird", "rat", "cow"];
    for lbl in y_train.iter().chain(y_test.iter()) {
        assert!(known.contains(lbl), "unexpected label: {}", lbl);
    }

    // No label appears in both partitions (labels are all unique here)
    for lbl in y_test.iter() {
        assert!(
            !y_train.iter().any(|l| l == lbl),
            "label {:?} appears in both partitions",
            lbl
        );
    }
}

// None random_state: non-deterministic but valid

/// None seed produces a valid split with the correct total row count
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

// Error paths

/// Empty dataset (0 rows) yields EmptyInput
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

/// x.nrows() != y.len() yields DimensionMismatch
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

/// Out-of-range test_size values all yield InvalidParameter.
///
/// Covers the exclusive lower boundary (0.0), the exclusive upper boundary (1.0),
/// a negative value (-0.1), and a value above 1.0 (1.5).
#[test]
fn test_error_test_size_out_of_range() {
    for test_size in [0.0_f64, 1.0, -0.1, 1.5] {
        let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
        let y = Array1::from_iter(0..5i32);
        let err = train_test_split(x, y, Some(test_size), Some(42)).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "expected InvalidParameter for test_size={}, got {:?}",
            test_size,
            err
        );
    }
}

/// n_samples == 1 yields InvalidInput (cannot form both a train and test set)
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

// Stratified split

/// Every class keeps at least one sample on each side, even when classes are imbalanced
#[test]
fn test_stratified_keeps_every_class_on_both_sides() {
    // 8 of class 0 and 2 of class 1: a plain split could drop class 1 from the train set
    let n = 10usize;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    let mut labels = vec![0i32; 8];
    labels.extend(vec![1i32; 2]);
    let y = Array1::from(labels);

    let (_x_train, _x_test, y_train, y_test) =
        train_test_split_stratified(x, y, Some(0.3), Some(42)).unwrap();

    // Both classes must appear in both partitions
    for class in [0i32, 1i32] {
        assert!(
            y_train.iter().any(|&l| l == class),
            "class {class} missing from train"
        );
        assert!(
            y_test.iter().any(|&l| l == class),
            "class {class} missing from test"
        );
    }
}

/// Per-class counts follow the requested test_size on a balanced dataset
#[test]
fn test_stratified_preserves_class_proportions() {
    // 6 of class 0 and 6 of class 1, test_size=0.5 -> 3 test and 3 train per class
    let n = 12usize;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    let mut labels = vec![0i32; 6];
    labels.extend(vec![1i32; 6]);
    let y = Array1::from(labels);

    let (_x_train, _x_test, y_train, y_test) =
        train_test_split_stratified(x, y, Some(0.5), Some(7)).unwrap();

    let count = |arr: &Array1<i32>, class: i32| arr.iter().filter(|&&l| l == class).count();
    assert_eq!(count(&y_test, 0), 3, "class 0 test count");
    assert_eq!(count(&y_test, 1), 3, "class 1 test count");
    assert_eq!(count(&y_train, 0), 3, "class 0 train count");
    assert_eq!(count(&y_train, 1), 3, "class 1 train count");
}

/// x[i] and y[i] stay paired after a stratified split
#[test]
fn test_stratified_x_y_alignment() {
    let n = 12usize;
    // x[i, 0] encodes the row index, y[i] alternates class but x stays the identifier
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    let y = Array1::from_iter((0..n as i32).map(|i| i % 3));

    let (x_train, x_test, y_train, y_test) =
        train_test_split_stratified(x, y, Some(0.34), Some(42)).unwrap();

    // The class of each output row must equal row_index % 3
    for (x_val, y_val) in x_train
        .column(0)
        .iter()
        .zip(y_train.iter())
        .chain(x_test.column(0).iter().zip(y_test.iter()))
    {
        let row = *x_val as i32;
        assert_eq!(row % 3, *y_val, "alignment broken at row {row}");
    }
}

/// The same seed reproduces a stratified split exactly
#[test]
fn test_stratified_same_seed_identical_split() {
    let n = 12usize;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64);
    let y = Array1::from_iter((0..n as i32).map(|i| i % 2));

    let (xa_train, xa_test, _, _) =
        train_test_split_stratified(x.clone(), y.clone(), Some(0.3), Some(42)).unwrap();
    let (xb_train, xb_test, _, _) = train_test_split_stratified(x, y, Some(0.3), Some(42)).unwrap();

    assert_eq!(xa_train, xb_train, "train must match for same seed");
    assert_eq!(xa_test, xb_test, "test must match for same seed");
}

/// A class with fewer than 2 samples yields InvalidInput
#[test]
fn test_stratified_error_singleton_class() {
    let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
    // Class 2 appears only once, so it cannot land on both sides
    let y = Array1::from(vec![0i32, 0, 1, 1, 2]);
    let err = train_test_split_stratified(x, y, Some(0.3), Some(42)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput for a singleton class, got {:?}",
        err
    );
}
