//! Integration tests for `utils::label_encoding`.
//!
//! Every expected value is derived from the mathematical definition of one-hot
//! encoding or the argmax operation. All arithmetic is shown inline.
//!
//! # One-hot encoding (to_categorical)
//!
//! For labels = [l_0, l_1, ..., l_{n-1}] with n_classes C, the output matrix
//! M has shape (n, C) where M[i, j] = 1.0 iff j == l_i, else 0.0.
//!
//! # Argmax decode (to_sparse_categorical)
//!
//! For a matrix P of shape (n, C), the label for row i is
//!   argmax_j P[i, j]
//! which is the column index of the maximum value in that row.

use ndarray::{arr2, array};
use rustyml::error::Error;
use rustyml::utils::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};

use crate::common::assert_allclose;

// ─────────────────────────────────────────────────────────────────────────
// to_categorical: basic one-hot encoding
// ─────────────────────────────────────────────────────────────────────────

/// Labels [0,1,2,1,0], num_classes=None → inferred max=2, so n_classes=3.
///
/// Hand-computed matrix (row → one-hot column):
///   row 0: label 0 → [1.0, 0.0, 0.0]
///   row 1: label 1 → [0.0, 1.0, 0.0]
///   row 2: label 2 → [0.0, 0.0, 1.0]
///   row 3: label 1 → [0.0, 1.0, 0.0]
///   row 4: label 0 → [1.0, 0.0, 0.0]
#[test]
fn to_categorical_basic_full_matrix() {
    let labels = array![0i32, 1, 2, 1, 0];
    let result = to_categorical(&labels, None).unwrap();

    assert_eq!(result.shape(), &[5, 3], "shape must be (5, 3)");

    let expected = arr2(&[
        [1.0f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    assert_allclose(&result, &expected, 0.0);
}

/// Single-class input: labels = [0], num_classes=None → shape (1, 1).
///
/// Only one class, label 0 → M[0,0] = 1.0.
#[test]
fn to_categorical_single_class() {
    let labels = array![0i32];
    let result = to_categorical(&labels, None).unwrap();

    assert_eq!(result.shape(), &[1, 1]);
    let expected = arr2(&[[1.0f64]]);
    assert_allclose(&result, &expected, 0.0);
}

/// labels = [0, 2], num_classes=Some(5) → shape (2, 5).
///
/// Explicit num_classes=5 is larger than max_label+1=3, so columns 3 and 4
/// must be zero-padded for all rows.
///
///   row 0: label 0 → [1.0, 0.0, 0.0, 0.0, 0.0]
///   row 1: label 2 → [0.0, 0.0, 1.0, 0.0, 0.0]
#[test]
fn to_categorical_explicit_num_classes_wider_with_zero_padding() {
    let labels = array![0i32, 2];
    let result = to_categorical(&labels, Some(5)).unwrap();

    assert_eq!(result.shape(), &[2, 5], "shape must be (2, 5)");

    let expected = arr2(&[[1.0f64, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]);
    assert_allclose(&result, &expected, 0.0);
    // Extra columns (indices 3, 4) are zero — checked by full-matrix comparison above.
    assert_eq!(result[[0, 3]], 0.0);
    assert_eq!(result[[0, 4]], 0.0);
    assert_eq!(result[[1, 3]], 0.0);
    assert_eq!(result[[1, 4]], 0.0);
}

/// Exact num_classes: num_classes=Some(3) with max_label=2 → accepted (3 >= 2+1).
#[test]
fn to_categorical_exact_num_classes_ok() {
    let labels = array![0i32, 1, 2];
    let result = to_categorical(&labels, Some(3)).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    // Sanity-check one cell: row 1 (label=1) has column 1 = 1.0.
    assert_eq!(result[[1, 1]], 1.0);
    assert_eq!(result[[1, 0]], 0.0);
}

/// Empty label array → shape (0, 1). The implementation defaults n_classes=1 when
/// the input is empty. No panic; all-zeros matrix of shape (0, 1).
#[test]
fn to_categorical_empty_labels() {
    let labels = ndarray::Array1::<i32>::from_vec(vec![]); // length-0 array
    let result = to_categorical(&labels, None).unwrap();
    // Source: n_samples=0, n_classes defaults to 1 for empty input.
    assert_eq!(result.shape(), &[0, 1]);
    assert_eq!(result.len(), 0);
}

// ─────────────────────────────────────────────────────────────────────────
// to_categorical: error paths
// ─────────────────────────────────────────────────────────────────────────

/// Negative label -1 must produce InvalidInput (labels must be non-negative).
#[test]
fn to_categorical_negative_label_is_invalid_input() {
    let labels = array![0i32, -1, 2];
    let err = to_categorical(&labels, None).unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput(_)),
        "expected InvalidInput, got {:?}",
        err
    );
}

/// num_classes=2 with max_label=2 means num_classes < max_label+1 = 3 → InvalidParameter.
#[test]
fn to_categorical_num_classes_too_small_is_invalid_parameter() {
    let labels = array![0i32, 1, 2]; // max_label = 2, need >= 3 classes
    let err = to_categorical(&labels, Some(2)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

/// num_classes=0 with max_label=1 → InvalidParameter (0 < 1+1 = 2).
#[test]
fn to_categorical_num_classes_zero_with_labels_is_invalid_parameter() {
    let labels = array![0i32, 1];
    let err = to_categorical(&labels, Some(0)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

// ─────────────────────────────────────────────────────────────────────────
// to_sparse_categorical: argmax decode
// ─────────────────────────────────────────────────────────────────────────

/// Strict one-hot matrix → argmax recovers integer labels exactly.
///
/// Input:
///   [[1.0, 0.0, 0.0],
///    [0.0, 1.0, 0.0],
///    [0.0, 0.0, 1.0]]
///
/// Argmax by row:
///   row 0: max at column 0 → label 0
///   row 1: max at column 1 → label 1
///   row 2: max at column 2 → label 2
#[test]
fn to_sparse_categorical_strict_one_hot() {
    let categorical = arr2(&[[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let labels = to_sparse_categorical(&categorical).unwrap();
    let expected = array![0i32, 1, 2];
    assert_eq!(labels, expected);
}

/// Soft (non-binary) probability matrix: argmax of each row.
///
///   row 0: [0.1, 0.9, 0.0]  → max = 0.9 at col 1 → label 1
///   row 1: [0.2, 0.3, 0.5]  → max = 0.5 at col 2 → label 2
///   row 2: [0.6, 0.3, 0.1]  → max = 0.6 at col 0 → label 0
#[test]
fn to_sparse_categorical_soft_probabilities() {
    let probs = arr2(&[[0.1f64, 0.9, 0.0], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]);
    let labels = to_sparse_categorical(&probs).unwrap();
    let expected = array![1i32, 2, 0];
    assert_eq!(labels, expected);
}

/// Tie-breaking: ties resolve to the FIRST index, matching numpy/sklearn/keras `argmax`.
/// (Rust's `Iterator::max_by` would keep the last element on ties; `to_sparse_categorical`
/// uses a first-wins `reduce` instead to stay consistent with the ecosystem.)
///
///   row 0: [0.5, 0.5, 0.0]  → tie at columns 0 and 1; first wins → label 0
///   row 1: [0.0, 0.5, 0.5]  → tie at columns 1 and 2; first wins → label 1
#[test]
fn to_sparse_categorical_tie_breaks_to_first_index() {
    let tied = arr2(&[[0.5f64, 0.5, 0.0], [0.0, 0.5, 0.5]]);
    let labels = to_sparse_categorical(&tied).unwrap();
    // First-wins argmax: the earliest column among equal maxima is chosen.
    // row 0: index 0 vs index 1 both have 0.5; index 0 (first) wins.
    // row 1: index 1 vs index 2 both have 0.5; index 1 (first) wins.
    assert_eq!(labels[0], 0i32, "tie in row 0 should give index 0 (first)");
    assert_eq!(labels[1], 1i32, "tie in row 1 should give index 1 (first)");
}

// ─────────────────────────────────────────────────────────────────────────
// to_sparse_categorical: error paths
// ─────────────────────────────────────────────────────────────────────────

/// NaN in the matrix must produce NonFinite.
#[test]
fn to_sparse_categorical_nan_is_non_finite() {
    let mat = arr2(&[[0.1f64, f64::NAN, 0.5], [0.2, 0.3, 0.5]]);
    let err = to_sparse_categorical(&mat).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {:?}",
        err
    );
}

/// Positive infinity in the matrix must produce NonFinite.
#[test]
fn to_sparse_categorical_pos_inf_is_non_finite() {
    let mat = arr2(&[[0.0f64, f64::INFINITY, 0.0]]);
    let err = to_sparse_categorical(&mat).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {:?}",
        err
    );
}

/// Negative infinity in the matrix must produce NonFinite.
#[test]
fn to_sparse_categorical_neg_inf_is_non_finite() {
    let mat = arr2(&[[0.5f64, f64::NEG_INFINITY, 0.3]]);
    let err = to_sparse_categorical(&mat).unwrap_err();
    assert!(
        matches!(err, Error::NonFinite(_)),
        "expected NonFinite, got {:?}",
        err
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Round-trip: to_categorical → to_sparse_categorical
// ─────────────────────────────────────────────────────────────────────────

/// encode([0,1,2,1,0]) then decode must recover [0,1,2,1,0].
///
/// This holds because to_categorical produces strict one-hot rows, and
/// argmax of a one-hot row is exactly the index of the 1.
#[test]
fn round_trip_to_categorical_then_to_sparse() {
    let original = array![0i32, 1, 2, 1, 0];
    let one_hot = to_categorical(&original, None).unwrap();
    // Shape sanity: (5, 3)
    assert_eq!(one_hot.shape(), &[5, 3]);

    let recovered = to_sparse_categorical(&one_hot).unwrap();
    assert_eq!(
        recovered, original,
        "round-trip must recover original labels"
    );
}

/// Round-trip with labels [3, 0, 1, 2], num_classes=4.
///
///   encode:
///     row 0: label 3 → [0.0, 0.0, 0.0, 1.0]
///     row 1: label 0 → [1.0, 0.0, 0.0, 0.0]
///     row 2: label 1 → [0.0, 1.0, 0.0, 0.0]
///     row 3: label 2 → [0.0, 0.0, 1.0, 0.0]
///   decode (argmax): [3, 0, 1, 2]
#[test]
fn round_trip_non_monotone_labels() {
    let original = array![3i32, 0, 1, 2];
    let one_hot = to_categorical(&original, Some(4)).unwrap();
    assert_eq!(one_hot.shape(), &[4, 4]);
    let recovered = to_sparse_categorical(&one_hot).unwrap();
    assert_eq!(recovered, original);
}

// ─────────────────────────────────────────────────────────────────────────
// to_categorical_with_mapping: non-consecutive / arbitrary labels
// ─────────────────────────────────────────────────────────────────────────

/// Non-consecutive integer labels [10, 20, 30, 20, 10].
///
/// First-seen insertion order (guaranteed by the or_insert(len) idiom):
///   10 → 0   (first occurrence at position 0)
///   20 → 1   (first occurrence at position 1)
///   30 → 2   (first occurrence at position 2)
///
/// Expected one-hot matrix (5 × 3):
///   row 0: label 10, idx 0 → [1.0, 0.0, 0.0]
///   row 1: label 20, idx 1 → [0.0, 1.0, 0.0]
///   row 2: label 30, idx 2 → [0.0, 0.0, 1.0]
///   row 3: label 20, idx 1 → [0.0, 1.0, 0.0]
///   row 4: label 10, idx 0 → [1.0, 0.0, 0.0]
#[test]
fn to_categorical_with_mapping_non_consecutive_integers() {
    let labels = vec![10i32, 20, 30, 20, 10];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    assert_eq!(matrix.shape(), &[5, 3], "shape must be (5, 3)");

    // Mapping: first-seen insertion order.
    assert_eq!(mapping[&10i32], 0, "10 must map to index 0 (first seen)");
    assert_eq!(mapping[&20i32], 1, "20 must map to index 1 (second seen)");
    assert_eq!(mapping[&30i32], 2, "30 must map to index 2 (third seen)");

    // One-hot positions must match the mapping indices.
    let expected = arr2(&[
        [1.0f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    assert_allclose(&matrix, &expected, 0.0);

    // Cross-check: for each sample, the hot column equals the mapping index.
    for (i, &label) in labels.iter().enumerate() {
        let col = mapping[&label];
        assert_eq!(
            matrix[[i, col]],
            1.0,
            "row {} label {} should be hot at column {}",
            i,
            label,
            col
        );
        for j in 0..3 {
            if j != col {
                assert_eq!(
                    matrix[[i, j]],
                    0.0,
                    "row {} col {} should be 0 (only col {} is hot)",
                    i,
                    j,
                    col
                );
            }
        }
    }
}

/// String labels in first-seen order.
///
///   labels = ["cat", "dog", "bird", "dog", "cat"]
///   first-seen: "cat"→0, "dog"→1, "bird"→2
///
///   row 0: "cat"  idx 0 → [1.0, 0.0, 0.0]
///   row 1: "dog"  idx 1 → [0.0, 1.0, 0.0]
///   row 2: "bird" idx 2 → [0.0, 0.0, 1.0]
///   row 3: "dog"  idx 1 → [0.0, 1.0, 0.0]
///   row 4: "cat"  idx 0 → [1.0, 0.0, 0.0]
#[test]
fn to_categorical_with_mapping_string_labels() {
    let labels = vec!["cat", "dog", "bird", "dog", "cat"];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    assert_eq!(matrix.shape(), &[5, 3]);
    assert_eq!(mapping["cat"], 0, "\"cat\" must map to 0 (first seen)");
    assert_eq!(mapping["dog"], 1, "\"dog\" must map to 1 (second seen)");
    assert_eq!(mapping["bird"], 2, "\"bird\" must map to 2 (third seen)");

    // Verify that the hot column of every row matches the map.
    for (i, &label) in labels.iter().enumerate() {
        let col = mapping[label];
        assert_eq!(
            matrix[[i, col]],
            1.0,
            "row {i} should be hot at column {col}"
        );
        for j in 0..3 {
            if j != col {
                assert_eq!(matrix[[i, j]], 0.0);
            }
        }
    }
}

/// Explicit num_classes > unique count: labels = [1i32, 2], num_classes=Some(5).
/// 2 unique labels; 5 > 2 so OK. Shape must be (2, 5), extra columns zero.
///
///   first-seen: 1→0, 2→1
///   row 0: label 1, idx 0 → [1.0, 0.0, 0.0, 0.0, 0.0]
///   row 1: label 2, idx 1 → [0.0, 1.0, 0.0, 0.0, 0.0]
#[test]
fn to_categorical_with_mapping_explicit_wider_num_classes() {
    let labels = vec![1i32, 2];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, Some(5)).unwrap();

    assert_eq!(matrix.shape(), &[2, 5], "shape must be (2, 5)");
    assert_eq!(mapping[&1i32], 0);
    assert_eq!(mapping[&2i32], 1);

    // Active columns
    assert_eq!(matrix[[0, 0]], 1.0);
    assert_eq!(matrix[[1, 1]], 1.0);
    // Zero-padded columns (indices 2, 3, 4)
    for col in 2..5 {
        assert_eq!(matrix[[0, col]], 0.0, "row 0 col {col} must be 0");
        assert_eq!(matrix[[1, col]], 0.0, "row 1 col {col} must be 0");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// to_categorical_with_mapping: round-trip with to_sparse_categorical
// ─────────────────────────────────────────────────────────────────────────

/// Encode with mapping, then argmax-decode, then re-look-up via inverse map
/// must recover the original labels.
///
///   labels = [10, 20, 30, 20, 10]
///   mapping: 10→0, 20→1, 30→2
///   one-hot → to_sparse_categorical → [0, 1, 2, 1, 0]
///   inverse_map: 0→10, 1→20, 2→30
///   re-mapped: [10, 20, 30, 20, 10]  ✓
#[test]
fn round_trip_with_mapping() {
    let labels = vec![10i32, 20, 30, 20, 10];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    // Build the inverse map.
    let mut inverse: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    for (&label, &idx) in &mapping {
        inverse.insert(idx, label);
    }

    // Argmax-decode the one-hot matrix.
    let sparse = to_sparse_categorical(&matrix).unwrap();
    assert_eq!(sparse.shape(), &[5]);

    // Re-map indices back to original labels.
    let recovered: Vec<i32> = sparse.iter().map(|&idx| inverse[&(idx as usize)]).collect();
    assert_eq!(recovered, labels, "round-trip must recover original labels");
}

// ─────────────────────────────────────────────────────────────────────────
// to_categorical_with_mapping: error paths
// ─────────────────────────────────────────────────────────────────────────

/// num_classes=1 with 2 unique labels → InvalidParameter (1 < 2).
#[test]
fn to_categorical_with_mapping_insufficient_num_classes_is_invalid_parameter() {
    let labels = vec!["a", "b", "c"];
    let err = to_categorical_with_mapping(&labels, Some(2)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

/// num_classes=0 with any non-empty labels → InvalidParameter.
#[test]
fn to_categorical_with_mapping_zero_num_classes_non_empty_is_invalid_parameter() {
    let labels = vec![42i32];
    let err = to_categorical_with_mapping(&labels, Some(0)).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { .. }),
        "expected InvalidParameter, got {:?}",
        err
    );
}

// ─────────────────────────────────────────────────────────────────────────
// to_categorical_with_mapping: empty slice
// ─────────────────────────────────────────────────────────────────────────

/// Empty slice → shape (0, 0). Unlike `to_categorical` (whose None branch
/// defaults n_classes=1 for empty input, giving (0, 1) — see
/// `to_categorical_empty_labels`), the None branch here sets
/// n_classes = unique_classes = 0 because no labels were inserted into the map.
/// So `Array2::zeros((0, 0))` is returned and the mapping is empty.
#[test]
fn to_categorical_with_mapping_empty_slice_is_zero_by_zero() {
    let labels: Vec<i32> = Vec::new(); // length-0 slice; element type fixes T = i32
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    // Source: n_samples=0, unique_classes=0, None → n_classes=0 → shape (0, 0).
    // Contrast with to_categorical's empty case, which is (0, 1).
    assert_eq!(
        matrix.shape(),
        &[0, 0],
        "empty mapping input must give shape (0, 0)"
    );
    assert_eq!(matrix.len(), 0);
    assert!(mapping.is_empty(), "no labels → empty mapping");
}
