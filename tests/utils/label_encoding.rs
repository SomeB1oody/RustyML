//! Integration tests for `utils::label_encoding`: one-hot encoding
//! (`to_categorical`, `to_categorical_with_mapping`) and argmax decode
//! (`to_sparse_categorical`), covering happy paths, error paths, and round-trips

use ndarray::{arr2, array};
use rustyml::error::Error;
use rustyml::utils::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};

use crate::common::assert_allclose;

// to_categorical: basic one-hot encoding

/// Labels [0,1,2,1,0] with num_classes=None infer n_classes=3 and encode to one-hot
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

/// Single-class input labels=[0] with num_classes=None gives shape (1, 1)
#[test]
fn to_categorical_single_class() {
    let labels = array![0i32];
    let result = to_categorical(&labels, None).unwrap();

    assert_eq!(result.shape(), &[1, 1]);
    let expected = arr2(&[[1.0f64]]);
    assert_allclose(&result, &expected, 0.0);
}

/// Explicit num_classes=Some(5) wider than max_label+1=3 zero-pads columns 3 and 4
#[test]
fn to_categorical_explicit_num_classes_wider_with_zero_padding() {
    let labels = array![0i32, 2];
    let result = to_categorical(&labels, Some(5)).unwrap();

    assert_eq!(result.shape(), &[2, 5], "shape must be (2, 5)");

    let expected = arr2(&[[1.0f64, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]);
    assert_allclose(&result, &expected, 0.0);
    // Extra columns (indices 3, 4) are zero
    assert_eq!(result[[0, 3]], 0.0);
    assert_eq!(result[[0, 4]], 0.0);
    assert_eq!(result[[1, 3]], 0.0);
    assert_eq!(result[[1, 4]], 0.0);
}

/// num_classes=Some(3) with max_label=2 is accepted since 3 >= 2+1
#[test]
fn to_categorical_exact_num_classes_ok() {
    let labels = array![0i32, 1, 2];
    let result = to_categorical(&labels, Some(3)).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    // Row 1 (label 1) is hot at column 1
    assert_eq!(result[[1, 1]], 1.0);
    assert_eq!(result[[1, 0]], 0.0);
}

/// Empty label array yields shape (0, 1) since n_classes defaults to 1 for empty input
#[test]
fn to_categorical_empty_labels() {
    let labels = ndarray::Array1::<i32>::from_vec(vec![]); // length-0 array
    let result = to_categorical(&labels, None).unwrap();
    assert_eq!(result.shape(), &[0, 1]);
    assert_eq!(result.len(), 0);
}

// to_categorical: error paths

/// Negative label -1 produces InvalidInput
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

/// num_classes=2 with max_label=2 is too small (2 < 3) and gives InvalidParameter
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

/// num_classes=0 with max_label=1 gives InvalidParameter (0 < 2)
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

// to_sparse_categorical: argmax decode

/// Argmax of a strict one-hot matrix recovers the integer labels exactly
#[test]
fn to_sparse_categorical_strict_one_hot() {
    let categorical = arr2(&[[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let labels = to_sparse_categorical(&categorical).unwrap();
    let expected = array![0i32, 1, 2];
    assert_eq!(labels, expected);
}

/// Argmax of a soft (non-binary) probability matrix picks the max column per row
#[test]
fn to_sparse_categorical_soft_probabilities() {
    let probs = arr2(&[[0.1f64, 0.9, 0.0], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]);
    let labels = to_sparse_categorical(&probs).unwrap();
    let expected = array![1i32, 2, 0];
    assert_eq!(labels, expected);
}

/// Ties resolve to the first index, matching numpy/sklearn/keras argmax semantics
#[test]
fn to_sparse_categorical_tie_breaks_to_first_index() {
    let tied = arr2(&[[0.5f64, 0.5, 0.0], [0.0, 0.5, 0.5]]);
    let labels = to_sparse_categorical(&tied).unwrap();
    // First-wins argmax: the earliest column among equal maxima is chosen
    assert_eq!(labels[0], 0i32, "tie in row 0 should give index 0 (first)");
    assert_eq!(labels[1], 1i32, "tie in row 1 should give index 1 (first)");
}

// to_sparse_categorical: error paths

/// NaN in the matrix produces NonFinite
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

/// Positive infinity in the matrix produces NonFinite
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

/// Negative infinity in the matrix produces NonFinite
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

// Round-trip: to_categorical -> to_sparse_categorical

/// Encoding [0,1,2,1,0] then decoding recovers the original labels
#[test]
fn round_trip_to_categorical_then_to_sparse() {
    let original = array![0i32, 1, 2, 1, 0];
    let one_hot = to_categorical(&original, None).unwrap();
    assert_eq!(one_hot.shape(), &[5, 3]);

    let recovered = to_sparse_categorical(&one_hot).unwrap();
    assert_eq!(
        recovered, original,
        "round-trip must recover original labels"
    );
}

/// Round-trip with non-monotone labels [3, 0, 1, 2] and num_classes=4
#[test]
fn round_trip_non_monotone_labels() {
    let original = array![3i32, 0, 1, 2];
    let one_hot = to_categorical(&original, Some(4)).unwrap();
    assert_eq!(one_hot.shape(), &[4, 4]);
    let recovered = to_sparse_categorical(&one_hot).unwrap();
    assert_eq!(recovered, original);
}

// to_categorical_with_mapping: non-consecutive / arbitrary labels

/// Non-consecutive integers [10,20,30,20,10] map in first-seen order and encode to one-hot
#[test]
fn to_categorical_with_mapping_non_consecutive_integers() {
    let labels = vec![10i32, 20, 30, 20, 10];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    assert_eq!(matrix.shape(), &[5, 3], "shape must be (5, 3)");

    // Mapping follows first-seen insertion order
    assert_eq!(mapping[&10i32], 0, "10 must map to index 0 (first seen)");
    assert_eq!(mapping[&20i32], 1, "20 must map to index 1 (second seen)");
    assert_eq!(mapping[&30i32], 2, "30 must map to index 2 (third seen)");

    // One-hot positions match the mapping indices
    let expected = arr2(&[
        [1.0f64, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    assert_allclose(&matrix, &expected, 0.0);

    // For each sample, the hot column equals the mapping index
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

/// String labels map and encode to one-hot in first-seen order
#[test]
fn to_categorical_with_mapping_string_labels() {
    let labels = vec!["cat", "dog", "bird", "dog", "cat"];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    assert_eq!(matrix.shape(), &[5, 3]);
    assert_eq!(mapping["cat"], 0, "\"cat\" must map to 0 (first seen)");
    assert_eq!(mapping["dog"], 1, "\"dog\" must map to 1 (second seen)");
    assert_eq!(mapping["bird"], 2, "\"bird\" must map to 2 (third seen)");

    // The hot column of every row matches the map
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

/// Explicit num_classes=Some(5) wider than the 2 unique labels gives shape (2, 5) with zero padding
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
    // Zero-padded columns (indices 2, 3, 4) are all zero
    for col in 2..5 {
        assert_eq!(matrix[[0, col]], 0.0, "row 0 col {col} must be 0");
        assert_eq!(matrix[[1, col]], 0.0, "row 1 col {col} must be 0");
    }
}

// to_categorical_with_mapping: round-trip with to_sparse_categorical

/// Encode with mapping, argmax-decode, then re-look-up via inverse map recovers original labels
#[test]
fn round_trip_with_mapping() {
    let labels = vec![10i32, 20, 30, 20, 10];
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    // Build the inverse map
    let mut inverse: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    for (&label, &idx) in &mapping {
        inverse.insert(idx, label);
    }

    // Argmax-decode the one-hot matrix
    let sparse = to_sparse_categorical(&matrix).unwrap();
    assert_eq!(sparse.shape(), &[5]);

    // Re-map indices back to original labels
    let recovered: Vec<i32> = sparse.iter().map(|&idx| inverse[&(idx as usize)]).collect();
    assert_eq!(recovered, labels, "round-trip must recover original labels");
}

// to_categorical_with_mapping: error paths

/// num_classes=2 with 3 unique labels gives InvalidParameter
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

/// num_classes=0 with non-empty labels gives InvalidParameter
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

// to_categorical_with_mapping: empty slice

/// Empty slice gives shape (0, 0) and an empty mapping, unlike to_categorical's (0, 1)
#[test]
fn to_categorical_with_mapping_empty_slice_is_zero_by_zero() {
    let labels: Vec<i32> = Vec::new(); // length-0 slice; element type fixes T = i32
    let (matrix, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    // n_samples=0 and unique_classes=0, so None gives n_classes=0 and shape (0, 0)
    assert_eq!(
        matrix.shape(),
        &[0, 0],
        "empty mapping input must give shape (0, 0)"
    );
    assert_eq!(matrix.len(), 0);
    assert!(mapping.is_empty(), "no labels → empty mapping");
}
