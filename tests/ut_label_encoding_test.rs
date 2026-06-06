#![cfg(feature = "utility")]

use ndarray::prelude::*;
use rustyml::error::ModelError;
use rustyml::utility::label_encoding::*;

#[test]
fn test_to_categorical_basic() {
    let labels = array![0, 1, 2, 1, 0];
    let categorical = to_categorical(&labels, None).unwrap();

    assert_eq!(categorical.shape(), &[5, 3]);
    assert_eq!(categorical[[0, 0]], 1.0);
    assert_eq!(categorical[[1, 1]], 1.0);
    assert_eq!(categorical[[2, 2]], 1.0);
    assert_eq!(categorical[[3, 1]], 1.0);
    assert_eq!(categorical[[4, 0]], 1.0);
}

#[test]
fn test_to_categorical_with_num_classes() {
    let labels = array![0, 1];
    let categorical = to_categorical(&labels, Some(5)).unwrap();

    assert_eq!(categorical.shape(), &[2, 5]);
    assert_eq!(categorical[[0, 0]], 1.0);
    assert_eq!(categorical[[1, 1]], 1.0);
}

#[test]
fn test_to_categorical_with_mapping() {
    let labels = vec!["cat", "dog", "bird", "dog"];
    let (categorical, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    assert_eq!(categorical.shape(), &[4, 3]);
    assert_eq!(mapping.len(), 3);
    assert!(mapping.contains_key("cat"));
    assert!(mapping.contains_key("dog"));
    assert!(mapping.contains_key("bird"));
}

#[test]
fn test_to_sparse_categorical() {
    let categorical = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ];
    let labels = to_sparse_categorical(&categorical).unwrap();

    assert_eq!(labels, array![0, 1, 2, 1]);
}

#[test]
fn test_round_trip() {
    let original_labels = array![0, 1, 2, 1, 0];
    let categorical = to_categorical(&original_labels, None).unwrap();
    let recovered_labels = to_sparse_categorical(&categorical).unwrap();

    assert_eq!(original_labels, recovered_labels);
}

#[test]
fn test_to_sparse_categorical_with_probabilities() {
    // Test with soft probabilities (not strict one-hot)
    let categorical = array![[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.1, 0.1, 0.8]];
    let labels = to_sparse_categorical(&categorical).unwrap();

    assert_eq!(labels, array![0, 1, 2]);
}

#[test]
fn test_to_sparse_categorical_rejects_non_finite() {
    // Non-finite inputs must error rather than panic in the argmax comparison.
    let categorical = array![[1.0, f64::NAN, 0.0]];
    assert!(matches!(
        to_sparse_categorical(&categorical),
        Err(ModelError::InputValidationError(_))
    ));
}

#[test]
fn test_ahash_map_functionality() {
    // Test AHashMap functionality specifically
    let labels = vec![10, 20, 30, 20, 10]; // Non-consecutive integers
    let (categorical, mapping) = to_categorical_with_mapping(&labels, None).unwrap();

    assert_eq!(categorical.shape(), &[5, 3]);
    assert_eq!(mapping.len(), 3);

    // Verify all original labels are in the mapping
    assert!(mapping.contains_key(&10));
    assert!(mapping.contains_key(&20));
    assert!(mapping.contains_key(&30));

    // Verify the mapping values are consecutive indices
    let mut indices: Vec<usize> = mapping.values().copied().collect();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_negative_labels() {
    // Negative labels can no longer index a one-hot column: expect a validation error.
    let labels = array![0, -1, 2];
    let err = to_categorical(&labels, None).unwrap_err();
    assert!(matches!(err, ModelError::InputValidationError(msg) if msg.contains("non-negative")));
}

#[test]
fn test_insufficient_num_classes() {
    // num_classes smaller than (max_label + 1) is invalid.
    let labels = array![0, 1, 2];
    let err = to_categorical(&labels, Some(2)).unwrap_err();
    assert!(matches!(err, ModelError::InputValidationError(msg) if msg.contains("num_classes")));
}

#[test]
fn test_to_categorical_with_mapping_insufficient_num_classes() {
    // Fewer classes than unique labels is invalid.
    let labels = vec!["a", "b", "c"];
    assert!(to_categorical_with_mapping(&labels, Some(2)).is_err());
}
