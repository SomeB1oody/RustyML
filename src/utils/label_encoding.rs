//! Label encoding utilities for converting between sparse integer labels and
//! one-hot (categorical) representations
//!
//! Provides `to_categorical` and `to_categorical_with_mapping` for one-hot encoding,
//! and `to_sparse_categorical` for the inverse via per-row argmax

use crate::error::Error;
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

/// Converts sparse categorical labels to one-hot encoded format
///
/// Takes a 1D array of integer labels and produces a 2D one-hot encoded matrix
/// where each row is a sample and each column is a class. The value is 1.0 for
/// the corresponding class and 0.0 for all other classes
///
/// # Parameters
///
/// - `labels` - A 1D array of integer labels (e.g. \[0, 1, 2, 1, 0\])
/// - `num_classes` - Optional class count; if None, inferred from the maximum label value + 1
///
/// # Returns
///
/// - `Result<Array2<f64>, Error>` - A 2D one-hot encoded matrix of shape (n_samples, n_classes)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::label_encoding::to_categorical;
///
/// let labels = array![0, 1, 2, 1, 0];
/// let categorical = to_categorical(&labels, None).unwrap();
/// assert_eq!(
///     categorical,
///     array![
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///         [0.0, 0.0, 1.0],
///         [0.0, 1.0, 0.0],
///         [1.0, 0.0, 0.0]
///     ]
/// );
/// ```
///
/// # Errors
///
/// - [`Error::InvalidInput`] - If any label is negative
/// - [`Error::InvalidParameter`] - If `num_classes` is smaller than the maximum label + 1
pub fn to_categorical<S>(
    labels: &ArrayBase<S, Ix1>,
    num_classes: Option<usize>,
) -> Result<Array2<f64>, Error>
where
    S: Data<Elem = i32>,
{
    let n_samples = labels.len();

    // Reject negative labels: they cannot index a one-hot column
    if let Some(&label) = labels.iter().find(|&&label| label < 0) {
        return Err(Error::invalid_input(format!(
            "Labels must be non-negative, found: {}",
            label
        )));
    }

    let max_label = labels.iter().max().copied().unwrap_or(0) as usize;
    let n_classes = match num_classes {
        Some(n) => {
            if n < max_label + 1 {
                return Err(Error::invalid_parameter(
                    "num_classes",
                    format!(
                        "num_classes ({}) must be at least {} (max_label + 1)",
                        n,
                        max_label + 1
                    ),
                ));
            }
            n
        }
        None => {
            if labels.is_empty() {
                1 // Default to at least 1 class for empty input
            } else {
                max_label + 1
            }
        }
    };

    let mut categorical = Array2::<f64>::zeros((n_samples, n_classes));

    for (i, &label) in labels.iter().enumerate() {
        categorical[[i, label as usize]] = 1.0;
    }

    Ok(categorical)
}

/// Converts sparse categorical labels to one-hot format with a custom label mapping
///
/// Useful when labels are non-consecutive integers or strings that need to be
/// mapped to consecutive integers first. Classes are indexed in first-seen order
///
/// # Parameters
///
/// - `labels` - A slice of labels that can be compared and hashed
/// - `num_classes` - Optional class count; if None, inferred from the number of unique labels
///
/// # Returns
///
/// - `Result<(Array2<f64>, AHashMap<T, usize>), Error>` - The one-hot encoded matrix paired with the mapping from original labels to class indices
///
/// # Examples
///
/// ```rust
/// use rustyml::utils::label_encoding::to_categorical_with_mapping;
///
/// let labels = vec!["cat", "dog", "bird", "dog", "cat"];
/// let (categorical, mapping) = to_categorical_with_mapping(&labels, None).unwrap();
/// // Classes are indexed in first-seen order.
/// assert_eq!(mapping["cat"], 0);
/// assert_eq!(mapping["dog"], 1);
/// assert_eq!(mapping["bird"], 2);
/// // "cat" -> column 0, so the first row is one-hot at index 0.
/// assert_eq!(categorical.row(0).to_vec(), vec![1.0, 0.0, 0.0]);
/// ```
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If `num_classes` is smaller than the number of unique labels
pub fn to_categorical_with_mapping<T>(
    labels: &[T],
    num_classes: Option<usize>,
) -> Result<(Array2<f64>, AHashMap<T, usize>), Error>
where
    T: Clone + Eq + std::hash::Hash,
{
    let n_samples = labels.len();

    // Map unique labels to indices in first-seen order
    let mut label_to_index = AHashMap::new();

    for label in labels.iter() {
        let len = label_to_index.len();
        label_to_index.entry(label.clone()).or_insert(len);
    }

    let unique_classes = label_to_index.len();
    let n_classes = match num_classes {
        Some(n) => {
            if n < unique_classes {
                return Err(Error::invalid_parameter(
                    "num_classes",
                    format!(
                        "num_classes ({}) must be at least the number of unique labels ({})",
                        n, unique_classes
                    ),
                ));
            }
            n
        }
        None => unique_classes,
    };

    let mut categorical = Array2::<f64>::zeros((n_samples, n_classes));

    for (i, label) in labels.iter().enumerate() {
        let class_index = label_to_index[label];
        categorical[[i, class_index]] = 1.0;
    }

    Ok((categorical, label_to_index))
}

/// Converts one-hot encoded format back to sparse categorical labels
///
/// Inverse of `to_categorical`: each row is reduced to the index of its highest
/// value, making it suitable for turning model predictions back into class labels;
/// ties resolve to the first (lowest) index, matching numpy/sklearn/keras `argmax`
///
/// # Parameters
///
/// - `categorical` - A 2D one-hot encoded matrix where each row is a sample
///
/// # Returns
///
/// - `Result<Array1<i32>, Error>` - A 1D array of integer labels in sparse categorical format
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utils::label_encoding::to_sparse_categorical;
///
/// let categorical = array![[1.0, 0.0, 0.0],
///                         [0.0, 1.0, 0.0],
///                         [0.0, 0.0, 1.0]];
/// let sparse_labels = to_sparse_categorical(&categorical).unwrap();
/// assert_eq!(sparse_labels, array![0, 1, 2]);
/// ```
///
/// # Errors
///
/// - [`Error::NonFinite`] - If the input contains NaN or infinite values
pub fn to_sparse_categorical<S>(categorical: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, Error>
where
    S: Data<Elem = f64>,
{
    // Reject non-finite values up front so the per-row argmax comparison is total
    super::validation::check_finite(categorical)?;

    let labels = categorical
        .rows()
        .into_iter()
        .map(|row| {
            // First-index argmax: strict `>` in `reduce` keeps the earliest column on
            // ties (unlike `max_by`, which would keep the last), matching `argmax`
            row.iter()
                .enumerate()
                .reduce(|best, cur| if cur.1 > best.1 { cur } else { best })
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0)
        })
        .collect();

    Ok(labels)
}
