use crate::error::Error;
use ahash::AHashMap;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

/// Converts sparse categorical labels to categorical (one-hot encoded) format
///
/// This function takes a 1D array of integer labels and converts them to a 2D
/// one-hot encoded matrix where each row represents a sample and each column
/// represents a class. The value is 1.0 for the corresponding class and 0.0
/// for all other classes.
///
/// # Parameters
///
/// - `labels` - A 1D array of integer labels (e.g., \[0, 1, 2, 1, 0\])
/// - `num_classes` - Optional number of classes. If None, it will be inferred from the maximum label value + 1
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

    // Reject negative labels: they cannot index a one-hot column.
    if let Some(&label) = labels.iter().find(|&&label| label < 0) {
        return Err(Error::invalid_input(format!(
            "Labels must be non-negative, found: {}",
            label
        )));
    }

    // Determine number of classes
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

    // Create one-hot encoded matrix
    let mut categorical = Array2::<f64>::zeros((n_samples, n_classes));

    for (i, &label) in labels.iter().enumerate() {
        categorical[[i, label as usize]] = 1.0;
    }

    Ok(categorical)
}

/// Converts sparse categorical labels to categorical format with custom mapping
///
/// This function is useful when you have non-consecutive integer labels or
/// string labels that need to be mapped to consecutive integers first.
///
/// # Parameters
///
/// - `labels` - A slice of labels that can be compared and hashed
/// - `num_classes` - Optional number of classes. If None, it will be inferred from the number of unique labels
///
/// # Returns
///
/// - `Result<(Array2<f64>, AHashMap<T, usize>), Error>` - A tuple containing the one-hot encoded matrix and the mapping from original labels to class indices
///
/// # Errors
///
/// - [`Error::InvalidParameter`] - If `num_classes` is smaller than the number of unique labels
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
pub fn to_categorical_with_mapping<T>(
    labels: &[T],
    num_classes: Option<usize>,
) -> Result<(Array2<f64>, AHashMap<T, usize>), Error>
where
    T: Clone + Eq + std::hash::Hash,
{
    let n_samples = labels.len();

    // Create mapping from unique labels to indices using AHashMap
    let mut label_to_index = AHashMap::new();

    for label in labels.iter() {
        let len = label_to_index.len();
        label_to_index.entry(label.clone()).or_insert(len);
    }

    // Determine number of classes
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

    // Create one-hot encoded matrix
    let mut categorical = Array2::<f64>::zeros((n_samples, n_classes));

    for (i, label) in labels.iter().enumerate() {
        let class_index = label_to_index[label];
        categorical[[i, class_index]] = 1.0;
    }

    Ok((categorical, label_to_index))
}

/// Converts categorical (one-hot encoded) format to sparse categorical labels
///
/// This function performs the inverse operation of `to_categorical`, converting
/// a one-hot encoded matrix back to integer labels (sparse categorical format).
///
/// # Parameters
///
/// - `categorical` - A 2D one-hot encoded matrix where each row represents a sample
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
///
/// # Note
///
/// This function finds the class with the highest probability for each sample,
/// making it suitable for converting model predictions back to class labels.
pub fn to_sparse_categorical<S>(categorical: &ArrayBase<S, Ix2>) -> Result<Array1<i32>, Error>
where
    S: Data<Elem = f64>,
{
    // Reject non-finite values up front so the per-row argmax comparison is total.
    super::validation::check_finite(categorical)?;

    let labels = categorical
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0)
        })
        .collect();

    Ok(labels)
}
