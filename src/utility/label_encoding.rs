use super::*;

/// Converts sparse categorical labels to categorical (one-hot encoded) format
///
/// This function takes a 1D array of integer labels and converts them to a 2D
/// one-hot encoded matrix where each row represents a sample and each column
/// represents a class. The value is 1.0 for the corresponding class and 0.0
/// for all other classes.
///
/// # Parameters
///
/// * `labels` - A 1D array of integer labels (e.g., \[0, 1, 2, 1, 0\])
/// * `num_classes` - Optional number of classes. If None, it will be inferred
///   from the maximum label value + 1
///
/// # Returns
///
/// * `Array2<f64>` - A 2D one-hot encoded matrix of shape (n_samples, n_classes)
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utility::to_categorical;
///
/// let labels = array![0, 1, 2, 1, 0];
/// let categorical = to_categorical(&labels, None);
/// // Result: [[1.0, 0.0, 0.0],
/// //          [0.0, 1.0, 0.0],
/// //          [0.0, 0.0, 1.0],
/// //          [0.0, 1.0, 0.0],
/// //          [1.0, 0.0, 0.0]]
/// ```
///
/// # Panics
///
/// Panics if any label is negative or if the specified num_classes is smaller
/// than the maximum label + 1.
pub fn to_categorical(labels: &Array1<i32>, num_classes: Option<usize>) -> Array2<f64> {
    let n_samples = labels.len();

    // Check for negative labels
    for &label in labels.iter() {
        if label < 0 {
            panic!("Labels must be non-negative, found: {}", label);
        }
    }

    // Determine number of classes
    let max_label = *labels.iter().max().unwrap_or(&0) as usize;
    let n_classes = match num_classes {
        Some(n) => {
            if n <= max_label {
                panic!(
                    "num_classes ({}) must be greater than the maximum label ({})",
                    n, max_label
                );
            }
            n
        }
        None => max_label + 1,
    };

    // Create one-hot encoded matrix
    let mut categorical = Array2::<f64>::zeros((n_samples, n_classes));

    for (i, &label) in labels.iter().enumerate() {
        categorical[[i, label as usize]] = 1.0;
    }

    categorical
}

/// Converts sparse categorical labels to categorical format with custom mapping
///
/// This function is useful when you have non-consecutive integer labels or
/// string labels that need to be mapped to consecutive integers first.
///
/// # Parameters
///
/// * `labels` - A slice of labels that can be compared and hashed
/// * `num_classes` - Optional number of classes. If None, it will be inferred
///   from the number of unique labels
///
/// # Returns
///
/// * `(Array2<f64>, AHashMap<T, usize>)` - A tuple containing:
///   - One-hot encoded matrix of shape (n_samples, n_classes)
///   - Mapping from original labels to class indices using AHashMap
///
/// # Examples
///
/// ```rust
/// use rustyml::utility::to_categorical_with_mapping;
///
/// let labels = vec!["cat", "dog", "bird", "dog", "cat"];
/// let (categorical, mapping) = to_categorical_with_mapping(&labels, None);
/// // categorical: one-hot encoded matrix
/// // mapping: {"cat": 0, "dog": 1, "bird": 2} (order may vary)
/// ```
pub fn to_categorical_with_mapping<T>(
    labels: &[T],
    num_classes: Option<usize>,
) -> (Array2<f64>, AHashMap<T, usize>)
where
    T: Clone + Eq + std::hash::Hash,
{
    let n_samples = labels.len();

    // Create mapping from unique labels to indices using AHashMap
    let mut label_to_index = AHashMap::new();
    let mut index = 0;

    for label in labels.iter() {
        if !label_to_index.contains_key(label) {
            label_to_index.insert(label.clone(), index);
            index += 1;
        }
    }

    // Determine number of classes
    let unique_classes = label_to_index.len();
    let n_classes = match num_classes {
        Some(n) => {
            if n < unique_classes {
                panic!(
                    "num_classes ({}) must be at least the number of unique labels ({})",
                    n, unique_classes
                );
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

    (categorical, label_to_index)
}

/// Converts categorical (one-hot encoded) format to sparse categorical labels
///
/// This function performs the inverse operation of `to_categorical`, converting
/// a one-hot encoded matrix back to integer labels (sparse categorical format).
///
/// # Parameters
///
/// * `categorical` - A 2D one-hot encoded matrix where each row represents a sample
///
/// # Returns
///
/// * `Array1<i32>` - A 1D array of integer labels in sparse categorical format
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use rustyml::utility::to_sparse_categorical;
///
/// let categorical = array![[1.0, 0.0, 0.0],
///                         [0.0, 1.0, 0.0],
///                         [0.0, 0.0, 1.0]];
/// let sparse_labels = to_sparse_categorical(&categorical);
/// // Result: [0, 1, 2]
/// ```
///
/// # Note
///
/// This function finds the class with the highest probability for each sample,
/// making it suitable for converting model predictions back to class labels.
pub fn to_sparse_categorical(categorical: &Array2<f64>) -> Array1<i32> {
    let n_samples = categorical.nrows();
    let mut labels = Array1::<i32>::zeros(n_samples);

    for (i, row) in categorical.rows().into_iter().enumerate() {
        // Find the index of the maximum value in the row
        let mut max_index = 0;
        let mut max_value = row[0];

        for (j, &value) in row.iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = j;
            }
        }

        labels[i] = max_index as i32;
    }

    labels
}
