use crate::error::ModelError;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::{SeedableRng, rng, rngs::StdRng, seq::SliceRandom};

/// Splits a dataset into training and test sets
///
/// # Parameters
///
/// - `x` - Feature matrix with shape (n_samples, n_features)
/// - `y` - Target values with shape (n_samples)
/// - `test_size` - Size of the test set, default is 0.3 (30%)
/// - `random_state` - Random seed, default is None
///
/// # Returns
///
/// - `Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), ModelError>` - Returns a tuple `(x_train, x_test, y_train, y_test)` if processing successfully
///
/// # Errors
///
/// - Returns `ModelError::InputValidationError` if the dataset is empty, if `x` and `y` have different lengths, if `test_size` is not between 0 and 1, or if the dataset is too small to split.
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::train_test_split::train_test_split;
///
/// let x = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
/// let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
/// let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.4), Some(42)).unwrap();
/// ```
pub fn train_test_split(
    x: Array2<f64>,
    y: Array1<f64>,
    test_size: Option<f64>,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), ModelError> {
    let n_samples = x.nrows();

    // Early return for edge cases
    if n_samples == 0 {
        return Err(ModelError::InputValidationError(
            "Cannot split empty dataset".to_string(),
        ));
    }

    // Ensure x and y have the same number of samples
    if n_samples != y.len() {
        return Err(ModelError::InputValidationError(format!(
            "x and y must have the same number of samples, x rows: {}, y length: {}",
            n_samples,
            y.len()
        )));
    }

    // Set test size, default is 0.3
    let test_size = test_size.unwrap_or(0.3);
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(ModelError::InputValidationError(format!(
            "test_size must be between 0 and 1 (exclusive), got {}",
            test_size
        )));
    }

    // Calculate the number of test samples
    // For small datasets, ensure at least one sample in both train and test sets
    let n_test = if n_samples == 1 {
        return Err(ModelError::InputValidationError(
            "Cannot split a dataset with only 1 sample into train and test sets".to_string(),
        ));
    } else if n_samples == 2 {
        1 // Special case: with 2 samples, always put 1 in test set regardless of test_size
    } else {
        // For larger datasets, use rounding to get closest to the expected proportion
        let calculated = (n_samples as f64 * test_size).round() as usize;
        calculated.max(1).min(n_samples - 1) // Ensure both train and test have at least 1 sample
    };

    // Create random indices
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Shuffle indices based on random state
    match random_state {
        Some(seed) => {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        None => {
            let mut rng = rng();
            indices.shuffle(&mut rng);
        }
    }

    // Split indices into train and test sets
    let (test_indices, train_indices) = indices.split_at(n_test);

    // Create train and test datasets using ndarray's select method for better performance
    let x_train = x.select(Axis(0), train_indices);
    let x_test = x.select(Axis(0), test_indices);
    let y_train = y.select(Axis(0), train_indices);
    let y_test = y.select(Axis(0), test_indices);

    Ok((x_train, x_test, y_train, y_test))
}
