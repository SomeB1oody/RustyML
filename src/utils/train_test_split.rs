//! Train/test split utility
//!
//! Provides the [`TrainTestSplit`] type alias and the [`train_test_split()`] function for
//! partitioning a feature matrix and label vector into training and test subsets

use crate::error::Error;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::seq::SliceRandom;

/// The four arrays produced by [`train_test_split`], in order:
/// `(x_train, x_test, y_train, y_test)`
pub type TrainTestSplit<A> = (Array2<f64>, Array2<f64>, Array1<A>, Array1<A>);

/// Splits a dataset into training and test sets
///
/// # Parameters
///
/// - `x` - Feature matrix with shape (n_samples, n_features)
/// - `y` - Target values with shape (n_samples); the label type `A` is generic, so integer,
///   float, or string labels all work (only `Clone` is required)
/// - `test_size` - Size of the test set, default is 0.3 (30%)
/// - `random_state` - Random seed, default is None
///
/// # Type Parameters
///
/// - `A` - The label element type; must be `Clone` so rows can be gathered into the output arrays
///
/// # Returns
///
/// - `Result<TrainTestSplit<A>, Error>` - `(x_train, x_test, y_train, y_test)` on success
///
/// # Errors
///
/// - [`Error::EmptyInput`] if the dataset is empty
/// - [`Error::DimensionMismatch`] if `x` and `y` have different lengths
/// - [`Error::InvalidParameter`] if `test_size` is not between 0 and 1
/// - [`Error::InvalidInput`] if the dataset is too small to split
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utils::train_test_split::train_test_split;
///
/// let x = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
/// // Integer class labels work just as well as floats.
/// let y = Array1::from(vec![0, 1, 0, 1, 0]);
/// let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.4), Some(42)).unwrap();
/// ```
pub fn train_test_split<A>(
    x: Array2<f64>,
    y: Array1<A>,
    test_size: Option<f64>,
    random_state: Option<u64>,
) -> Result<TrainTestSplit<A>, Error>
where
    A: Clone,
{
    let n_samples = x.nrows();

    if n_samples == 0 {
        return Err(Error::empty_input("dataset"));
    }

    if n_samples != y.len() {
        return Err(Error::dimension_mismatch(n_samples, y.len()));
    }

    let test_size = test_size.unwrap_or(0.3);
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(Error::invalid_parameter(
            "test_size",
            format!(
                "test_size must be between 0 and 1 (exclusive), got {}",
                test_size
            ),
        ));
    }

    // Clamp so both train and test always keep at least one sample
    let n_test = if n_samples == 1 {
        return Err(Error::invalid_input(
            "Cannot split a dataset with only 1 sample into train and test sets",
        ));
    } else if n_samples == 2 {
        1 // with 2 samples, always put 1 in test regardless of test_size
    } else {
        let calculated = (n_samples as f64 * test_size).round() as usize;
        calculated.max(1).min(n_samples - 1)
    };

    let mut indices: Vec<usize> = (0..n_samples).collect();

    let mut rng = crate::random::make_rng(random_state);
    indices.shuffle(&mut rng);

    let (test_indices, train_indices) = indices.split_at(n_test);

    // select gathers rows by index in one pass
    let x_train = x.select(Axis(0), train_indices);
    let x_test = x.select(Axis(0), test_indices);
    let y_train = y.select(Axis(0), train_indices);
    let y_test = y.select(Axis(0), test_indices);

    Ok((x_train, x_test, y_train, y_test))
}
