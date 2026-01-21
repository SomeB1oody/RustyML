use super::raw_data::iris_raw::*;
use ndarray::prelude::*;
use std::sync::OnceLock;

// Use `OnceLock` for thread-safe delayed initialization
static IRIS_DATA: OnceLock<(Array1<&'static str>, Array2<f64>, Array1<&'static str>)> =
    OnceLock::new();

/// Internal function to load and process the raw iris dataset.
///
/// This function loads the raw iris dataset, parses the CSV-like format,
/// and converts it into structured ndarray arrays. It handles the parsing
/// of headers and data rows, extracting features and labels from the dataset.
///
/// # Returns
///
/// - `Array1<&'static str>`: Array of column headers from the dataset
/// - `Array2<f64>`: Feature matrix with shape (150, 4) where each row represents a flower sample and each column represents a feature
/// - `Array1<&'static str>`: Target labels array with shape (150,) containing species classifications (Iris-setosa, Iris-versicolor, Iris-virginica)
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (150 samples, 5 columns total)
/// - Memory allocation fails during array creation
fn load_iris_internal() -> (Array1<&'static str>, Array2<f64>, Array1<&'static str>) {
    let iris_data_raw = load_iris_raw_data();

    let lines: Vec<&str> = iris_data_raw.trim().lines().collect();
    let headers_line = lines[0];
    let headers = headers_line.split(',').collect::<Vec<&str>>();

    let mut features = Vec::with_capacity(150 * 4);
    let mut labels = Vec::with_capacity(150);

    for line in &lines[1..] {
        let cols: Vec<&str> = line.split(',').collect();

        for i in 0..4 {
            features.push(cols[i].parse::<f64>().unwrap());
        }

        labels.push(cols[4]);
    }

    let headers_array = Array1::from_vec(headers);
    let features_array = Array2::from_shape_vec((150, 4), features).unwrap();
    let labels_array = Array1::from_vec(labels);

    (headers_array, features_array, labels_array)
}

/// Loads the Iris dataset with memoization
///
/// The Iris dataset contains measurements of 150 iris flowers from three different species:
/// - Iris-setosa
/// - Iris-versicolor
/// - Iris-virginica
///
/// This function uses memoization to ensure the dataset is loaded only once and returns
/// static references to the cached data for optimal performance.
///
/// # Returns
///
/// - `&'static Array1<&'static str>`: Static reference to the headers of the dataset
/// - `&'static Array2<f64>`: Static reference to a 2D array of shape (150, 4) containing the feature measurements:
///     - sepal length in cm
///     - sepal width in cm
///     - petal length in cm
///     - petal width in cm
/// - `&'static Array1<&'static str>`: Static reference to a 1D array of length 150 containing the species labels
///
/// # Example
/// ```rust
/// use rustyml::dataset::iris::load_iris;
///
/// let (headers, features, labels) = load_iris();
/// assert_eq!(headers.len(), 5);
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
/// ```
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (150 samples, 5 columns total)
/// - Memory allocation fails during array creation
pub fn load_iris() -> (
    &'static Array1<&'static str>,
    &'static Array2<f64>,
    &'static Array1<&'static str>,
) {
    let (headers, features, labels) = IRIS_DATA.get_or_init(load_iris_internal);
    (headers, features, labels)
}

/// Loads the Iris dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer `load_iris()` which returns references.
///
/// The Iris dataset contains measurements of 150 iris flowers from three different species:
/// - Iris-setosa
/// - Iris-versicolor
/// - Iris-virginica
///
/// # Returns
///
/// - `Array1<&'static str>`: Owned array of column headers from the dataset, containing 5 feature names plus the target label name
/// - `Array2<f64>`: Owned feature matrix with shape (150, 4) where each row represents a flower sample and each column represents a feature (sepal length, sepal width, petal length, petal width)
/// - `Array1<&'static str>`: Owned target labels array with shape (150,) containing species classifications (Iris-setosa, Iris-versicolor, Iris-virginica)
///
/// # Performance Notes
///
/// This function creates owned copies by cloning the static data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use `load_iris()` instead for better performance.
///
/// # Examples
/// ```rust
/// use rustyml::dataset::iris::load_iris_owned;
///
/// let (mut headers, mut features, mut labels) = load_iris_owned();
///
/// // You can now modify the data since these are owned copies
/// assert_eq!(headers.len(), 5);
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 5.5;
/// labels[0] = "Modified-setosa";
/// ```
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (150 samples, 5 columns total)
/// - Memory allocation fails during array creation
pub fn load_iris_owned() -> (Array1<&'static str>, Array2<f64>, Array1<&'static str>) {
    let (headers, features, labels) = load_iris();
    (headers.clone(), features.clone(), labels.clone())
}
