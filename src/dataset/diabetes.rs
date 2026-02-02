use super::raw_data::diabetes_raw::load_diabetes_raw_data;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

// Use `OnceLock` for thread-safe delayed initialization
static DIABETES_DATA: OnceLock<(Array1<&'static str>, Array2<f64>, Array1<f64>)> = OnceLock::new();

/// Internal function to load and process the raw diabetes dataset.
///
/// This function loads the raw diabetes dataset, parses the CSV-like format,
/// and converts it into structured ndarray arrays. It handles the parsing
/// of headers and data rows, extracting features and labels from the dataset.
///
/// # Returns
///
/// - `&'static Array1<&'static str>` - Static reference to the headers of the dataset.
/// - `&'static Array2<f64>` - Static reference to the feature matrix (768x8).
/// - `&'static Array1<f64>` - Static reference to the binary labels (0 or 1).
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (768 samples, 9 columns total)
/// - Memory allocation fails during array creation
fn load_diabetes_internal() -> (Array1<&'static str>, Array2<f64>, Array1<f64>) {
    let raw_data = load_diabetes_raw_data();
    let mut lines = raw_data.trim().lines();

    // First line contains headers
    let header_line = lines.next().unwrap();
    let headers = header_line.split(',').collect::<Vec<&str>>();

    let mut features = Vec::with_capacity(768 * 8);
    let mut labels = Vec::with_capacity(768);

    // Process remaining lines as data
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split(',').collect();

        for i in 0..8 {
            features.push(cols[i].parse::<f64>().unwrap());
        }

        labels.push(cols[8].parse::<f64>().unwrap());
    }

    let headers_array = Array1::from_vec(headers);
    let features_array = Array2::from_shape_vec((768, 8), features).unwrap();
    let labels_array = Array1::from_vec(labels);

    (headers_array, features_array, labels_array)
}

/// Loads the diabetes dataset
///
/// # Returns
///
/// - `&'static Array1<&'static str>`: Static reference to the headers of the dataset
/// - `&'static Array2<f64>`: Static reference to the feature matrix where each row is a sample and each column is a feature
/// - `&'static Array1<f64>`: Static reference to class variable (0 or 1)
///
/// # Examples
/// ```rust
/// use rustyml::dataset::diabetes::load_diabetes;
///
/// let (headers, features, classes) = load_diabetes();
/// assert_eq!(headers.len(), 9);
/// assert_eq!(features.shape(), &[768, 8]);
/// assert_eq!(classes.len(), 768);
/// ```
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (768 samples, 9 columns total)
/// - Memory allocation fails during array creation
pub fn load_diabetes() -> (
    &'static Array1<&'static str>,
    &'static Array2<f64>,
    &'static Array1<f64>,
) {
    let (headers, features, labels) = DIABETES_DATA.get_or_init(load_diabetes_internal);
    (headers, features, labels)
}

/// Loads the diabetes dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer `load_diabetes()` which returns references.
///
/// # Returns
///
/// - `Array1<&'static str>`: Owned array of column headers from the dataset, containing 9 feature names plus the target label name
/// - `Array2<f64>`: Owned feature matrix with shape (768, 8) where each row represents a patient sample and each column represents a feature (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age)
/// - `Array1<f64>`: Owned target labels array with shape (768,) containing binary classification outcomes (0.0 for non-diabetic, 1.0 for diabetic)
///
/// # Performance Notes
///
/// This function creates owned copies by cloning the static data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use `load_diabetes()` instead for better performance.
///
/// # Examples
/// ```rust
/// use rustyml::dataset::diabetes::load_diabetes_owned;
///
/// let (mut headers, mut features, mut labels) = load_diabetes_owned();
///
/// // You can now modify the data since these are owned copies
/// assert_eq!(headers.len(), 9);
/// assert_eq!(features.shape(), &[768, 8]);
/// assert_eq!(labels.len(), 768);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 10.0;
/// labels[0] = 1.0;
/// ```
///
/// # Panics
///
/// This function will panic if:
/// - The raw data cannot be parsed as valid f64 values
/// - The dataset structure doesn't match the expected format (768 samples, 9 columns total)
/// - Memory allocation fails during array creation
pub fn load_diabetes_owned() -> (Array1<&'static str>, Array2<f64>, Array1<f64>) {
    let (headers, features, labels) = load_diabetes();
    (headers.clone(), features.clone(), labels.clone())
}
