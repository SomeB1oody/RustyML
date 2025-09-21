use super::raw_data::load_diabetes_raw_data;
use ndarray::prelude::*;

/// Loads the diabetes dataset
///
/// # Returns
///
/// A tuple containing:
/// - `Array1<&'static str>`: The headers of the dataset
/// - `Array2<f64>`: The feature matrix where each row is a sample and each column is a feature
/// - `Array1<f64>`: Class variable (0 or 1)
///
/// # Examples
///
/// ```
/// use rustyml::dataset::diabetes::load_diabetes;
///
/// let (headers, features, classes) = load_diabetes();
/// assert_eq!(headers.len(), 9);
/// assert_eq!(features.shape(), &[768, 8]);
/// assert_eq!(classes.len(), 768);
/// ```
pub fn load_diabetes() -> (Array1<&'static str>, Array2<f64>, Array1<f64>) {
    let (diabetes_data_headers_raw, diabetes_data_raw) = load_diabetes_raw_data();

    let headers = diabetes_data_headers_raw
        .trim()
        .lines()
        .collect::<Vec<&str>>();
    let mut features = Vec::with_capacity(768 * 8);
    let mut labels = Vec::with_capacity(768);

    for line in diabetes_data_raw.trim().lines() {
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
