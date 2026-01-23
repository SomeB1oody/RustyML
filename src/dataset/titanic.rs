use super::raw_data::titanic_raw::load_titanic_raw_data;
use ndarray::prelude::*;
use std::sync::OnceLock;

// Use `OnceLock` for thread-safe delayed initialization
static TITANIC_DATA: OnceLock<(
    Array1<&'static str>,
    Array1<&'static str>,
    Array2<String>,
    Array2<f64>,
)> = OnceLock::new();

/// Internal function to load and process the raw Titanic dataset.
///
/// This function loads the raw Titanic dataset, parses the CSV format,
/// and converts it into structured ndarray arrays. It separates string features
/// from numeric features and converts gender to binary encoding (female=0, male=1).
///
/// # Returns
///
/// - `Array1<&'static str>`: Array of string feature headers
/// - `Array1<&'static str>`: Array of numeric feature headers
/// - `Array2<String>`: String features matrix (Name, Ticket, Cabin, Embarked)
/// - `Array2<f64>`: Numeric features matrix (PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare)
///
/// # Notes
///
/// - Sex is converted to numeric: female=0.0, male=1.0
/// - Missing values in numeric columns are handled as 0.0
/// - Missing string values are handled as empty strings
///
/// # Panics
///
/// This function will panic if:
/// - The dataset structure doesn't match the expected format
/// - Memory allocation fails during array creation
fn load_titanic_internal() -> (
    Array1<&'static str>,
    Array1<&'static str>,
    Array2<String>,
    Array2<f64>,
) {
    let raw_data = load_titanic_raw_data();

    // Split into lines and extract headers from first line
    let lines: Vec<&str> = raw_data.trim().lines().collect();
    if lines.is_empty() {
        panic!("No data found");
    }

    // Parse headers from first line
    let all_headers: Vec<&str> = lines[0].trim().split(',').collect();

    // First pass: collect all data rows and determine column types
    let mut all_rows = Vec::new();

    // Skip the header line, process data lines
    for line in lines.iter().skip(1) {
        if line.trim().is_empty() {
            continue;
        }

        // Parse CSV with quoted strings
        let mut cols = Vec::new();
        let mut in_quotes = false;
        let mut current_col = String::new();
        let mut chars = line.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '"' => {
                    in_quotes = !in_quotes;
                }
                ',' if !in_quotes => {
                    cols.push(current_col.trim().to_string());
                    current_col.clear();
                }
                _ => {
                    current_col.push(ch);
                }
            }
        }
        cols.push(current_col.trim().to_string());
        all_rows.push(cols);
    }

    if all_rows.is_empty() {
        panic!("No data rows found");
    }

    let num_cols = all_headers.len();
    let mut is_numeric = vec![true; num_cols];

    // Determine column types by examining data
    for row in &all_rows {
        for (col_idx, value) in row.iter().enumerate() {
            if col_idx >= num_cols {
                continue;
            }

            // Skip empty values for type detection
            if value.is_empty() {
                continue;
            }

            // Special handling for Sex column - treat as numeric since we'll convert it
            if all_headers[col_idx].trim() == "Sex" && (value == "male" || value == "female") {
                continue; // Keep as numeric
            }

            // Try to parse as number
            if value.parse::<f64>().is_err() {
                is_numeric[col_idx] = false;
            }
        }
    }

    // Separate headers and indices
    let mut string_headers = Vec::new();
    let mut numeric_headers = Vec::new();
    let mut string_indices = Vec::new();
    let mut numeric_indices = Vec::new();

    for (i, &header) in all_headers.iter().enumerate() {
        if is_numeric[i] {
            numeric_headers.push(header.trim());
            numeric_indices.push(i);
        } else {
            string_headers.push(header.trim());
            string_indices.push(i);
        }
    }

    // Second pass: extract features based on determined types
    let mut string_features = Vec::new();
    let mut numeric_features = Vec::new();
    let row_count = all_rows.len();

    for row in &all_rows {
        // Extract string features
        for &idx in &string_indices {
            if idx < row.len() {
                string_features.push(row[idx].clone());
            } else {
                string_features.push(String::new());
            }
        }

        // Extract numeric features
        for &idx in &numeric_indices {
            if idx < row.len() {
                let value = if all_headers[idx].trim() == "Sex" {
                    // Convert sex to numeric: female=0, male=1
                    match row[idx].as_str() {
                        "female" => 0.0,
                        "male" => 1.0,
                        _ => 0.0, // Default to female
                    }
                } else {
                    // Parse other numeric values
                    row[idx].parse::<f64>().unwrap_or(0.0)
                };
                numeric_features.push(value);
            } else {
                numeric_features.push(0.0);
            }
        }
    }

    let string_headers_array = Array1::from_vec(string_headers);
    let numeric_headers_array = Array1::from_vec(numeric_headers);
    let string_features_array =
        Array2::from_shape_vec((row_count, string_indices.len()), string_features).unwrap();
    let numeric_features_array =
        Array2::from_shape_vec((row_count, numeric_indices.len()), numeric_features).unwrap();

    (
        string_headers_array,
        numeric_headers_array,
        string_features_array,
        numeric_features_array,
    )
}

/// Loads the Titanic dataset with separate string and numeric features
///
/// This function provides access to the famous Titanic dataset with passengers' information
/// and survival outcomes. The dataset is automatically split into string and numeric features
/// based on data type detection. Sex is converted to binary encoding: female=0, male=1.
///
/// # Notes
///
/// - Sex is converted to numeric: female=0.0, male=1.0
/// - Missing values in numeric columns are handled as 0.0
/// - Missing string values are handled as empty strings
///
/// # Feature Order
///
/// Based on the original Titanic dataset structure, the typical feature order is:
/// - String headers (in original order): Name, Ticket, Cabin, Embarked
/// - Numeric headers (in original order): PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare
///
/// # Returns
///
/// - `&'static Array1<&'static str>`: Static reference to string feature headers
/// - `&'static Array1<&'static str>`: Static reference to numeric feature headers
/// - `&'static Array2<String>`: Static reference to string features matrix
/// - `&'static Array2<f64>`: Static reference to numeric features matrix
///
/// # Examples
/// ```rust
/// use rustyml::dataset::titanic::load_titanic;
///
/// let (string_headers, numeric_headers, string_features, numeric_features) = load_titanic();
///
/// // Access headers - typical expected counts
/// println!("String headers: {:?}", string_headers);  // Should be: Name, Ticket, Cabin, Embarked
/// println!("Numeric headers: {:?}", numeric_headers); // Should be: PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare
///
/// // Check feature matrices shapes
/// println!("String features shape: {:?}", string_features.shape());
/// println!("Numeric features shape: {:?}", numeric_features.shape());
/// ```
///
/// # Panics
///
/// This function will panic if:
/// - The dataset structure doesn't match the expected format
/// - Memory allocation fails during array creation
pub fn load_titanic() -> (
    &'static Array1<&'static str>,
    &'static Array1<&'static str>,
    &'static Array2<String>,
    &'static Array2<f64>,
) {
    let (string_headers, numeric_headers, string_features, numeric_features) =
        TITANIC_DATA.get_or_init(load_titanic_internal);
    (
        string_headers,
        numeric_headers,
        string_features,
        numeric_features,
    )
}

/// Loads the Titanic dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer `load_titanic()` which returns references.
///
/// # Notes
///
/// - Sex is converted to numeric: female=0.0, male=1.0
/// - Missing values in numeric columns are handled as 0.0
/// - Missing string values are handled as empty strings
///
/// # Feature Order
///
/// Based on the original Titanic dataset structure, the typical feature order is:
/// - String headers (in original order): Name, Ticket, Cabin, Embarked
/// - Numeric headers (in original order): PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare
///
/// Note: The actual order may vary if the dataset structure changes, as column types
/// are determined dynamically by analyzing the data values.
///
/// # Returns
///
/// - `Array1<&'static str>`: Owned array of string feature headers
/// - `Array1<&'static str>`: Owned array of numeric feature headers
/// - `Array2<String>`: Owned string features matrix
/// - `Array2<f64>`: Owned numeric features matrix
///
/// # Performance Notes
///
/// This function creates owned copies by cloning the static data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use `load_titanic()` instead for better performance.
///
/// # Examples
/// ```rust
/// use rustyml::dataset::titanic::load_titanic_owned;
///
/// let (mut string_headers, mut numeric_headers, mut string_features, mut numeric_features) = load_titanic_owned();
///
/// // You can now modify the data since these are owned copies
/// println!("String headers: {:?}", string_headers);  // Should be: Name, Ticket, Cabin, Embarked
/// println!("Numeric headers: {:?}", numeric_headers); // Should be: PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare
///
/// // Example: Modify feature values (not possible with references)
/// if numeric_features.nrows() > 0 {
///     numeric_features[[0, 0]] = 999.0;
/// }
/// ```
///
/// # Panics
///
/// This function will panic if:
/// - The dataset structure doesn't match the expected format
/// - Memory allocation fails during array creation
pub fn load_titanic_owned() -> (
    Array1<&'static str>,
    Array1<&'static str>,
    Array2<String>,
    Array2<f64>,
) {
    let (string_headers, numeric_headers, string_features, numeric_features) = load_titanic();
    (
        string_headers.clone(),
        numeric_headers.clone(),
        string_features.clone(),
        numeric_features.clone(),
    )
}
