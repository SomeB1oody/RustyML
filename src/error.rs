#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
use std::fs::File;
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
use std::io::BufReader;

/// Error types that can occur during model operations
///
/// # Variants
///
/// - `NotFitted` - Indicates that the model has not been fitted yet
/// - `InputValidationError` - indicates the input data provided does not meet the expected format, type, or validation rules
/// - `TreeError` - indicates that there is something wrong with the tree
/// - `ProcessingError` - indicates that there is something wrong while processing
#[cfg(any(
    feature = "machine_learning",
    feature = "math",
    feature = "neural_network",
    feature = "utility"
))]
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    NotFitted,
    InputValidationError(String),
    TreeError(&'static str),
    ProcessingError(String),
}

#[cfg(any(
    feature = "machine_learning",
    feature = "math",
    feature = "neural_network",
    feature = "utility"
))]
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NotFitted => {
                write!(
                    f,
                    "Model has not been fitted. Certain methods require the model to be fitted before use."
                )
            }
            ModelError::InputValidationError(msg) => write!(f, "Input validation error: {}", msg),
            ModelError::TreeError(msg) => write!(f, "Tree structure error: {}", msg),
            ModelError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

#[cfg(any(
    feature = "machine_learning",
    feature = "math",
    feature = "neural_network",
    feature = "utility"
))]
impl std::error::Error for ModelError {}

/// Input/Output error types that can occur during model serialization and file operations
///
/// # Variants
///
/// - `StdIoError` - Wraps standard I/O errors from file system operations (reading, writing, file access)
/// - `JsonError` - Wraps JSON serialization/deserialization errors when working with JSON data formats
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
#[derive(Debug)]
pub enum IoError {
    StdIoError(std::io::Error),
    JsonError(serde_json::Error),
}

#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
impl IoError {
    pub fn load_in_buf_reader(path: &str) -> Result<BufReader<File>, IoError> {
        let file = File::open(path).map_err(IoError::StdIoError)?;
        Ok(BufReader::new(file))
    }
}

#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::StdIoError(e) => write!(f, "IO error: {}", e),
            IoError::JsonError(e) => write!(f, "JSON error: {}", e),
        }
    }
}

#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
impl std::error::Error for IoError {}
