/// Binary Cross Entropy loss function for binary classification
pub mod binary_cross_entropy;
/// Categorical Cross Entropy loss function for multi-class classification
pub mod categorical_cross_entropy;
/// Mean Absolute Error loss function
pub mod mean_absolute_error;
/// Mean Squared Error loss function
pub mod mean_squared_error;
/// Sparse Categorical Cross Entropy loss function for multi-class classification
pub mod sparse_categorical_cross_entropy;

pub use binary_cross_entropy::*;
pub use categorical_cross_entropy::*;
pub use mean_absolute_error::*;
pub use mean_squared_error::*;
pub use sparse_categorical_cross_entropy::*;
