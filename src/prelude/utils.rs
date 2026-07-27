//! Prelude re-exports for the `utils` module: preprocessing, dataset splitting, and the
//! shared estimator traits the preprocessing transformers implement

pub use crate::traits::{Fit, FitTransform, Predict, Transform};
pub use crate::utils::label_encoding::{
    to_categorical, to_categorical_with_mapping, to_sparse_categorical,
};
pub use crate::utils::normalize::{normalize, NormalizationAxis, NormalizationOrder};
pub use crate::utils::scaler::{
    MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler,
};
pub use crate::utils::standardize::{standardize, StandardizationAxis};
pub use crate::utils::train_test_split::{train_test_split, train_test_split_stratified};
