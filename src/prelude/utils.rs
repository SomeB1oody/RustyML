//! Prelude re-exports for the `utils` module: preprocessing and dataset splitting

pub use crate::utils::label_encoding::{
    to_categorical, to_categorical_with_mapping, to_sparse_categorical,
};
pub use crate::utils::normalize::normalize;
pub use crate::utils::standardize::standardize;
pub use crate::utils::train_test_split::{train_test_split, train_test_split_stratified};
