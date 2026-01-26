use crate::error::ModelError;
use crate::utility::*;
use approx::assert_abs_diff_eq;
use ndarray::prelude::*;
use std::error::Error;

mod kernel_pca_test;
mod label_encoding_test;
mod linear_discriminant_analysis_test;
mod normalize_test;
mod principal_component_analysis_test;
mod standardize_test;
mod t_sne_test;
mod train_test_split_test;
