//! Discriminant analysis
//!
//! Groups [`LDA`], used for both classification and supervised dimensionality
//! reduction, with its [`Solver`] and [`Shrinkage`] configuration enums

/// Linear Discriminant Analysis for classification and supervised dimensionality reduction
pub mod lda;

pub use lda::{LDA, Shrinkage, Solver};
