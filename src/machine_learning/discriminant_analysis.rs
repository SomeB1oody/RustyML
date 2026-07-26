//! Discriminant analysis
//!
//! Groups [`LDA`] for classification and supervised dimensionality reduction,
//! with its [`DiscriminantSolver`] and [`Shrinkage`] configuration enums

/// Linear Discriminant Analysis for classification and supervised dimensionality reduction
pub mod lda;

pub use lda::{DiscriminantSolver, LDA, Shrinkage};
