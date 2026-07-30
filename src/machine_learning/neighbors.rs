//! Nearest-neighbor models
//!
//! Groups the [`KNN`] classifier and the [`WeightingStrategy`] enum, which controls the
//! weight of each neighbor's vote

/// K-Nearest Neighbors (KNN) classification
pub mod knn;

pub use knn::{KNN, WeightingStrategy};
