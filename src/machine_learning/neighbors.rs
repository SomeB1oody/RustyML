//! Nearest-neighbor models
//!
//! Groups the [`KNN`] classifier and the [`WeightingStrategy`] enum that controls how
//! neighbor votes are weighted

/// K-Nearest Neighbors (KNN) classification
pub mod knn;

pub use knn::{KNN, WeightingStrategy};
