//! Clustering estimators
//!
//! Groups the unsupervised clustering models: density-based [`DBSCAN`], centroid-based
//! [`KMeans`], and mode-seeking [`MeanShift`] (with its [`estimate_bandwidth`] helper)

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN) clustering
pub mod dbscan;
/// K-means clustering for unsupervised learning
pub mod kmeans;
/// Mean Shift clustering
pub mod mean_shift;

pub use dbscan::DBSCAN;
pub use kmeans::KMeans;
pub use mean_shift::{MeanShift, estimate_bandwidth};
