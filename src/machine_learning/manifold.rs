//! Manifold-learning estimators for nonlinear dimensionality reduction
//!
//! [`TSNE`], t-Distributed Stochastic Neighbor Embedding for visualizing high-dimensional
//! data. Unlike the [`decomposition`](crate::machine_learning::decomposition) estimators,
//! t-SNE learns no out-of-sample projection - it only embeds the data it is given - so it
//! implements [`FitTransform`](crate::machine_learning::traits::FitTransform) but not
//! [`Transform`](crate::machine_learning::traits::Transform)

/// t-SNE algorithm for dimensionality reduction
pub mod t_sne;

pub use t_sne::{Init, TSNE, TSNEMethod};
