//! Manifold-learning estimators for nonlinear dimensionality reduction
//!
//! [`TSNE`] implements t-distributed Stochastic Neighbor Embedding (t-SNE) for visualizing
//! high-dimensional data. Unlike the [`decomposition`](crate::machine_learning::decomposition)
//! estimators, t-SNE learns no out-of-sample projection. It only embeds the data it is given.
//! It implements [`FitTransform`](crate::traits::FitTransform) but not
//! [`Transform`](crate::traits::Transform).

/// t-SNE algorithm for dimensionality reduction
pub mod t_sne;

pub use t_sne::{Init, TSNE, TSNEMethod};
