//! Matrix-decomposition estimators for linear and kernel dimensionality reduction
//!
//! Houses the unsupervised "decomposition" family, mirroring scikit-learn's
//! `sklearn.decomposition`: [`PCA`] for linear dimensionality reduction and
//! [`KernelPCA`] for its nonlinear, kernelized counterpart. Both learn their
//! components from a feature matrix at `fit` time and project new data through
//! `transform`. Both implement the shared [`Fit`](crate::traits::Fit),
//! [`Transform`](crate::traits::Transform), and
//! [`FitTransform`](crate::traits::FitTransform) traits

pub use crate::machine_learning::{Gamma, KernelType};

/// Kernel Principal Component Analysis
pub mod kernel_pca;
/// Principal Component Analysis (PCA)
pub mod pca;

pub use kernel_pca::{EigenSolver, KernelPCA};
pub use pca::{PCA, SVDSolver};
