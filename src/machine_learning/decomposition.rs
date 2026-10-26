//! Matrix-decomposition estimators for linear and kernel dimensionality reduction
//!
//! [`PCA`] provides linear dimensionality reduction. [`KernelPCA`] provides kernel dimensionality reduction.
//! Both models fit components from training samples and transform new samples with those components.
//!
//! [`Fit`]: crate::traits::Fit
//! [`Transform`]: crate::traits::Transform
//! [`FitTransform`]: crate::traits::FitTransform

pub use crate::machine_learning::{Gamma, KernelType};

/// Kernel Principal Component Analysis
pub mod kernel_pca;
/// Principal Component Analysis (PCA)
pub mod pca;

pub use kernel_pca::{EigenSolver, KernelPCA};
pub use pca::{PCA, SVDSolver};
