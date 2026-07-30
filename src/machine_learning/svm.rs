//! Support vector machines
//!
//! Groups the kernel-based [`SVC`] (Sequential Minimal Optimization) and the linear,
//! SGD-trained [`LinearSVC`]

/// Linear Support Vector Classifier (LinearSVC)
pub mod linear_svc;
/// Support Vector Classifier (SVC)
pub mod svc;

pub use linear_svc::{LinearSVC, Loss};
pub use svc::SVC;
