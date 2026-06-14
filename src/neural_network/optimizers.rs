//! Optimizers for neural network training
//!
//! Re-exports the optimizer types and declares the submodules holding each algorithm
//! (SGD, AdaGrad, RMSprop, Adam), the shared flat-slice update kernels, and parameter validation

pub use ada_grad::AdaGrad;
pub use adam::Adam;
pub use adam_w::AdamW;
pub use rms_prop::RMSprop;
pub use sgd::SGD;

/// AdaGrad (Adaptive Gradient Algorithm) optimizer
pub mod ada_grad;
/// Adam (Adaptive Moment Estimation) optimizer with classic coupled L2 weight decay
pub mod adam;
/// Shared state and update machinery for the Adam-family optimizers (`Adam`, `AdamW`)
mod adam_core;
/// AdamW optimizer: Adam with decoupled weight decay
pub mod adam_w;
/// Flat-slice per-parameter update kernels shared by all optimizers
pub mod kernels;
/// RMSprop (Root Mean Square Propagation) optimizer
pub mod rms_prop;
/// SGD (Stochastic Gradient Descent) optimizer
pub mod sgd;
/// Input validation functions for optimizers
mod validation;
