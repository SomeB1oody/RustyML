pub use ada_grad::AdaGrad;
pub use adam::Adam;
pub use rms_prop::RMSprop;
pub use sgd::SGD;

/// AdaGrad (Adaptive Gradient Algorithm) optimizer
pub mod ada_grad;
/// Adam (Adaptive Moment Estimation) optimizer
pub mod adam;
/// Flat-slice per-parameter update kernels shared by all optimizers
pub mod kernels;
/// RMSprop (Root Mean Square Propagation) optimizer
pub mod rms_prop;
/// SGD (Stochastic Gradient Descent) optimizer
pub mod sgd;
/// Input validation functions for optimizers
mod validation;
