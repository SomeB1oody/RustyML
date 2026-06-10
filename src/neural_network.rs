use ndarray::ArrayD;

/// Type alias for n-dimensional arrays used as tensors in the neural network
pub type Tensor = ArrayD<f32>;

/// Module that contains neural network layer implementations
pub mod layers;
/// Module that contains loss function implementations
pub mod losses;
/// Module that contains optimization algorithms for neural network training
pub mod optimizers;
/// Module that contains implementations for sequential model architecture
pub mod sequential;
/// Module containing trait definitions for machine neural network model interfaces
pub mod traits;
