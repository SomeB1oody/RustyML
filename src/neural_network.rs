//! Neural network primitives: layers, loss functions, optimizers, the sequential
//! model, and the traits that tie them together

use ndarray::ArrayD;

/// Type alias for n-dimensional arrays used as tensors in the neural network
pub type Tensor = ArrayD<f32>;

/// Module that contains neural network layer implementations
pub mod layers;
/// Module that contains loss function implementations
pub mod losses;
/// Crate-internal rayon-parallel matrix multiply shared by the layers
pub(crate) mod matmul;
/// Shared parallel/serial gate thresholds for the elementwise kernel classes
pub(crate) mod parallel_gates;
/// Module that contains optimization algorithms for neural network training
pub mod optimizers;
/// Module that contains implementations for sequential model architecture
pub mod sequential;
/// Module that defines the trait interfaces for neural network models
pub mod traits;
