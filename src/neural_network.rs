//! Neural network primitives: layers, loss functions, optimizers, the sequential
//! model, and the traits that tie them together

use ndarray::ArrayD;

/// N-dimensional array used as a tensor in the neural network
pub type Tensor = ArrayD<f32>;

/// Neural network layer implementations
pub mod layers;
/// Loss function implementations
pub mod losses;
/// Optimization algorithms for neural network training
pub mod optimizers;
/// Sequential model architecture
pub mod sequential;
/// Trait interfaces for neural network models
pub mod traits;
