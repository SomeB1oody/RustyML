/// Module that contains activation function implementations
pub mod activation;
/// Module that contains neural network layer implementations
pub mod layer;
/// Module that contains loss function implementations
pub mod loss_function;
/// Module that contains optimization algorithms for neural network training
pub mod optimizer;
/// Module that contains implementations for sequential model architecture
pub mod sequential;
/// Module containing trait definitions for machine neural network model interfaces
pub mod traits;

pub use activation::*;
pub use layer::*;
pub use loss_function::*;
pub use optimizer::*;
pub use sequential::*;
pub use traits::*;

use crate::error::ModelError;
use ndarray::prelude::*;

/// Type alias for n-dimensional arrays used as tensors in the neural network
pub type Tensor = ArrayD<f32>;
