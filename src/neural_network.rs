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

pub use activation::*;
pub use layer::*;
pub use loss_function::*;
pub use optimizer::*;
pub use sequential::*;

use crate::ModelError;
use ndarray::ArrayD;

/// Type alias for n-dimensional arrays used as tensors in the neural network
pub type Tensor = ArrayD<f32>;

pub use crate::traits::Layer;
pub use crate::traits::LossFunction;
pub use crate::traits::Optimizer;
