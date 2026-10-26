//! Neural network types, layers, losses, optimizers, and model builders.
//!
//! This category exports the `Average` merge layer.
//! With metrics enabled, the prelude root uses `Average` for the metrics enum instead.

pub use crate::neural_network::Ctx;
pub use crate::neural_network::Shape;
pub use crate::neural_network::Tensor;
pub use crate::neural_network::graph::{Graph, GraphBuilder, NodeId};
pub use crate::neural_network::layers::activation::*;
pub use crate::neural_network::layers::border::*;
pub use crate::neural_network::layers::convolution::*;
pub use crate::neural_network::layers::pooling::*;
pub use crate::neural_network::layers::recurrent::*;
pub use crate::neural_network::layers::regularization::*;
pub use crate::neural_network::layers::upsampling::*;
pub use crate::neural_network::layers::{
    Add, Average, Concatenate, Dense, Embedding, Flatten, Identity, Maximum, Minimum, Multiply,
    Permute, RepeatVector, Rescaling, Reshape, Reverse, Subtract,
};
pub use crate::neural_network::losses::*;
pub use crate::neural_network::optimizers::*;
pub use crate::neural_network::sequential::{History, Sequential, SequentialBuilder};
