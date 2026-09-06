//! Prelude re-exports for the neural network module: tensors, shapes, layers, losses,
//! optimizers, and the 2 model builders
//!
//! `Average` is the 1 name that 2 categories of the crate give to an item. This module gives
//! the merge layer that averages its inputs, and the metrics category gives the averaging mode
//! of the classification scores. The root of [`prelude`](crate::prelude) keeps the averaging
//! mode, so a caller that wants the layer through `use rustyml::prelude::*` names it
//! [`layers::Average`](crate::neural_network::layers::Average)

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
