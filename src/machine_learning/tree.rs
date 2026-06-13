//! Decision tree models
//!
//! Groups the [`DecisionTree`] estimator (ID3, C4.5, and CART) with its node types
//! ([`Node`], [`NodeType`]), the [`Algorithm`] selector, and the [`DecisionTreeParams`]
//! hyperparameters

/// Decision tree for classification and regression
pub mod decision_tree;

pub use decision_tree::{Algorithm, DecisionTree, DecisionTreeParams, Node, NodeType};
