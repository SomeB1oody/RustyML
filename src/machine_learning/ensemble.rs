//! Ensemble models
//!
//! Groups [`IsolationForest`] for unsupervised anomaly detection and its underlying
//! [`IsolationTree`] node type

/// Isolation Forest for anomaly detection
pub mod isolation_forest;

pub use isolation_forest::{IsolationForest, IsolationTree};
