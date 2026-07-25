//! Ensemble models
//!
//! Groups [`IsolationForest`] for unsupervised anomaly detection, its underlying
//! [`IsolationTree`] node type, and the [`Contamination`] threshold rule

/// Isolation Forest for anomaly detection
pub mod isolation_forest;

pub use isolation_forest::{Contamination, IsolationForest, IsolationTree};
