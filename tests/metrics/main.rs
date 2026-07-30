//! Integration tests for the `metrics` feature. It has 1 submodule for each metric domain
//! (classification, clustering, regression), plus shared helpers in [`common`].

/// Shared assertion helpers used by the classification, clustering, and regression test modules.
mod common;

/// Tests for confusion matrices, accuracy, ROC/PR curves, log loss, and other classification
/// metrics.
mod classification;
/// Tests for extrinsic and intrinsic clustering metrics, such as ARI, NMI, and silhouette score.
mod clustering;
/// Tests for regression error metrics and variance-explained scores, such as MSE, MAE, and R2.
mod regression;
