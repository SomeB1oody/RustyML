//! Machine learning models for clustering, classification, regression, dimensionality
//! reduction, and anomaly detection
//!
//! Models are grouped by algorithm family into submodules: [`clustering`](crate::machine_learning::clustering),
//! [`decomposition`](crate::machine_learning::decomposition), [`linear_model`](crate::machine_learning::linear_model),
//! [`manifold`](crate::machine_learning::manifold), [`svm`](crate::machine_learning::svm),
//! [`tree`](crate::machine_learning::tree), [`neighbors`](crate::machine_learning::neighbors),
//! [`discriminant_analysis`](crate::machine_learning::discriminant_analysis), and
//! [`ensemble`](crate::machine_learning::ensemble). Every estimator is also re-exported here,
//! so it is reachable directly as `machine_learning::<Model>`. Supervised and unsupervised
//! estimators implement the shared [`Fit`](crate::machine_learning::traits::Fit) /
//! [`Predict`](crate::machine_learning::traits::Predict) traits; the dimensionality-reduction
//! transformers implement [`Transform`](crate::machine_learning::traits::Transform) /
//! [`FitTransform`](crate::machine_learning::traits::FitTransform)
//!
//! # Supervised learning
//!
//! ## Classification
//! - **LogisticRegression**: binary classification via gradient descent, with L1/L2 regularization
//! - **KNN**: k-nearest neighbors with selectable distance metric (Euclidean/Manhattan/Minkowski) and weighting
//! - **DecisionTree**: classifier supporting ID3, C4.5, and CART, with pruning options
//! - **SVC**: support vector classifier using Sequential Minimal Optimization (SMO) with kernels
//! - **LinearSVC**: linear support vector classifier for large datasets, with hinge loss
//! - **LDA**: linear discriminant analysis for classification and supervised dimensionality reduction
//!
//! ## Regression
//! - **LinearRegression**: simple and multivariate linear regression with L1/L2 regularization
//!
//! # Unsupervised learning
//!
//! ## Clustering
//! - **KMeans**: k-means with k-means++ initialization and parallel assignment
//! - **DBSCAN**: density-based clustering of arbitrary-shaped clusters, with noise detection
//! - **MeanShift**: non-parametric clustering that discovers the number of clusters
//!
//! ## Dimensionality reduction
//! - **PCA**: principal component analysis (linear)
//! - **KernelPCA**: nonlinear reduction via kernels (RBF, Linear, Poly, Sigmoid, Cosine)
//! - **TSNE**: t-distributed stochastic neighbor embedding for visualization
//!
//! ## Anomaly detection
//! - **IsolationForest**: ensemble isolation-based anomaly detector
//!
//! # Shared types
//!
//! - [`DistanceCalculationMetric`](crate::math::DistanceCalculationMetric): Euclidean/Manhattan/Minkowski
//!   dispatcher, re-exported from [`crate::math`]
//! - [`RegularizationType`](crate::machine_learning::types::RegularizationType): L1 / L2 regularization
//! - [`KernelType`](crate::machine_learning::types::KernelType) /
//!   [`Gamma`](crate::machine_learning::types::Gamma): kernel selection and coefficient
//!
//! # Examples
//!
//! ```rust
//! use rustyml::machine_learning::LinearRegression;
//! use ndarray::array;
//!
//! let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6).unwrap();
//! let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
//! let y = array![6.0, 9.0, 12.0];
//! model.fit(&x, &y).unwrap();
//! ```

pub use crate::math::DistanceCalculationMetric;
pub use types::{Gamma, KernelType, RegularizationType};

/// Clustering estimators: DBSCAN, K-means, and Mean Shift
pub mod clustering;
/// Decomposition estimators: PCA and Kernel PCA
pub mod decomposition;
/// Discriminant analysis: Linear Discriminant Analysis (LDA)
pub mod discriminant_analysis;
/// Ensemble models: Isolation Forest
pub mod ensemble;
/// Linear models: linear and logistic regression
pub mod linear_model;
/// Manifold-learning estimators: t-SNE
pub mod manifold;
/// Nearest-neighbor models: K-Nearest Neighbors
pub mod neighbors;
/// Support vector machines: SVC and Linear SVC
pub mod svm;
/// Tree models: decision trees
pub mod tree;

/// Common `Fit` / `Predict` traits implemented by every estimator
pub mod traits;

/// Internal linear-algebra primitives shared across estimator families: dense factorizations (symmetric eigendecomposition, SVD, thin QR) plus iterative top-`k` eigensolvers (power iteration, Lanczos)
pub(crate) mod linalg;
/// Internal shared helpers for parallel/sequential dispatch across models
mod parallel;
/// Internal kd-tree spatial index for fixed-radius and k-nearest-neighbor queries
pub(crate) mod spatial;
/// kernels (`KernelType`), kernel coefficient (`Gamma`), and regularization (`RegularizationType`)
pub mod types;
/// Internal shared input-validation helpers used by every model
mod validation;

pub use clustering::{DBSCAN, KMeans, MeanShift, estimate_bandwidth};
pub use decomposition::{EigenSolver, KernelPCA, PCA, SVDSolver};
pub use discriminant_analysis::{LDA, Shrinkage, Solver};
pub use ensemble::{IsolationForest, IsolationTree};
pub use linear_model::{LinearRegression, LogisticRegression, generate_polynomial_features};
pub use manifold::{Init, TSNE, TSNEMethod};
pub use neighbors::{KNN, WeightingStrategy};
pub use svm::{LinearSVC, Loss, SVC};
pub use traits::{Fit, FitTransform, Predict, Transform};
pub use tree::{Algorithm, DecisionTree, DecisionTreeParams, Node, NodeType};
