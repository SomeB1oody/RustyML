//! Prelude re-exports for the machine learning estimators, traits, and shared enums

pub use crate::machine_learning::traits::{Fit, Predict};
pub use crate::machine_learning::{DistanceCalculationMetric, RegularizationType};

pub use crate::machine_learning::KernelType;
pub use crate::machine_learning::clustering::{DBSCAN, KMeans, MeanShift, estimate_bandwidth};
pub use crate::machine_learning::discriminant_analysis::{LDA, Shrinkage, Solver};
pub use crate::machine_learning::ensemble::IsolationForest;
pub use crate::machine_learning::linear_model::{
    LinearRegression, LogisticRegression, generate_polynomial_features,
};
pub use crate::machine_learning::neighbors::{KNN, WeightingStrategy};
pub use crate::machine_learning::svm::{LinearSVC, SVC};
pub use crate::machine_learning::tree::{Algorithm, DecisionTree, DecisionTreeParams};
