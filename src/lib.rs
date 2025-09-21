/// Error types that can occur during model operations
///
/// # Variants
///
/// - `NotFitted` - Indicates that the model has not been fitted yet
/// - `InputValidationError` - indicates the input data provided does not meet the expected format, type, or validation rules
/// - `TreeError` - indicates that there is something wrong with the tree
/// - `ProcessingError` - indicates that there is something wrong while processing
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    NotFitted,
    InputValidationError(String),
    TreeError(&'static str),
    ProcessingError(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NotFitted => {
                write!(f, "Model has not been fitted. Parameters are unavailable.")
            }
            ModelError::InputValidationError(msg) => write!(f, "Input validation error: {}", msg),
            ModelError::TreeError(msg) => write!(f, "Tree structure error: {}", msg),
            ModelError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

/// Implements the standard error trait for ModelError
impl std::error::Error for ModelError {}

/// # Module `math` contains mathematical utility functions for statistical operations and model evaluation.
///
/// # Included formula
///
/// - Sum of square total (SST) for measuring data variability
/// - Sum of squared errors (SSE) for evaluating prediction errors
/// - Sigmoid function for logistic regression and neural networks
/// - Logistic loss (log loss) for binary classification models
/// - Accuracy score for classification model evaluation
/// - Calculate the squared Euclidean distance between two points
/// - Calculate the Manhattan distance between two points
/// - Calculate the Minkowski distance between two points
/// - Calculate the Gaussian kernel (RBF kernel)
/// - Calculates the entropy of a label set
/// - Calculates the Gini impurity of a label set
/// - Calculates the information gain when splitting a dataset
/// - Calculates the gain ratio for a dataset split
/// - Calculates the Mean Squared Error (MSE) of a set of values
/// - Calculates the leaf node adjustment factor c(n)
/// - Calculates the standard deviation of a set of values
///
/// # Example
/// ```rust
/// use rustyml::math::sum_of_squared_errors;
/// use ndarray::array;
///
/// // Example data
/// let predicted = array![2.1, 3.8, 5.2, 7.1];
/// let actual = array![2.0, 4.0, 5.0, 7.0];
///
/// // Calculate error metrics
/// let sse = sum_of_squared_errors(predicted.view(), actual.view());
/// ```
#[cfg(feature = "math")]
#[cfg_attr(docsrs, doc(cfg(feature = "math")))]
pub mod math;

/// Module `machine_learning` provides implementations of various machine learning algorithms and models.
///
/// This module includes a collection of supervised and unsupervised learning algorithms
/// that can be used for tasks such as classification, regression, and clustering:
///
/// # Supervised Learning Algorithms
///
/// ## Classification
/// - **LogisticRegression**: Binary classification using logistic regression with gradient descent optimization
/// - **KNN**: K-Nearest Neighbors classifier with customizable distance metrics and weighting strategies
/// - **DecisionTree**: Decision tree classifier with various splitting criteria and pruning options
/// - **Support Vector Machines (SVM)**: A set of supervised learning methods used for classification, regression, and outlier detection. This implementation uses the Sequential Minimal Optimization (SMO) algorithm.
/// - **Linear Support Vector Classifier (LinearSVC)**: Implements a classifier similar to sklearn's LinearSVC, trained using the hinge loss function.
/// - **Linear Discriminant Analysis (LDA)**: A classifier and dimensionality reduction technique that projects data onto a lower-dimensional space while maintaining class separability
///
/// ## Regression
/// - **LinearRegression**: Simple and multivariate linear regression with optional intercept fitting
///
/// # Unsupervised Learning Algorithms
///
/// ## Clustering
/// - **KMeans**: K-means clustering with customizable initialization and convergence criteria
/// - **DBSCAN**: Density-based spatial clustering of applications with noise
/// - **MeanShift**: Non-parametric clustering that finds clusters by identifying density modes
///
/// ## Anomaly Detection
/// - **IsolationForest**: Isolation Forest for identifying anomalies and outliers in the data
///
/// # Utility Functions
/// - **estimate_bandwidth**: Estimates the bandwidth parameter for MeanShift clustering
/// - **generate_polynomial_features**: Creates polynomial features for enhancing model complexity
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::linear_regression::*;
/// use ndarray::{Array1, Array2, array};
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None);
///
/// // Prepare training data
/// let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
/// let raw_y = vec![6.0, 9.0, 12.0];
///
/// // Convert Vec to ndarray types
/// let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
/// let y = Array1::from_vec(raw_y);
///
/// // Train the model
/// model.fit(x.view(), y.view()).unwrap();
///
/// // Make predictions
/// let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
/// let predictions = model.predict(new_data.view());
///
/// // Since Clone is implemented, the model can be easily cloned
/// let model_copy = model.clone();
///
/// // Since Debug is implemented, detailed model information can be printed
/// println!("{:?}", model);
/// ```
#[cfg(feature = "machine_learning")]
#[cfg_attr(docsrs, doc(cfg(feature = "machine_learning")))]
pub mod machine_learning;

/// A convenience module that re-exports the most commonly used types and traits from this crate.
///
/// This module provides a single import point for frequently used items from this library's machine learning modules,
/// allowing users to import multiple items with a single `use` statement.
///
/// # Examples
/// ```rust
/// // Import all common items
/// use rustyml::prelude::*;
///
/// // Now you can use items like DBSCAN, KMeans, DecisionTree, etc. directly
/// ```
///
/// # Available Components
///
/// ## Machine Learning Models
///
/// ### Clustering
/// - `DBSCAN` - Density-based spatial clustering of applications with noise
/// - `KMeans` - K-means clustering algorithm
/// - `MeanShift` - Non-parametric clustering that finds clusters by identifying density modes
///
/// ### Classification
/// - `KNN` - K-Nearest Neighbors classifier
/// - `DecisionTree` - Decision tree classifier with various splitting criteria
/// - `LogisticRegression` - Binary classification using logistic regression
/// - `SVC` - Support Vector Classifier using Sequential Minimal Optimization
///
/// ### Regression
/// - `LinearRegression` - Simple and multivariate linear regression
///
/// ### Anomaly Detection
/// - `IsolationForest` - Isolation Forest for identifying anomalies and outliers
///
/// ## Utility Functions and Data Processing
///
/// ### Dimensionality Reduction
/// - `PCA` - Principal Component Analysis for dimensionality reduction
///
/// ### Data Preprocessing
/// - `standardize` - Data standardization utility
/// - `train_test_split` - Split datasets into training and testing sets
/// - `generate_polynomial_features` - Create polynomial features for enhanced model complexity
/// - `estimate_bandwidth` - Estimate bandwidth parameter for MeanShift clustering
///
/// ## Evaluation Metrics
/// - `ConfusionMatrix` - Binary classification evaluation matrix
/// - `accuracy` - Calculate classification accuracy
/// - `mean_squared_error` - Calculate mean squared error for regression
/// - `r2_score` - Calculate coefficient of determination (R²)
///
/// ## Neural Network Components
/// - Complete neural network framework including layers, optimizers, loss functions, and sequential models
/// - All components from the `neural_network` module are available
///
/// ## Configuration Types and Enums
/// - `DistanceCalculationMetric` - Distance metrics for various algorithms
/// - `KernelType` - Kernel types for SVM algorithms
/// - `RegularizationType` - Regularization types for regression models
/// - `WeightingStrategy` - Weighting strategies for KNN classifier
/// - `Algorithm` - Algorithm options for decision trees
/// - `DecisionTreeParams` - Parameter configuration for decision trees
///
/// ## Traits
/// - All common traits for machine learning models and neural networks
/// - Includes interfaces for model fitting, prediction, and parameter access
pub mod prelude;

/// A collection of utility functions and data processing tools to support machine learning operations.
///
/// This module provides various utility components that are commonly used across different machine learning
/// tasks, including data transformation, dimensionality reduction, and preprocessing techniques.
///
/// # Submodules
///
/// ## Dimensionality Reduction
/// * `principal_component_analysis` - Implementation of Principal Component Analysis (PCA) for
///   dimensionality reduction and feature extraction
/// * `kernel_pca` - Kernel Principal Component Analysis for non-linear dimensionality reduction
///   using kernel methods
/// * `t_sne` - t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation for
///   visualizing high-dimensional data
/// * `linear_discriminant_analysis` - Linear Discriminant Analysis (LDA) for classification
///   and dimensionality reduction with class separability preservation
///
/// ## Data Preprocessing
/// * `train_test_split` - Utility function for splitting datasets into training and test sets
///   with configurable test size and random state
///
/// # Features
///
/// - **Linear Dimensionality Reduction**: PCA for extracting principal components from data
/// - **Non-linear Dimensionality Reduction**: Kernel PCA and t-SNE for complex data patterns
/// - **Supervised Dimensionality Reduction**: LDA for class-aware dimensionality reduction
/// - **Data Splitting**: Train-test split functionality with random sampling
/// - **Flexible Configuration**: Customizable parameters for all algorithms
/// - **Performance Optimized**: Parallel processing support using Rayon
///
/// # Examples
/// ```rust
/// use rustyml::utility::principal_component_analysis::PCA;
/// use ndarray::{Array2, arr2};
///
/// // Create a new PCA instance with 2 components
/// let mut pca = PCA::new(2);
///
/// // Data to transform
/// let data = arr2(&[
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
/// ]);
///
/// // Fit and transform data
/// let transformed = pca.fit_transform(data.view()).unwrap();
/// ```
#[cfg(feature = "utility")]
#[cfg_attr(docsrs, doc(cfg(feature = "utility")))]
pub mod utility;

/// This module provides implementation of comprehensive evaluation metrics used in statistical analysis and machine learning models.
///
/// The metric module contains a comprehensive collection of evaluation functions and structures for measuring
/// the performance of machine learning models across various domains including regression, classification,
/// and probabilistic prediction tasks.
///
/// # Regression Metrics
///
/// The module offers several metrics for evaluating regression models:
///
/// - `mean_squared_error` - Calculates the Mean Squared Error (MSE), measuring the average of squared differences between predicted and actual values
/// - `root_mean_squared_error` - Calculates the Root Mean Squared Error (RMSE), providing MSE in the same units as the original data for better interpretability
/// - `mean_absolute_error` - Calculates the Mean Absolute Error (MAE), measuring the average magnitude of errors without considering direction
/// - `r2_score` - Calculates the coefficient of determination (R²) that measures how well a model explains the variance in the target variable
///
/// # Classification Metrics
///
/// The module provides comprehensive binary classification evaluation tools:
///
/// ## ConfusionMatrix Struct
/// * A comprehensive structure for binary classification evaluation containing:
///   - True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) counts
///   - Methods for calculating derived metrics: accuracy, error rate, precision, recall, specificity, and F1 score
///   - Summary generation for detailed performance reporting
///
/// ## Classification Functions
/// - `accuracy` - Standalone function for calculating classification accuracy from predicted and actual arrays
/// - `calculate_auc` - Computes the Area Under the ROC Curve (AUC-ROC) for binary classification models using the Mann-Whitney U statistic
///
/// # Advanced Statistical Metrics
///
/// - `adjusted_rand_index` - Calculates the Adjusted Rand Index for clustering evaluation, measuring similarity between predicted and ground truth clusterings
/// - `silhouette_score` - Computes the Silhouette Coefficient for clustering quality assessment, measuring how similar objects are to their own cluster versus other clusters
///
/// # Key Features
///
/// - **Robust Input Validation**: All functions include comprehensive input validation with clear error messages
/// - **Edge Case Handling**: Proper handling of empty arrays, division by zero, and other edge cases
/// - **Numerical Stability**: Implementation includes epsilon values and other techniques to handle floating-point precision issues
/// - **Performance Optimized**: Efficient single-pass calculations where possible to minimize computational overhead
/// - **Comprehensive Documentation**: Each function includes detailed parameter descriptions, return values, panic conditions, and practical examples
///
/// # Examples
/// ```rust
/// use rustyml::metric::*;
/// use ndarray::{Array1, array};
///
/// // Regression metrics example
/// let predictions = array![3.0, 2.0, 3.5, 4.1];
/// let actuals = array![2.8, 2.1, 3.3, 4.2];
///
/// let mse = mean_squared_error(actuals.view(), predictions.view());
/// let rmse = root_mean_squared_error(predictions.view(), actuals.view());
/// let mae = mean_absolute_error(predictions.view(), actuals.view());
/// let r2 = r2_score(predictions.view(), actuals.view());
///
/// println!("MSE: {:.4}, RMSE: {:.4}, MAE: {:.4}, R²: {:.4}", mse, rmse, mae, r2);
///
/// // Binary classification metrics example
/// let predicted = Array1::from(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
/// let actual = Array1::from(vec![1.0, 0.0, 0.0, 1.0, 1.0]);
///
/// // Using ConfusionMatrix for comprehensive evaluation
/// let cm = ConfusionMatrix::new(predicted.view(), actual.view());
/// println!("Accuracy: {:.3}", cm.accuracy());
/// println!("Precision: {:.3}", cm.precision());
/// println!("Recall: {:.3}", cm.recall());
/// println!("F1 Score: {:.3}", cm.f1_score());
/// println!("{}", cm.summary());
///
/// // Using standalone accuracy function
/// let acc = accuracy(predicted.view(), actual.view());
/// println!("Standalone accuracy: {:.3}", acc);
///
/// // AUC-ROC calculation example
/// let scores = array![0.1, 0.4, 0.35, 0.8];
/// let labels = array![false, true, false, true];
/// let auc = calculate_auc(scores.view(), labels.view());
/// println!("AUC-ROC: {:.3}", auc);
/// ```
#[cfg(feature = "metric")]
#[cfg_attr(docsrs, doc(cfg(feature = "metric")))]
pub mod metric;

/// This module provides access to common datasets used for testing and benchmarking machine learning algorithms
///
/// The dataset module contains standardized datasets that are frequently used in
/// machine learning research and education, making it easier to test algorithms
/// against well-known data.
///
/// # Available Datasets
///
/// - `iris` - The famous Iris flower dataset containing measurements for three species of iris
/// - `diabetes` - The diabetes dataset for regression analysis
/// - `wine_quality` - The wine quality data set (red wine and white wine)
///
/// # Usage Example
/// ``` rust
/// use rustyml::dataset::iris;
///
/// // Load the iris dataset
/// let (headers, data, class) = iris::load_iris();
/// println!("Loaded iris dataset with {} samples", data.shape()[0]);
/// ```
#[cfg(feature = "dataset")]
#[cfg_attr(docsrs, doc(cfg(feature = "dataset")))]
pub mod dataset;

/// This module provides components for building and training neural networks.
///
/// # Features
///
/// - Various layer implementations for neural network construction
/// - Loss functions for measuring model performance
/// - Optimizers for parameter updates during training
/// - Sequential model for building linear neural networks
///
/// # Core Components
///
/// - `Layer`: Trait for neural network layers with forward and backward propagation
/// - `LossFunction`: Trait for computing loss and gradients
/// - `Optimizer`: Trait for parameter optimization strategies
/// - `Tensor`: Type alias for n-dimensional arrays used throughout the module
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::*;
/// use ndarray::Array;
///
/// //Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 3, Activation::ReLU))
/// .add(Dense::new(3, 1, Activation::ReLU));
/// model.compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // Print model structure (summary)
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 3).unwrap();
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("Prediction results: {:?}", prediction);
/// ```
///
/// Each submodule contains specialized components:
/// - `layer`: Different neural network layers (Dense, Activation, etc.)
/// - `loss_function`: Various loss functions (MSE, CrossEntropy, etc.)
/// - `optimizer`: Parameter optimization algorithms (SGD, Adam, RMSProp, etc.)
/// - `sequential`: Sequential model for creating feed-forward neural networks
#[cfg(feature = "neural_network")]
#[cfg_attr(docsrs, doc(cfg(feature = "neural_network")))]
pub mod neural_network;

#[cfg(test)]
mod test;
