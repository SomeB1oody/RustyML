/// Error types that can occur during model operations
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// Indicates that the model has not been fitted yet
    NotFitted,
    /// indicates the input data provided does not meet the expected format, type, or validation rules.
    InputValidationError(String),
    /// indicates that there is something wrong with the tree
    TreeError(&'static str),
    /// indicates that there is something wrong while processing
    ProcessingError(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NotFitted => write!(f, "Model has not been fitted. Parameters are unavailable."),
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
/// This module provides implementations of common statistical measures used in machine learning:
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
/// These functions are particularly useful for regression model evaluation and
/// performance assessment in machine learning applications.
///
/// # Examples
///
/// ```
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
pub mod math;

#[cfg(test)]
mod math_module_test;

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
///
/// ```
/// use rustyml::machine_learning::linear_regression::LinearRegression;
/// use ndarray::{Array1, Array2, array};
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6);
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
pub mod machine_learning;

#[cfg(test)]
mod machine_learning_test;

/// A convenience module that re-exports the most commonly used types and traits from this crate.
///
/// This module provides a single import point for frequently used items from this library's machine learning modules,
/// allowing users to import multiple items with a single `use` statement.
///
/// # Examples
///
/// ```
/// // Import all common items
/// use rustyml::prelude::*;
///
/// // Now you can use items like DBSCAN, KMeans, DecisionTree, etc. directly
/// ```
///
/// # Available Components
///
/// This prelude exports the following machine learning algorithms and utilities:
/// - Clustering: `DBSCAN`, `KMeans`, `MeanShift`
/// - Classification: `KNN`, `DecisionTree`, `LogisticRegression`
/// - Regression: `LinearRegression`
/// - Dimensionality Reduction: `PCA`
/// - Anomaly Detection: `IsolationForest`
/// - Common parameters and metrics: `DistanceCalculationMetric`, `WeightingStrategy`, etc.
pub mod prelude;

/// # A collection of utility functions and data processing tools to support machine learning operations.
///
/// This module provides various utility components that are commonly used across different machine learning
/// tasks, including data transformation, dimensionality reduction, and preprocessing techniques.
///
/// ## Submodules
///
/// * `principal_component_analysis` - Implementation of Principal Component Analysis (PCA) for
///   dimensionality reduction and feature extraction.
///
/// ## Examples
///
/// ```
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
pub mod utility;

#[cfg(test)]
mod utility_test;

/// # This module provides implementation of common evaluation metrics used in statistical analysis and machine learning models.
///
/// ## Regression Metrics
///
/// The module offers several metrics for evaluating regression models:
///
/// * `root_mean_squared_error` - Calculates the square root of the mean of squared differences between predicted and actual values
/// * `mean_absolute_error` - Calculates the mean of absolute differences between predicted and actual values
/// * `r2_score` - Calculates the coefficient of determination (R²) that measures how well a model explains the variance in the target variable
///
/// ## Classification Metrics
///
/// The module provides the `ConfusionMatrix` struct for binary classification evaluation:
///
/// * Contains counts for true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
/// * Calculates various derived metrics:
///   - Accuracy
///   - Error rate
///   - Precision
///   - Recall
///   - Specificity
///   - F1 score
///
/// ## Examples
///
/// ```
/// use rustyml::metric::*;
/// use ndarray::{Array1, array};
///
/// // Regression metrics example
/// let predictions = array![3.0, 2.0, 3.5, 4.1];
/// let actuals = array![2.8, 2.1, 3.3, 4.2];
///
/// let rmse = root_mean_squared_error(predictions.view(), actuals.view());
/// let mae = mean_absolute_error(predictions.view(), actuals.view());
/// let r2 = r2_score(predictions.view(), actuals.view());
///
/// // Classification metrics example
/// let predicted = Array1::from(vec![1.0, 0.0, 1.0, 1.0, 0.0]);
/// let actual = Array1::from(vec![1.0, 0.0, 0.0, 1.0, 1.0]);
///
/// let cm = ConfusionMatrix::new(predicted.view(), actual.view());
/// println!("{}", cm.summary());
/// ```
pub mod metric;

#[cfg(test)]
mod metric_module_test;

/// # This module provides access to common datasets used for testing and benchmarking machine learning algorithms
///
/// The dataset module contains standardized datasets that are frequently used in
/// machine learning research and education, making it easier to test algorithms
/// against well-known data.
///
/// ## Available Datasets
///
/// * `iris` - The famous Iris flower dataset containing measurements for three species of iris
/// * `diabetes` - The diabetes dataset for regression analysis
///
/// ## Usage Example
///
/// ```
/// use rustyml::dataset::iris;
///
/// // Load the iris dataset
/// let (data, target) = iris::load_iris();
/// println!("Loaded iris dataset with {} samples", data.shape()[0]);
/// ```
pub mod dataset;

/// # This module provides components for building and training neural networks.
///
/// ## Features
///
/// - Various layer implementations for neural network construction
/// - Loss functions for measuring model performance
/// - Optimizers for parameter updates during training
/// - Sequential model for building linear neural networks
///
/// ## Core Components
///
/// - `Layer`: Trait for neural network layers with forward and backward propagation
/// - `LossFunction`: Trait for computing loss and gradients
/// - `Optimizer`: Trait for parameter optimization strategies
/// - `Tensor`: Type alias for n-dimensional arrays used throughout the module
///
/// ## Examples
///
/// ```
/// use rustyml::neural_network::*;
/// use ndarray::Array;
///
/// //Create input and target tensors, assuming input dimension is 4, output dimension is 3, batch_size = 2
/// let x = Array::ones((2, 4)).into_dyn();
/// let y = Array::ones((2, 1)).into_dyn();
///
/// // Build the model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 3))
/// .add(Dense::new(3, 1));
/// model.compile(SGD::new(0.01), MeanSquaredError::new());
///
/// // Print model structure (summary)
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 3);
///
/// // Use predict for forward propagation prediction
/// let prediction = model.predict(&x);
/// println!("Prediction results: {:?}", prediction);
/// ```
///
/// Each submodule contains specialized components:
/// - `layer`: Different neural network layers (Dense, Activation, Dropout, etc.)
/// - `loss_function`: Various loss functions (MSE, CrossEntropy, etc.)
/// - `optimizer`: Parameter optimization algorithms (SGD, Adam, RMSProp, etc.)
/// - `sequential`: Sequential model for creating feed-forward neural networks
pub mod neural_network;

#[cfg(test)]
mod neural_network_test;