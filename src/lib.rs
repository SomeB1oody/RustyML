//! # RustyML - A Comprehensive Machine Learning and Deep Learning Library in Pure Rust
//!
//! RustyML is a high-performance machine learning and deep learning library written entirely in Rust,
//! leveraging Rust's memory safety, concurrency features, and zero-cost abstractions to provide
//! efficient implementations of classical ML algorithms, neural networks, and data processing utilities.
//!
//! ## Overview
//!
//! This crate offers a complete ecosystem for machine learning tasks, from data preprocessing
//! and feature engineering to model training and evaluation. All implementations are designed
//! with production use in mind, featuring robust error handling, parallel processing optimization,
//! and comprehensive input validation.
//!
//! ## Architecture
//!
//! The library is organized into six main modules, each gated by feature flags:
//!
//! ### [`machine_learning`]
//! Classical machine learning algorithms for supervised and unsupervised learning:
//! - **Regression**: Linear Regression with L1/L2 regularization
//! - **Classification**: Logistic Regression, KNN, Decision Tree, SVC, Linear SVC, LDA
//! - **Clustering**: KMeans, DBSCAN, MeanShift
//! - **Anomaly Detection**: Isolation Forest
//!
//! ### [`neural_network`]
//! Complete neural network framework with flexible architecture design:
//! - **Layers**: Dense, RNN, LSTM, Convolution, Pooling, Dropout
//! - **Optimizers**: SGD, Adam, RMSProp, AdaGrad
//! - **Loss Functions**: MSE, MAE, Binary/Categorical Cross-Entropy
//! - **Models**: Sequential architecture for feed-forward networks
//!
//! ### [`utility`]
//! Data preprocessing and dimensionality reduction utilities:
//! - **Dimensionality Reduction**: PCA, Kernel PCA, LDA, t-SNE
//! - **Preprocessing**: Standardization, train-test splitting
//! - **Kernel Functions**: RBF, Linear, Polynomial, Sigmoid
//!
//! ### [`metric`]
//! Comprehensive evaluation metrics for model performance assessment:
//! - **Regression**: MSE, RMSE, MAE, R² score
//! - **Classification**: Accuracy, Confusion Matrix, AUC-ROC, F1-score
//! - **Clustering**: Adjusted Rand Index, Normalized/Adjusted Mutual Information, Silhouette Score
//!
//! ### [`math`]
//! Mathematical utilities and statistical functions:
//! - **Distance Metrics**: Euclidean, Manhattan, Minkowski
//! - **Impurity Measures**: Entropy, Gini, Information Gain
//! - **Statistical Functions**: Variance, standard deviation, SST, SSE
//! - **Activation Functions**: Sigmoid, logistic loss
//!
//! ### [`dataset`]
//! Access to standardized datasets for experimentation:
//! - Iris, Diabetes, Boston Housing, Wine Quality, Titanic
//! - Pre-processed and ready for immediate use
//!
//! ## Quick Start
//!
//! ### Machine Learning Example
//!
//! Add RustyML to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rustyml = { version = "*", features = ["machine_learning"] }
//! # Or use features = ["full"] to enable all modules
//! # Or use `features = ["default"]` to enable default modules (`machine_learning` and `neural_network`)
//! ```
//!
//! In your Rust code, write:
//! ```rust, ignore
//! use rustyml::machine_learning::linear_regression::*; // or `use rustyml::prelude::*`
//! use ndarray::{Array1, Array2};
//!
//! // Create a linear regression model
//! let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();
//!
//! // Prepare training data
//! let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
//! let raw_y = vec![6.0, 9.0, 12.0];
//!
//! // Convert Vec to ndarray types
//! let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
//! let y = Array1::from_vec(raw_y);
//!
//! // Train the model
//! model.fit(&x.view(), &y.view()).unwrap();
//!
//! // Make predictions
//! let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
//! let _predictions = model.predict(&new_data.view());
//!
//! // Save the trained model to a file
//! model.save_to_path("linear_regression_model.json").unwrap();
//!
//! // Load the model from the file
//! let loaded_model = LinearRegression::load_from_path("linear_regression_model.json").unwrap();
//!
//! // Use the loaded model for predictions
//! let _loaded_predictions = loaded_model.predict(&new_data.view());
//!
//! // Since Clone is implemented, the model can be easily cloned
//! let _model_copy = model.clone();
//!
//! // Since Debug is implemented, detailed model information can be printed
//! println!("{:?}", model);
//! ```
//!
//! ## Neural Network Example
//!
//! Add RustyML to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rustyml = { version = "*", features = ["neural_network"] }
//! # Or use `features = ["full"]` to enable all modules
//! # Or use `features = ["default"]` to enable default modules (`machine_learning` and `neural_network`)
//! ```
//!
//! In your Rust code, write:
//! ```rust, ignore
//! use rustyml::prelude::*;
//! use ndarray::Array;
//!
//! // Create training data
//! let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features
//! let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
//!
//! // Build a neural network
//! let mut model = Sequential::new();
//! model
//!     .add(Dense::new(784, 128, ReLU::new()).unwrap())
//!     .add(Dense::new(128, 64, ReLU::new()).unwrap())
//!     .add(Dense::new(64, 10, Softmax::new()).unwrap())
//!     .compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), CategoricalCrossEntropy::new());
//!
//! // Display model structure
//! model.summary();
//!
//! // Train the model
//! model.fit(&x, &y, 10).unwrap();
//!
//! // Save model weights to file
//! model.save_to_path("model.json").unwrap();
//!
//! // Create a new model with the same architecture
//! let mut new_model = Sequential::new();
//! new_model
//!     .add(Dense::new(784, 128, ReLU::new()).unwrap())
//!     .add(Dense::new(128, 64, ReLU::new()).unwrap())
//!     .add(Dense::new(64, 10, Softmax::new()).unwrap());
//!
//! // Load weights from file
//! new_model.load_from_path("model.json").unwrap();
//!
//! // Compile before using (required for training, optional for prediction)
//! new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), CategoricalCrossEntropy::new());
//!
//! // Make predictions with loaded model
//! let predictions = new_model.predict(&x);
//! println!("Predictions shape: {:?}", predictions.shape());
//! ```
//!
//! ## Feature Flags
//!
//! The crate uses feature flags for modular compilation:
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `machine_learning` | Classical ML algorithms (depends on `math`, `utility`) |
//! | `neural_network` | Neural network framework |
//! | `utility` | Data preprocessing and dimensionality reduction |
//! | `metric` | Evaluation metrics |
//! | `math` | Mathematical utilities |
//! | `dataset` | Standard datasets |
//! | `default` | Enables `machine_learning` and `neural_network` |
//! | `full` | Enables all features |

/// A macro that generates a getter method for any field.
///
/// This macro creates a public getter method that returns the value or reference
/// of the specified field. The generated method includes appropriate documentation
/// describing the field being accessed.
///
/// # Parameters
///
/// - `$method_name` - The name of the getter method (e.g., get_fit_intercept)
/// - `$field_name` - The name of the field to access (e.g., fit_intercept)
/// - `$return_type` - The return type of the getter method
///
/// # Generated Method
///
/// The macro generates a method that returns the field value,
/// with documentation that describes what field is being accessed.
#[cfg(any(feature = "machine_learning", feature = "utility"))]
macro_rules! get_field {
    ($method_name:ident, $field_name:ident, $return_type:ty) => {
        #[doc = concat!("Gets the `", stringify!($field_name), "` field.\n\n")]
        #[doc = "# Returns\n\n"]
        #[doc = concat!("* `", stringify!($return_type), "` - The value of the `", stringify!($field_name), "` field")]
        pub fn $method_name(&self) -> $return_type {
            self.$field_name
        }
    };
}

/// A macro that generates a public getter method returning a reference to a field.
///
/// This macro creates a method that provides immutable reference access to a private field
/// in a struct, following the Rust convention of getter methods.
///
/// # Parameters
///
/// - `$method_name` - The identifier for the generated getter method name
/// - `$field_name` - The identifier of the struct field to access
/// - `$return_type` - The type expression for the return value (typically a reference type like `&Type`)
///
/// # Generated Method
///
/// The macro generates a method that returns the field value as a reference,
/// with documentation that describes what field is being accessed
#[cfg(any(feature = "machine_learning", feature = "utility"))]
macro_rules! get_field_as_ref {
    ($method_name:ident, $field_name:ident, $return_type:ty) => {
        #[doc = concat!("Gets the `", stringify!($field_name), "` field.\n\n")]
        #[doc = "# Returns\n\n"]
        #[doc = concat!("* `", stringify!($return_type), "` - The value of the `", stringify!($field_name), "` field as a reference")]
        pub fn $method_name(&self) -> $return_type {
            self.$field_name.as_ref()
        }
    };
}

/// Macro for generating save_to_path and load_from_path methods for model structs.
///
/// This macro generates two methods:
/// - `save_to_path`: Saves the model to a JSON file at the specified path
/// - `load_from_path`: Loads a model from a JSON file at the specified path
///
/// # Parameters
///
/// * `$model_type` - The type of the model struct (e.g., LinearRegression, LogisticRegression)
#[cfg(any(feature = "machine_learning", feature = "utility"))]
macro_rules! model_save_and_load_methods {
    ($model_type:ty) => {
        /// Saves the trained model to a JSON file at the specified path.
        ///
        /// This method serializes the entire model state including coefficients, intercept,
        /// hyperparameters, and training metadata to JSON format using serde_json.
        ///
        /// # Parameters
        ///
        /// * `path` - File path where the model will be saved (e.g., "stored_model.json")
        ///
        /// # Returns
        ///
        /// - `Ok(())` - Model successfully saved to file
        /// - `Err(IoError::StdIoError)` - File creation or write operation failed
        /// - `Err(IoError::JsonError)` - Serialization to JSON failed
        pub fn save_to_path(&self, path: &str) -> Result<(), crate::error::IoError> {
            use crate::error::IoError;
            use serde_json::to_writer_pretty;
            use std::fs::File;
            use std::io::{BufWriter, Write};

            // Create or overwrite the file
            let file = File::create(path).map_err(IoError::StdIoError)?;
            let mut writer = BufWriter::new(file);

            // Serialize the model to JSON and write to file
            to_writer_pretty(&mut writer, self).map_err(IoError::JsonError)?;

            // Ensure all data is written to disk
            writer.flush().map_err(IoError::StdIoError)?;

            Ok(())
        }

        /// Loads a trained model from a JSON file at the specified path.
        ///
        /// This method deserializes a previously saved model from JSON format,
        /// restoring all model parameters, hyperparameters, and training state.
        ///
        /// # Parameters
        ///
        /// * `path` - File path from which to load the model (e.g., "stored_model.json")
        ///
        /// # Returns
        ///
        /// - `Ok(Self)` - Successfully loaded model instance
        /// - `Err(IoError::StdIoError)` - File not found or read operation failed
        /// - `Err(IoError::JsonError)` - Deserialization from JSON failed (invalid format or corrupted data)
        pub fn load_from_path(path: &str) -> Result<Self, crate::error::IoError> {
            use crate::error::IoError;
            use serde_json::from_reader;

            // Open and buffer the file for reading
            let reader = IoError::load_in_buf_reader(path)?;

            // Deserialize the model from JSON
            let model: $model_type = from_reader(reader).map_err(IoError::JsonError)?;

            Ok(model)
        }
    };
}

/// Error handling module containing custom error types for machine learning operations.
///
/// This module defines comprehensive error types that can occur throughout the machine learning
/// library, providing structured error handling for various failure modes including model state
/// validation, input data validation, processing errors, and I/O operations.
///
/// # Error Types
///
/// ## ModelError
/// Primary error type for machine learning model operations:
/// - **NotFitted**: Model hasn't been trained/fitted before prediction or transformation
/// - **InputValidationError**: Invalid input data format, dimensions, or values (NaN/infinite)
/// - **TreeError**: Decision tree structure or operation errors
/// - **ProcessingError**: General computation or algorithm execution errors
///
/// ## IoError
/// Error type for file operations and serialization:
/// - **StdIoError**: Standard I/O errors during file system operations
/// - **JsonError**: JSON serialization/deserialization failures
#[cfg(any(
    feature = "machine_learning",
    feature = "neural_network",
    feature = "utility"
))]
pub mod error;

/// Module `math` contains mathematical utility functions for statistical operations and model evaluation.
///
/// This module provides comprehensive mathematical functions essential for machine learning algorithms,
/// including impurity measures for decision trees, distance calculations for clustering algorithms,
/// statistical measures for evaluation, and various mathematical utilities for data processing.
///
/// # Core Functions
///
/// ## Decision Tree Mathematics
/// - `entropy` - Calculates the entropy of a label set for information-based splitting
/// - `gini` - Calculates the Gini impurity for CART-based splitting
/// - `information_gain` - Measures information gained from dataset splitting
/// - `gain_ratio` - Normalized information gain for C4.5 algorithm
/// - `c` - Calculates the average path length adjustment factor for isolation trees
///
/// ## Distance Calculations
/// - `squared_euclidean_distance_row` - Squared Euclidean distance between two vectors
/// - `manhattan_distance_row` - Manhattan (L1) distance between two vectors
/// - `minkowski_distance_row` - Generalized Minkowski distance with parameter p
/// - Finds the appropriate sigma value for a single sample's distances to achieve target perplexity
///
/// ## Statistical Functions
/// - `sum_of_square_total` - Total variability measurement (SST)
/// - `sum_of_squared_errors` - Sum of squared prediction errors (SSE)
/// - `variance` - Mean squared error or variance of a dataset
/// - `standard_deviation` - Population standard deviation calculation
/// - `average_path_length_factor` - Adjustment factor for isolation forest algorithms
///
/// ## Activation and Loss Functions
/// - `sigmoid` - Sigmoid activation function for neural networks and logistic regression
/// - `logistic_loss` - Cross-entropy loss for binary classification
///
/// # Example
/// ```rust
/// use rustyml::math::{entropy, gini, sigmoid, squared_euclidean_distance_row};
/// use ndarray::array;
///
/// // Decision tree impurity measures
/// let labels = array![0.0, 1.0, 1.0, 0.0];
/// let ent = entropy(&labels);
/// let gini_val = gini(&labels);
///
/// // Distance calculations
/// let v1 = array![1.0, 2.0];
/// let v2 = array![4.0, 6.0];
/// let dist = squared_euclidean_distance_row(&v1, &v2);
///
/// // Activation function
/// let activated = sigmoid(0.5);
/// ```
#[cfg(feature = "math")]
pub mod math;

/// Module `machine_learning` provides implementations of various machine learning algorithms and models.
///
/// This module includes a comprehensive collection of supervised and unsupervised learning algorithms
/// with parallel processing optimization and robust error handling for production use.
///
/// # Supervised Learning Algorithms
///
/// ## Classification
/// - **LogisticRegression**: Binary classification with gradient descent optimization and regularization support
/// - **KNN**: K-Nearest Neighbors with customizable distance metrics (Euclidean, Manhattan, Minkowski) and weighting strategies
/// - **DecisionTree**: Decision tree classifier supporting ID3, C4.5, and CART algorithms with pruning options
/// - **SVC**: Support Vector Classifier using Sequential Minimal Optimization (SMO) algorithm with kernel support
/// - **LinearSVC**: Linear Support Vector Classifier optimized for large datasets with hinge loss
/// - **LinearDiscriminantAnalysis**: LDA for classification and dimensionality reduction with class separability preservation
///
/// ## Regression
/// - **LinearRegression**: Simple and multivariate linear regression with L1/L2 regularization options
///
/// # Unsupervised Learning Algorithms
///
/// ## Clustering
/// - **KMeans**: K-means clustering with K-means++ initialization and parallel processing
/// - **DBSCAN**: Density-based clustering for discovering clusters of arbitrary shapes with noise detection
/// - **MeanShift**: Non-parametric clustering that automatically determines cluster centers
///
/// ## Anomaly Detection
/// - **IsolationForest**: Ensemble method for efficient anomaly detection in high-dimensional data
///
/// # Distance Metrics and Utilities
/// - `DistanceCalculationMetric` - Enum defining Euclidean, Manhattan, and Minkowski distance metrics
/// - `RegularizationType` - L1 and L2 regularization options for preventing overfitting
/// - Helper macros and validation functions for consistent model interfaces
///
/// # Examples
/// ```rust
/// use rustyml::machine_learning::*;
/// use ndarray::{Array1, Array2, array};
///
/// // Linear regression example
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
/// let y = array![6.0, 9.0, 12.0];
/// model.fit(&x, &y).unwrap();
///
/// // K-means clustering example
/// let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42)).unwrap();
/// let data = array![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]];
/// kmeans.fit(&data).unwrap();
/// let labels = kmeans.predict(&data).unwrap();
/// ```
#[cfg(feature = "machine_learning")]
pub mod machine_learning;

/// Module `prelude` re-exports the most commonly used types and traits from this crate.
///
/// This module provides a single import point for frequently used items from the machine learning library,
/// enabling quick access to essential components with a single `use` statement.
///
/// # Available Components
///
/// ## Machine Learning Models
/// - Classification algorithms (KNN, DecisionTree, LogisticRegression, SVC, LinearSVC, LinearDiscriminantAnalysis)
/// - Regression algorithms (LinearRegression)
/// - Clustering algorithms (KMeans, DBSCAN, MeanShift)
/// - Anomaly detection (IsolationForest)
///
/// ## Data Processing and Utilities
/// - Dimensionality reduction (PCA, kernel PCA, t-SNE, LDA)
/// - Data preprocessing (standardize, train_test_split)
/// - Feature engineering utilities
/// - and more (See details at documentation in utility module)
///
/// ## Evaluation Metrics
/// - Classification metrics (ConfusionMatrix, accuracy, calculate_auc)
/// - Regression metrics (mean_squared_error, r2_score, mean_absolute_error)
/// - Clustering metrics (adjusted_rand_index, silhouette_score)
/// - and more (See details at documentation in metric module)
///
/// ## Neural Network Components
/// - Complete neural network framework with layers, optimizers, loss functions
/// - Sequential model architecture for building feed-forward networks
/// - and more (See details at documentation in neural_network module)
///
/// # Examples
/// ```rust
/// use rustyml::prelude::*;
///
/// // Quick access to all commonly used components
/// ```
pub mod prelude;

/// Module `utility` provides a collection of utility functions and data processing tools to support machine learning operations.
///
/// This module provides essential data transformation and preprocessing capabilities that complement
/// the main machine learning algorithms, including dimensionality reduction techniques, data splitting
/// utilities, and various preprocessing functions.
///
/// # Dimensionality Reduction Techniques
///
/// ## Linear Methods
/// - **PCA (Principal Component Analysis)**: Linear dimensionality reduction for feature extraction and data visualization
/// - **LDA (Linear Discriminant Analysis)**: Supervised dimensionality reduction with class separability optimization
///
/// ## Non-linear Methods
/// - **Kernel PCA**: Non-linear dimensionality reduction using kernel methods for complex data patterns
/// - **t-SNE**: t-Distributed Stochastic Neighbor Embedding for high-dimensional data visualization
///
/// # Data Preprocessing
/// - **train_test_split**: Utility for splitting datasets into training and testing sets with configurable ratios
/// - **standardize**: Data standardization (z-score normalization) for feature scaling
/// - **KernelType**: Enumeration of supported kernel functions (RBF, Linear, Polynomial, Sigmoid)
///
/// # Key Features
/// - **Parallel Processing**: Rayon-based parallel computation for performance optimization
/// - **Flexible Configuration**: Customizable parameters for all algorithms
/// - **Memory Efficient**: Optimized implementations for large datasets
/// - **Robust Error Handling**: Comprehensive input validation and error reporting
///
/// # Examples
/// ```rust
/// use rustyml::utility::*;
/// use ndarray::{Array2, arr2};
///
/// // PCA dimensionality reduction
/// let mut pca = PCA::new(2).unwrap();
/// let data = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// let transformed = pca.fit_transform(&data).unwrap();
/// // and more
/// ```
#[cfg(feature = "utility")]
pub mod utility;

/// Module `metric` provides comprehensive evaluation metrics for statistical analysis and machine learning model performance assessment.
///
/// This module provides a complete collection of evaluation functions and structures for measuring
/// the performance of machine learning models across regression, classification, and clustering tasks
/// with robust statistical foundations and optimized implementations.
///
/// # Regression Metrics
/// - **mean_squared_error**: Average of squared differences between predicted and actual values
/// - **root_mean_squared_error**: Square root of MSE, providing error in original data units
/// - **mean_absolute_error**: Average magnitude of prediction errors without considering direction
/// - **r2_score**: Coefficient of determination measuring explained variance (R² score)
///
/// # Classification Metrics
///
/// ## ConfusionMatrix Structure
/// Comprehensive binary classification evaluation with:
/// - True/False Positive and Negative counts (TP, FP, TN, FN)
/// - Derived metrics: accuracy, precision, recall, specificity, F1-score, error rate
/// - Formatted summary generation for detailed performance reporting
///
/// ## Classification Functions
/// - **accuracy**: Standalone accuracy calculation for multi-class and binary classification
/// - **calculate_auc**: Area Under ROC Curve using Mann-Whitney U statistic for binary classification
///
/// # Clustering Evaluation Metrics
/// - **adjusted_rand_index**: Adjusted Rand Index for clustering similarity measurement with chance correction
/// - **normalized_mutual_info**: Normalized Mutual Information measuring clustering agreement
/// - **adjusted_mutual_info**: Mutual information adjusted for chance agreement between clusterings
///
/// # Key Features
/// - **Robust Input Validation**: Comprehensive error checking with informative messages
/// - **Numerical Stability**: Epsilon handling and stable algorithms for edge cases
/// - **Performance Optimized**: Single-pass calculations and efficient implementations
/// - **Statistical Rigor**: Theoretically sound implementations with proper mathematical foundations
///
/// # Examples
/// ```rust
/// use rustyml::metric::*;
/// use ndarray::{Array1, array};
///
/// // Regression evaluation
/// let predictions = array![3.0, 2.0, 3.5, 4.1];
/// let actuals = array![2.8, 2.1, 3.3, 4.2];
/// let mse = mean_squared_error(&actuals.view(), &predictions.view());
/// let r2 = r2_score(&predictions.view(), &actuals.view());
///
/// // Classification evaluation with confusion matrix
/// let predicted = array![1.0, 0.0, 1.0, 1.0, 0.0];
/// let actual = array![1.0, 0.0, 0.0, 1.0, 1.0];
/// let cm = ConfusionMatrix::new(&predicted.view(), &actual.view());
/// println!("F1 Score: {:.3}", cm.f1_score());
///
/// // AUC-ROC for binary classification
/// let scores = array![0.1, 0.4, 0.35, 0.8];
/// let labels = array![false, true, false, true];
/// let auc = calculate_auc(&scores.view(), &labels.view());
/// ```
#[cfg(feature = "metric")]
pub mod metric;

/// Module `dataset` provides access to standardized datasets for machine learning experimentation and algorithm benchmarking.
///
/// This module provides convenient access to well-known datasets commonly used in machine learning
/// research, education, and algorithm validation. All datasets are pre-processed and ready for
/// immediate use with the library's machine learning algorithms.
///
/// # Available Datasets
/// - **iris**: Classic iris flower dataset for multi-class classification (150 samples, 4 features, 3 classes)
/// - **diabetes**: Regression dataset for predicting diabetes progression (442 samples, 10 features)
/// - **boston_housing**: Housing price prediction dataset for regression tasks
/// - **wine_quality**: Wine quality datasets for both red and white wines (classification/regression)
/// - **titanic**: Famous Titanic survival prediction dataset for binary classification
/// - **raw_data**: Access to raw data loading utilities
///
/// # Data Format
/// All datasets return tuples in the format `(headers, data, target)` where:
/// - `headers`: Vector of feature names as strings
/// - `data`: 2D ndarray with samples as rows and features as columns
/// - `target`: 1D ndarray with target values or class labels
///
/// # Examples
/// ```rust
/// use rustyml::dataset::iris;
///
/// // Load the iris dataset
/// let (headers, data, class) = iris::load_iris();
/// println!("Dataset shape: {:?}", data.shape());
/// println!("Classes: {:?}", class);
/// println!("Features: {:?}", headers);
/// ```
#[cfg(feature = "dataset")]
pub mod dataset;

/// Module `neural_network` provides components for building and training neural networks with flexible architecture design.
///
/// This module provides a comprehensive framework for constructing, training, and deploying
/// neural networks with support for various layer types, optimization algorithms, loss functions,
/// and model architectures.
///
/// # Core Components
///
/// ## Layer Types
/// - **Dense**: Fully connected layers with customizable activation functions
/// - **Activation**: Standalone activation layers (ReLU, Sigmoid, Tanh, Softmax, Linear, etc.)
/// - **Pooling Layers**: Max pooling and average pooling operations for 1D, 2D, and 3D data
/// - **Global Pooling**: Global max pooling and global average pooling for 1D, 2D, and 3D tensors
/// - **Recurrent Layers**: Sequential Modeling Layers like RNN, LSTM, and GRU
/// - **Regularization layers**: Regularization layer to prevent overfitting during training
///
/// ## Optimization Algorithms
/// - **SGD**: Stochastic Gradient Descent with momentum support
/// - **Adam**: Adaptive moment estimation optimizer
/// - **RMSProp**: Root Mean Square Propagation optimizer
/// - **AdaGrad**: Adaptive gradient algorithm
///
/// ## Loss Functions
/// - **MeanSquaredError**: For regression tasks
/// - **BinaryCrossEntropy**: For binary classification
/// - **CategoricalCrossEntropy**: For multi-class classification
/// - **SparseCategoricalCrossEntropy**: For multi-class with integer labels
///
/// ## Model Architecture
/// - **Sequential**: Linear stack of layers for feed-forward neural networks
/// - **Tensor**: Type alias for n-dimensional arrays used throughout the framework
///
/// # Key Features
/// - **Flexible Architecture**: Easy model construction with intuitive API
/// - **Automatic Differentiation**: Built-in backpropagation implementation
/// - **Training Loop**: Integrated training with loss tracking and convergence monitoring
/// - **Prediction Interface**: Simple prediction methods for inference
///
/// # Examples
/// ```rust
/// use rustyml::neural_network::*;
/// use ndarray::Array;
///
/// // Create input and target tensors
/// let x = Array::ones((2, 4)).into_dyn();  // 2 samples, 4 features
/// let y = Array::ones((2, 1)).into_dyn();  // 2 samples, 1 output
///
/// // Build sequential model
/// let mut model = Sequential::new();
/// model.add(Dense::new(4, 8, ReLU::new()).unwrap())   // Input layer: 4 -> 8
///      .add(Dense::new(8, 3, ReLU::new()).unwrap())   // Hidden layer: 8 -> 3
///      .add(Dense::new(3, 1, Linear::new()).unwrap()); // Output layer: 3 -> 1
///
/// // Compile with optimizer and loss function
/// model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), MeanSquaredError::new());
///
/// // Display model architecture
/// model.summary();
///
/// // Train the model
/// model.fit(&x, &y, 100).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&x);
/// ```
#[cfg(feature = "neural_network")]
pub mod neural_network;

#[cfg(test)]
mod test;
