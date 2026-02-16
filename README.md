<div style="text-align: right;">

[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)
</div>

# RustyML
A comprehensive machine learning and deep learning library written in pure Rust.

[![Rust Version](https://img.shields.io/badge/Rust-v.1.85-brown)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)

## Overview
RustyML aims to be a feature-rich machine learning and deep learning framework that leverages Rust's performance, memory safety, and concurrency features. While currently in early development stages, the project's long-term vision is to provide a complete ecosystem for machine learning, deep learning, and transformer-based models.

## Key Features
- **Pure Rust Implementation**: No external C/C++ dependencies, ensuring memory safety and portability
- **Parallel Processing**: Leverages Rayon for efficient multithreaded computation
- **Rich Algorithm Collection**: Supervised, unsupervised learning, and neural networks
- **Comprehensive Metrics**: Evaluation tools for regression, classification, and clustering
- **Model Persistence**: Save and load trained models with JSON serialization

## Architecture

### Machine Learning (`features = ["machine_learning"]`)
Classical machine learning algorithms for supervised and unsupervised learning:

- **Regression**:
  - Linear Regression with L1/L2 regularization

- **Classification**:
  - Logistic Regression
  - KNN (K-Nearest Neighbors)
  - Decision Tree (ID3, C4.5, CART)
  - SVC (Support Vector Classification)
  - Linear SVC
  - LDA (Linear Discriminant Analysis)

- **Clustering**:
  - KMeans with K-means++ initialization
  - DBSCAN (Density-based clustering)
  - MeanShift

- **Anomaly Detection**:
  - Isolation Forest

### Neural Network (`features = ["neural_network"]`)
Complete neural network framework with flexible architecture design:

- **Layers**:
  - Dense: Fully connected layers with customizable activation functions
  - Activation: Standalone activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)
  - Pooling Layers: Max pooling and average pooling operations for 1D, 2D, and 3D data
  - Global Pooling: Global max pooling and global average pooling for 1D, 2D, and 3D tensors
  - Recurrent Layers: Sequential modeling layers like RNN, LSTM, and GRU
  - Dropout: Regularization layer to prevent overfitting during training

- **Optimizers**:
  - SGD (Stochastic Gradient Descent)
  - Adam
  - RMSProp
  - AdaGrad

- **Loss Functions**:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
  - Sparse Categorical Cross-Entropy

- **Models**:
  - Sequential architecture for feed-forward networks

- **Activation layers**:
  - ReLU, Tanh, Sigmoid, Softmax

### Utility (`features = ["utility"]`)
Data preprocessing and dimensionality reduction utilities:

- **Dimensionality Reduction**:
  - PCA (Principal Component Analysis)
  - Kernel PCA (with RBF, Linear, Polynomial, Sigmoid kernels)
  - LDA (Linear Discriminant Analysis)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)

- **Preprocessing**:
  - Standardization (z-score normalization)
  - Train-test splitting

- **Kernel Functions**:
  - RBF, Linear, Polynomial, Sigmoid

### Metric (`features = ["metric"]`)
Comprehensive evaluation metrics for model performance assessment:

- **Regression Metrics**:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² score

- **Classification Metrics**:
  - Accuracy
  - Confusion Matrix (with TP, FP, TN, FN, precision, recall, F1-score)
  - AUC-ROC

- **Clustering Metrics**:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Adjusted Mutual Information (AMI)
  - Silhouette Score

### Math (`features = ["math"]`)
Mathematical utilities and statistical functions:

- **Distance Metrics**:
  - Euclidean distance
  - Manhattan distance
  - Minkowski distance

- **Impurity Measures**:
  - Entropy
  - Gini impurity
  - Information gain
  - Gain ratio

- **Statistical Functions**:
  - Variance
  - Standard deviation
  - SST (Sum of Squares Total)
  - SSE (Sum of Squared Errors)

- **Activation Functions**:
  - Sigmoid
  - Logistic loss

### Dataset (`features = ["dataset"]`)
Access to standardized datasets for experimentation:

- Iris (150 samples, 4 features, 3 classes)
- Diabetes (442 samples, 10 features)
- Boston Housing
- Wine Quality (red and white wines)
- Titanic

## Getting Started

### Machine Learning Example

Add the library to your `Cargo.toml`:
```toml
[dependencies]
rustyml = {version = "*", features = ["machine_learning"]} 
# or use `features = ["full"]` to enable all features
# Or use features = ["default"] to enable default modules (`machine_learning` and `neural_network`)
# Add `"show_progress"` in `features` to show progress bars when training
```

In your Rust code, write:
``` rust
use rustyml::machine_learning::linear_regression::*;
use ndarray::{Array1, Array2};
    
// Create a linear regression model
let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();

// Prepare training data
let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
let raw_y = vec![6.0, 9.0, 12.0];

// Convert Vec to ndarray types
let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
let y = Array1::from_vec(raw_y);

// Train the model
model.fit(&x.view(), &y.view()).unwrap();

// Make predictions
let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
let _predictions = model.predict(&new_data.view());

// Save the trained model to a file
model.save_to_path("linear_regression_model.json").unwrap();

// Load the model from the file
let loaded_model = LinearRegression::load_from_path("linear_regression_model.json").unwrap();

// Use the loaded model for predictions
let _loaded_predictions = loaded_model.predict(&new_data.view());

// Since Clone is implemented, the model can be easily cloned
let _model_copy = model.clone();

// Since Debug is implemented, detailed model information can be printed
println!("{:?}", model);
```

### Neural Network Example

Add the library to your `Cargo.toml`:
```toml
[dependencies]
rustyml = {version = "*", features = ["neural_network"]} 
# or use `features = ["full"]` to enable all features
# Or use `features = ["default"]` to enable default modules (`machine_learning` and `neural_network`)
# Add `"show_progress"` in `features` to show progress bars when training
```

In your Rust code, write:
``` rust
use rustyml::neural_network::{
    sequential::Sequential,
    layer::{Dense, ReLU, Softmax},
    optimizer::Adam,
    loss_function::CategoricalCrossEntropy,
}; 
use ndarray::Array;  
  
// Create training data   
let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features 
let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
  
// Build a neural network   
let mut model = Sequential::new();  
model  
    .add(Dense::new(784, 128, ReLU::new()).unwrap())    
    .add(Dense::new(128, 64, ReLU::new()).unwrap())    
    .add(Dense::new(64, 10, Softmax::new()).unwrap())    
    .compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), CategoricalCrossEntropy::new());  
// Display model structure   
model.summary();  
  
// Train the model  
model.fit(&x, &y, 10).unwrap();  
  
// Save model weights to file  
model.save_to_path("model.json").unwrap();  
  
// Create a new model with the same architecture  
let mut new_model = Sequential::new();
new_model  
    .add(Dense::new(784, 128, ReLU::new()).unwrap())    
    .add(Dense::new(128, 64, ReLU::new()).unwrap())    
    .add(Dense::new(64, 10, Softmax::new()).unwrap());
  
// Load weights from file  
new_model.load_from_path("model.json").unwrap();  
  
// Compile before using (required for training, optional for prediction)    
new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(), CategoricalCrossEntropy::new());  
  
// Make predictions with loaded model  
let predictions = new_model.predict(&x);  
println!("Predictions shape: {:?}", predictions.shape());  
```

## Feature Flags
The crate uses feature flags for modular compilation:

| Feature            | Description                                     |  
|--------------------|-------------------------------------------------|
| `machine_learning` | Classical ML algorithms (depends on `math`)     |  
| `neural_network`   | Neural network framework                        |  
| `utility`          | Data preprocessing and dimensionality reduction |  
| `metric`           | Evaluation metrics                              |  
| `math`             | Mathematical utilities                          |  
| `dataset`          | Standard datasets                               | 
| `default`          | Enables `machine_learning` and `neural_network` |
| `full`             | Enables all features                            |
| `show_progress`    | Show progress bars when training                |

## Project Status
RustyML is in active development. While the API is stabilizing, breaking changes may occur in minor version updates until version 1.0.0.

## Contribution
Contributions are welcome! If you're interested in helping build a robust machine learning ecosystem in Rust, please feel free to:
1. Submit issues for bugs or feature requests
2. Create pull requests for improvements 
3. Provide feedback on the API design 
4. Help with documentation and examples 

## Authors
SomeB1oody – [stanyin64@gmail.com](mailto:stanyin64@gmail.com)

## License
Licensed under the [MIT License](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE). See the LICENSE file for details.