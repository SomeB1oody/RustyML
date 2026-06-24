//! Neural network primitives: layers, loss functions, optimizers, the sequential
//! model, and the traits that tie them together
//!
//! A framework for constructing, training, and deploying feed-forward neural networks. Layers,
//! optimizers, and losses live in the [`layers`](crate::neural_network::layers),
//! [`optimizers`](crate::neural_network::optimizers), and [`losses`](crate::neural_network::losses)
//! submodules; [`Sequential`](crate::neural_network::sequential::Sequential) stacks layers into a
//! trainable model, and the shared interfaces live in [`traits`](crate::neural_network::traits).
//! Every tensor flowing through the framework is a [`Tensor`](crate::neural_network::Tensor) (`f32`
//! n-dimensional array)
//!
//! # Core components
//!
//! ## Layers
//! - **Dense**: fully connected layer with a configurable activation
//! - **Activation**: standalone activation layers (ReLU, Sigmoid, Tanh, Softmax, Linear, ...)
//! - **Convolution**: 1D/2D/3D convolution, plus depthwise and separable variants
//! - **Pooling**: max / average pooling and their global variants for 1D, 2D, and 3D
//! - **Recurrent**: SimpleRNN, LSTM, and GRU sequence layers
//! - **Regularization**: dropout (incl. spatial), noise injection, and normalization layers
//!
//! ## Optimizers
//! - **SGD**: stochastic gradient descent with momentum
//! - **Adam**: adaptive moment estimation (coupled L2 weight decay)
//! - **AdamW**: Adam with decoupled weight decay
//! - **RMSProp**: root-mean-square propagation
//! - **AdaGrad**: adaptive gradient
//!
//! ## Loss functions
//! - **MeanSquaredError** / **MeanAbsoluteError**: regression
//! - **BinaryCrossEntropy**: binary classification
//! - **CategoricalCrossEntropy** / **SparseCategoricalCrossEntropy**: multi-class classification
//!
//! ## Model
//! - [`Sequential`](crate::neural_network::sequential::Sequential): a linear stack of layers with an
//!   integrated training loop, prediction, and weight save/load
//!
//! # Examples
//!
//! ```rust
//! use rustyml::neural_network::{
//!     sequential::Sequential,
//!     layers::{Activation, Dense},
//!     optimizers::Adam,
//!     losses::MeanSquaredError,
//! };
//! use ndarray::Array;
//!
//! // Create input and target tensors
//! let x = Array::ones((2, 4)).into_dyn();  // 2 samples, 4 features
//! let y = Array::ones((2, 1)).into_dyn();  // 2 samples, 1 output
//!
//! // Build sequential model
//! let mut model = Sequential::new();
//! model.add(Dense::new(4, 8, Activation::ReLU).unwrap())   // Input layer: 4 -> 8
//!      .add(Dense::new(8, 3, Activation::ReLU).unwrap())   // Hidden layer: 8 -> 3
//!      .add(Dense::new(3, 1, Activation::Linear).unwrap()); // Output layer: 3 -> 1
//!
//! // Compile with optimizer and loss function
//! model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(), MeanSquaredError::new());
//!
//! // Display model architecture
//! model.summary();
//!
//! // Train the model
//! model.fit(&x, &y, 100).unwrap();
//!
//! // Make predictions
//! let predictions = model.predict(&x);
//! ```

use ndarray::ArrayD;

/// N-dimensional array used as a tensor in the neural network
pub type Tensor = ArrayD<f32>;

/// Neural network layer implementations
pub mod layers;
/// Loss function implementations
pub mod losses;
/// Optimization algorithms for neural network training
pub mod optimizers;
/// Sequential model architecture
pub mod sequential;
/// Trait interfaces for neural network models
pub mod traits;
