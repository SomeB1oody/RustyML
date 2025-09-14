# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [v0.6.3] - 2025-8-30 (UTC-7)
### Changed
- Standardize doc comments

## [v0.6.3] - 2025-8-30 (UTC-7)
### Changed
- Improve input validation and error handling in `Sequential` model

## [v0.6.3] - 2025-8-30 (UTC-7)
### Changed
- Improve input validation and error handling across mathematical utilities
- Enhance input validation, edge case handling, and error reporting across clustering and classification algorithms

## [v0.6.3] - 2025-8-23 (UTC-7)
### Added
- Add doc comments for `SeparableConv2D` and `DepthwiseConv2D` layer weights
- Add lifetime parameter to `get_weights` return type across layers

### Changed
- Update dependencies(`rand`, `rayon`, and `nalgebra`)

## [v0.6.2] - 2025-6-5 (UTC-7)
### Added
- Refactor activation handling for convolutional layers

## [v0.6.2] - 2025-6-4 (UTC-7)
### Added
- Add `DepthwiseConv2D` layer and related utilities

## [v0.6.2] - 2025-6-3 (UTC-7)
### Added
- Add `GlobalMaxPooling3D` layer and tests
- Add `GlobalAveragePooling3D` layer and tests
- Add input dimensionality checks for pooling layers
- Add `Conv3D` layer and optimizer support
- Add input dimensionality checks for `Conv1D` and `Conv2D` layers
- Add `SeparableConv2D` layer and related utilities

### Changed
- Refactor to use `layer_functions_global_pooling` macro
- Separate `layer_weight` and `padding_type` into dedicated modules
- Add support for Flatten layer with 3D, 4D, and 5D tensors
- Replace `HashMap` and `HashSet` with `AHashMap` and `AHashSet`

## [v0.6.2] - 2025-6-2 (UTC-7)
### Added
- Add `AveragePooling3D` layer and tests

### Changed
- Refactor pooling layers to use macros for output shape calculation

## [v0.6.2] - 2025-6-1 (UTC-7)
### Changed
- Refactor layers without trainable parameters to use `no_trainable_parameters_layer_functions!` macro

## [v0.6.2] - 2025-5-31 (UTC-7)
### Added
- Add `MaxPooling3D` layer and tests

## [v0.6.2] - 2025-5-30 (UTC-7)
### Added
- Add `Conv1D` layer implementation and tests

### Changed
- Refactor tests to reuse `generate_data` function for pooling layers
- Rename `OptimizerCacheFEL` to `OptimizerCacheConv2D`
- Extracted the SGD parameter update logic into a `update_sgd_conv` macro to eliminate redundancy. Updated `Conv1D` and `Conv2D` layers to use the macro.
- Extracted the Adam parameter update logic into a `update_adam_conv` function to eliminate redundancy. Updated `Conv1D` and `Conv2D` layers to use the new method for weight and bias updates.

## [v0.6.2] - 2025-5-29 (UTC-7)
### Added
- Add `GlobalAveragePooling1D` layer implementation

## [v0.6.2] - 2025-5-28 (UTC-7)
### Added
- Add `GlobalMaxPooling1D` layer and corresponding tests

### Changed
- Change function `preliminary_check` from public into private
- PCA no longer requires `preliminary_check` function and integrates input validation functionality

## [v0.6.2] - 2025-5-27 (UTC-7)
### Added
- Add tests for `MaxPooling1D` layer
- Add more doc comments for `MaxPooling1D` layer

### Changed
- Introduced a shared `compute_output_shape` function to streamline output shape calculations across pooling layers.
- Centralized 1D and 2D pooling output shape logic into reusable helper functions (`calculate_output_shape_1d_pooling` and `calculate_output_shape_2d_pooling`)

## [v0.6.2] - 2025-5-26 (UTC-7)
### Added
- Add `MaxPooling1D` layer implementation

## [v0.6.2] - 2025-5-25 (UTC-7)
### Added
- Add more test functions for neural network

### Changed
- move all test functions to `test` module

## [v0.6.2] - 2025-5-24 (UTC-7)
### Changed
- Refactor and document `AveragePooling1D` module

## [v0.6.2] - 2025-5-23 (UTC-7)
### Added
- Add `AveragePooling1D` layer and corresponding tests

## [v0.6.1] - 2025-5-22 (UTC-7)
### Added
- Add doc comments for modules

### Changed
- Optimize parameter updates of struct `Conv2D` with parallelization

## [v0.6.1] - 2025-5-21 (UTC-7)
### Added
- Add tests for `GlobalAveragePooling2D` layer

### Changed
- Refactor `GlobalAveragePooling2D` with improved comments and docs
- Refactor global max pooling to leverage parallel processing

## [v0.6.1] - 2025-5-20 (UTC-7)
### Added
- Add `GlobalAveragePooling2D` layer implementation

## [v0.6.1] - 2025-5-19 (UTC-7)
### Added
- Add comprehensive weight structs for neural network layers

## [v0.6.1] - 2025-5-18 (UTC-7)
### Changed
- Refactor `GlobalMaxPooling2D` with improved comments and docs

## [v0.6.1] - 2025-5-17 (UTC-7)
### Added
- Add `GlobalMaxPooling2D` layer initial implementation

## [v0.6.1] - 2025-5-16 (UTC-7)
### Changed
- Rename `AveragePooling` to `AveragePooling2D` for clarity

## [v0.6.1] - 2025-5-15 (UTC-7)
### Added
- Add detailed comments and example to `AveragePooling` layer

## [v0.6.1] - 2025-5-14 (UTC-7)
### Added
- Add `AveragePooling` layer implementation and corresponding tests

## [v0.6.1] - 2025-5-13 (UTC-7)
### Added
- Update comments and documentation in `Flatten` layer

## [v0.6.1] - 2025-5-12 (UTC-7)
### Added
- Add `Flatten` layer initial implementation and associated tests

## [v0.6.1] - 2025-5-11 (UTC-7)
### Added
- Add detailed documentation and usage example for `MaxPooling2D`

## [v0.6.1] - 2025-5-10 (UTC-7)
### Added
- Add `MaxPooling2D` layer initial implementation

## [v0.6.1] - 2025-5-9 (UTC-7)
### Changed
- Rename `OptimizerCacheFEX` to `OptimizerCacheFEL` (FEL stands for feature extraction layer)

## [v0.6.1] - 2025-5-8 (UTC-7)
### Added
- Add complete test function for `Conv2D`
- Add comprehensive docstrings for `Conv2D` layer and methods

## [v0.6.1] - 2025-5-7 (UTC-7)
### Added
- Add `Debug`, `Clone`, and `Default` traits to optimizer structs

### Changed
- Use parallelized computation for performance improvement
- Replaced multiple optimizer-specific fields with a unified `optimizer_cache` structure
- Refactor optimizer cache initialization and parameter flattening
- Refactor optimizer caching to support feature extraction layers
- Refactor SGD parameter updates with parallelized helper methods

## [v0.6.1] - 2025-5-6 (UTC-7)
### Added
- Add Conv2D layer(initial implementation) support to neural_network module

### Changed
- Rename layer naming convention in Sequential model

### Removed
- Remove some default implementations in `Layer` trait

## [v0.6.0] - 2025-5-5 (UTC-7)
### Added
- Add `get_weights` method across layers and `LayerWeight` enum

## [v0.6.0] - 2025-5-4 (UTC-7)
### Changed
- Refactor optimizer state handling with unified cache

## [v0.6.0] - 2025-5-3 (UTC-7)
### Added
- Ensure `fit` validates optimizer and layers before training

### Changed
- Refactor parameter updates to use parallel processing

### Removed
- Remove method `update_parameters` in `Layer` trait
- Remove getter methods from Dense layer

## [v0.6.0] - 2025-5-2 (UTC-7)
### Changed
- Refactor RMSprop implementation with unified cache structure

## [v0.6.0] - 2025-5-1 (UTC-7)
### Added
- Add detailed documentation and examples to neural network layers

### Changed
- Refactor layers to enforce explicit activation usage
- Optimize LSTM computations with parallel processing using Rayon
- Refactor LSTM to consolidate gate logic into reusable structures
- Refactor Adam optimizer state management into `AdamStates`

## [v0.6.0] - 2025-4-30 (UTC-7)
### Added
- Add doc comment for `LSTM`

### Removed
- Remove redundant method documentation comments

## [v0.6.0] - 2025-4-29 (UTC-7)
### Added
- Add LSTM layer initial implementation and associated tests

## [v0.6.0] - 2025-4-28 (UTC-7)
### Changed
Refactor traits in neural_network module definitions into traits module

## [v0.6.0] - 2025-4-27 (UTC-7)
### Added
- Add doc comment for traits module

## [v0.6.0] - 2025-4-26 (UTC-7)
### Changed
- Refactor `LinearSVC` to use `RegularizationType` for penalties

## [v0.6.0] - 2025-4-25 (UTC-7)
### Changed
- Refactor regressors to use shared `RegressorCommonGetterFunctions` trait

## [v0.6.0] - 2025-4-24 (UTC-7)
### Added
- Add regularization support to linear and logistic regression

### Changed
- Refactor `KernelType` into machine_learning module

## [v0.5.1] - 2025-4-23 (UTC-7)
### Changed
- Modularize activation functions into a separate module

## [v0.5.1] - 2025-4-22 (UTC-7)
### Changed
- Optimize activation functions with parallel processing

## [v0.5.1] - 2025-4-21 (UTC-7)
### Changed
- Refactor optimizer implementations into separate files

## [v0.5.1] - 2025-4-20 (UTC-7)
### Changed
- Refactor loss functions into separate modules

## [v0.5.1] - 2025-4-19 (UTC-7)
### Added
- Add doc comments for layer `SimpleRNN`

## [v0.5.1] - 2025-4-18 (UTC-7)
### Changed
- Modularize `layer` into separate `dense` and `simple_rnn` modules

## [v0.5.1] - 2025-4-17 (UTC-7)
### Added
- Add `SimpleRNN` layer initial implementation

### Changed
- Use rand v0.9.1

## [v0.5.1] - 2025-4-16 (UTC-7)
### Changed
- Optimize neural network computations with parallelism

## [v0.5.1] - 2025-4-15 (UTC-7)
### Changed
- `ModelError` is used in the neural network implementation

## [v0.5.1] - 2025-4-14 (UTC-7)
### Changed
- Standardize documentation comments across modules

## [v0.5.0] - 2025-4-13 (UTC-7)
### Changed
- Replace `ndarray-linalg` with `nalgebra` for `PCA`, `LDA`, and `KernelPCA`

## [v0.5.0] - 2025-4-12 (UTC-7)
### Changed
- Refactor metrics API to remove `Result` usage and add panics

## [v0.5.0] - 2025-4-11 (UTC-7)
### Added
- Added getter methods for accessing key properties

### Changed
- Replaced public fields with private ones in `Dense`, `Adam`, and `RMSprop` structs to improve encapsulation.

## [v0.5.0] - 2025-4-10 (UTC-7)
### Added
- Add activation functions support to `Dense` layer

## [v0.4.0] - 2025-4-9 (UTC-7)
### Added
- Add RMSprop optimizer support to the neural network
- Add MAE(Mean Absolute Error) support in loss function computation
- Add doc comments for neural_network module

## [v0.4.0] - 2025-4-8 (UTC-7)
### Added
- Add `Adam` optimizer for neural network
- Added new loss functions: `CategoricalCrossEntropy` and `SparseCategoricalCrossEntropy`

### Changed
- Replaced slices with `ndarray`'s `ArrayView` to improve consistency and compatibility with numerical operations in metric module
- Refactor polynomial feature generation to exclude constant term

## [v0.4.0] - 2025-4-7 (UTC-7)
### Added
- Add neural network support(initial implementation)

## [v0.3.0] - 2025-4-6 (UTC-7)
### Added
- Add dataset module and put iris and diabetes datasets in it
- Enhance SVM/LDA documentation

## [v0.3.0] - 2025-4-5 (UTC-7)
### Changed
- Refactor to use `ArrayView` for memory-efficient data handling
- Refactor outputs to use `Array` instead of `Vec` in DBSCAN and KNN

## [v0.2.1] - 2025-4-4 (UTC-7)
### Added
- Add variance calculation in math module
- Add MSE calculation in metric module
- Print info after training completes for `fit` functions of struct `LDA`, `SVC`, `LinearSVC` and `PCA`
- Integrate Rayon for parallel computation across modules (**7 s faster** in `cargo test`!!!)

### Changed
- Refactor functions in math module and metric module to use `ArrayView1` for improved efficiency

### Removed
- Remove MSE calculation(named mean_squared_error, but actually calculate variance) in math module

## [v0.2.1] - 2025-4-3 (UTC-7)
### Added
- Add t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation

### Changed
- Remove gaussian kernel calculation function, put gaussian kernel calculation directly in function `fit` of struct `MeanShift`

## [v0.2.1] - 2025-4-2 (UTC-7)
### Added
- Add LinearSVC support
- Add KernelPCA support
- Add `ProcessingError(String)` to `crate::ModelError` 
- Add LDA(Linear Discriminant Analysis) support

### Changed
- Change the location of function `standardize` from `crate::utility::principal_component_analysis` to `crate::utility`

## [v0.2.1] - 2025-4-1 (UTC-7)
### Added
- Add SVC(Support Vector Classification) support

## [v0.2.0] - 2025-4-1 (UTC-7)
### Added
- Add `train_test_split` function in utility module to split dataset for training and dataset for test
- Add function `normalized_mutual_info` and `adjusted_mutual_info` to metric module to calculate NMI and AMI info
- Add AUC-ROC value calculation in metric module

## [v0.2.0] - 2025-3-31 (UTC-7)
### Changed
- Change `principal_component_analysis` module to `utility` module, change `principal_component_analysis_test` module to `utility_test` module
- Keep the algorithm functions in the math module, and move the functions that evaluate the model's performance (such as R-square values) and structures (confusion matrices) to the metric module. Some of them are used in both ways, then keep them in both modules.
- Change the output of some of the functions in math module and metric module from `T` to `Result<T, crate::ModelError>`

## [v0.1.1] - 2025-3-31 (UTC-7)
### Added
- Add function `preliminary_check` in machine_learning module to performs validation checks on the input data matrices
- Add confusion matrix in math module

### Changed
- Change type of field `coefficients` of struct `LinearRegression` from `Option<Vec<f64>>` to `Option<Array1<f64>>`
- Change the output of some methods of struct `LinearRegression` from `Vec<f64>` to `Array1<f64>`
- Change variant `InputValidationError` of enum type `ModelError` from `InputValidationError(&str)` to `InputValidationError(String)`

## [v0.1.0] - 2025-3-30 (UTC-7)
### Added
- Add function `fit_predict` for some models
- Add examples for functions in math.rs
- Add input validation
- Add doc comments for machine learning modules
- Add prelude module(all re-exports are there)

### Changed
- Change input types of function `fit`, `predict` and `fit_predict` to `Array1` and `Array2`
- Rename the crate from `rust_ai` to `rustyml`
- Change the output of function `fit` from `&mut Self` to `Result<&mut Self, ModelError>` or `Result<&mut Self, Box<dyn std::error::Error>>`

## [v0.1.0] - 2025-3-29 (UTC-7)
### Added
- Add function `generate_tree_structure` for `DecisionTree` to generate tree structure as string
- Add isolation forest implementation
- Add PCA(Principal Component Analysis) implementation
- Add function `standard_deviation` in math module to calculates the standard deviation of a set of values

## [v0.1.0] - 2025-3-28 (UTC-7)
### Added
- Add Decision Tree model
- Add following functions to math.rs:
    - `entropy`: Calculates the entropy of a label set
    - `gini`: Calculates the Gini impurity of a label set
    - `information_gain`: Calculates the information gain when splitting a dataset
    - `gain_ratio`: Calculates the gain ratio for a dataset split
    - `mean_squared_error`: Calculates the Mean Squared Error (MSE) of a set of values

### Changed
- Replaced string-based distance calculation method options with an enum `crate::machine_learning::DistanceCalculation`
- For KNN model: replaced string-based weight function options with an enum `crate::machine_learning::knn::WeightingStrategy`
- For decision tree: replaced string-based algorithm options with an enum `crate::machine_learning::decision_tree::Algorithm`

## [v0.1.0] - 2025-3-27 (UTC-7)
### Added
- Add changelog.md to record updates
- Add DBSCAN model
- Add function `fit_predict` to fit and predict in one step
- Add doc comments to tell user `p` value of function `minkowski_distance` in model is always 3

## [v0.1.0] - 2025-3-26 (UTC-7)
### Added
- Add "xx model converged at iteration x, cost: x" when finishing `fit`
- Add description for `n_iter` field
- Add getter functions for `KMeans`
- implement `Default` trait for `KMeans`

### Changed
- Rename `max_iteration` and `tolerance` to `max_iter` and `tol`
- Change doc comments to enhanced consistency

### Removed
- Remove examples in math.rs(add them back later)

## [v0.1.0] - 2025-3-25 (UTC-7)
### Added
- Add MeanShift model
- Add `InputValidationError` in `ModelError`, indicating the input data provided  does not meet the expected format, type, or validation rules
- Add `gaussian_kernel` in math module, calculate the Gaussian kernel (RBF kernel)

### Changed
- Change the output of all `predict` functions(except KNN) from `T` to `Result<T, crate::ModelError>`
- Correct doc comments