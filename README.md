[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

A high-performance machine learning and deep learning library written in **pure Rust**.

[![Rust Version](https://img.shields.io/badge/rustc-1.89%2B-brown)](https://www.rust-lang.org/)
[![Edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)
[![docs.rs](https://img.shields.io/docsrs/rustyml)](https://docs.rs/rustyml)

## Overview

RustyML is a complete ecosystem for machine learning and deep learning, built end to end in
Rust with no C/C++ dependencies. It covers the full workflow — from data preprocessing and
feature engineering, through model training, to evaluation — while leaning on Rust's memory
safety, fearless concurrency, and zero-cost abstractions.

Everything is organized into six feature-gated modules, so you compile only what you use:
`machine_learning`, `neural_network`, `utils`, `metrics`, `math`, and a shared `prelude`.

## Highlights

- **Pure Rust, no FFI** — memory-safe and portable, with nothing to link against.
- **Parallelized by default** — heavy kernels use [Rayon](https://github.com/rayon-rs/rayon) for multi-threaded computation.
- **Broad algorithm coverage** — classical supervised/unsupervised learning, anomaly detection, and a full neural-network framework.
- **Unified, structured error handling** — every fallible call returns `RustymlResult<T>`; errors are grouped into clear category variants instead of opaque strings.
- **Reproducible by design** — a single `set_global_seed` call makes every randomized component deterministic.
- **Model persistence** — save and load trained models and network weights as JSON via [Serde](https://serde.rs/).
- **Rich evaluation metrics** — regression, classification (binary & multiclass), and clustering, mirroring scikit-learn conventions.
- **Modular features** — pull in just `metrics`, just `math`, the `default` learning stack, or the `full` crate.

## Installation

Add RustyML to your `Cargo.toml`:

```toml
[dependencies]
rustyml = { version = "*", features = ["full"] }
ndarray = "0.17"
```

Pick the feature set that fits your needs:

```toml
# Default: classical ML + neural networks
rustyml = "*"

# Just the neural-network framework
rustyml = { version = "*", features = ["neural_network"] }

# Everything (ml, nn, utils, metrics, math)
rustyml = { version = "*", features = ["full"] }

# Show training progress bars in the terminal
rustyml = { version = "*", features = ["full", "show_progress"] }
```

> **MSRV:** Rust 1.89+ (edition 2024).

## Quick Start

### Classical Machine Learning

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

// Train a regularization-free linear regression model
let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();

let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
let y = array![6.0, 9.0, 12.0];

model.fit(&x, &y).unwrap();
let predictions = model.predict(&x).unwrap();
println!("{:?}", predictions);

// Persist and reload the trained model
model.save_to_path("linear_regression.json").unwrap();
let restored = LinearRegression::load_from_path("linear_regression.json").unwrap();
```

### Neural Networks

```rust
use rustyml::neural_network::sequential::Sequential;
use rustyml::prelude::neural_network::*;
use ndarray::Array;

// 32 samples, 784 input features, 10 output classes
let x = Array::ones((32, 784)).into_dyn();
let y = Array::ones((32, 10)).into_dyn();

let mut model = Sequential::new();
model
    .add(Dense::new(784, 128, Activation::ReLU, None).unwrap())
    .add(Dense::new(128, 64, Activation::ReLU, None).unwrap())
    .add(Dense::new(64, 10, Activation::Softmax, None).unwrap())
    .compile(
        Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap(),
        CategoricalCrossEntropy::new(),
    );

model.summary(); // print the architecture
model.fit(&x, &y, 10).unwrap();

let predictions = model.predict(&x).unwrap();
println!("Predictions shape: {:?}", predictions.shape());

// Save the trained weights, then load them into a fresh model
model.save_to_path("model.json").unwrap();
```

### Evaluating a Model

```rust
use rustyml::metrics::*;
use ndarray::array;

// Arguments are always (y_true, y_pred), matching scikit-learn
let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
println!("Accuracy: {:.3}", cm.accuracy());
println!("F1 score: {:.3}", cm.f1_score());
```

## Modules

### `machine_learning`

Classical supervised and unsupervised algorithms, all with parallel-processing optimization,
input validation, and JSON persistence.

| Category | Algorithms |
|----------|------------|
| **Regression** | Linear Regression (optional L1/L2 regularization) |
| **Classification** | Logistic Regression, K-Nearest Neighbors, Decision Tree (ID3 / C4.5 / CART), SVC (kernel SMO), Linear SVC, Linear Discriminant Analysis |
| **Clustering** | KMeans (K-means++ init), DBSCAN, MeanShift |
| **Anomaly Detection** | Isolation Forest |

Shared config types live in [`types`](https://docs.rs/rustyml/latest/rustyml/types/index.html):
`DistanceCalculationMetric` (Euclidean / Manhattan / Minkowski), `RegularizationType` (L1 / L2),
and `KernelType` (Linear / Poly / RBF / Sigmoid / Cosine). Models implement the unified `Fit`
and `Predict` traits.

### `neural_network`

A complete framework for building, training, and serializing feed-forward and
convolutional/recurrent networks via a Keras-style `Sequential` API.

- **Core layers** — `Dense`, `Flatten`
- **Activations** — `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Linear` (as the `Activation` enum or standalone layers)
- **Convolution** — `Conv1D`, `Conv2D`, `Conv3D`, `DepthwiseConv2D`, `SeparableConv2D`
- **Pooling** — Max / Average pooling for 1D, 2D, 3D, plus their global variants
- **Recurrent** — `SimpleRNN`, `LSTM`, `GRU`
- **Regularization** — `Dropout`, `SpatialDropout{1,2,3}D`, `GaussianNoise`, `GaussianDropout`
- **Normalization** — `BatchNormalization`, `LayerNormalization`, `InstanceNormalization`, `GroupNormalization`
- **Optimizers** — `SGD` (with momentum), `Adam`, `RMSprop`, `AdaGrad`
- **Losses** — `MeanSquaredError`, `MeanAbsoluteError`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`

Training supports full-batch (`fit`) and mini-batch (`fit_with_batches`) loops, weight
inspection (`get_weights`), and JSON serialization (`save_to_path` / `load_from_path`).

### `utils`

Data preprocessing and dimensionality reduction.

- **Dimensionality reduction** — `PCA` (multiple SVD solvers), `KernelPCA` (RBF / Linear / Poly / Sigmoid / Cosine kernels), `TSNE`
- **Scaling** — `standardize` (z-score), `normalize` (configurable axis & order)
- **Label encoding** — `to_categorical`, `to_categorical_with_mapping`, `to_sparse_categorical`
- **Splitting** — `train_test_split` with a configurable ratio

### `metrics`

A broad evaluation suite. All functions take `(y_true, y_pred)` and panic on precondition
violations (mismatched lengths, empty input) rather than returning `Result`, keeping this leaf
module dependency-light.

- **Regression** — MSE, RMSE, MAE, median absolute error, MAPE, R², explained variance
- **Classification** — accuracy, `ConfusionMatrix` & `MulticlassConfusionMatrix`, ROC AUC, log loss, Cohen's κ, top-k accuracy, average precision, ROC & precision-recall curves
- **Clustering** — Adjusted Rand Index, Normalized / Adjusted Mutual Information, homogeneity / completeness / V-measure, Fowlkes–Mallows, silhouette, Davies–Bouldin, Calinski–Harabasz

### `math`

Pure, stateless numerical primitives shared across the crate: impurity measures (`entropy`,
`gini`), distances (`squared_euclidean_distance_row`, `manhattan_distance_row`,
`minkowski_distance_row`), statistics (`variance`, `standard_deviation`, `sum_of_square_total`,
`sum_of_squared_errors`), and activation/loss helpers (`sigmoid`, `logistic_loss`,
`hinge_loss`).

### `prelude`

One-stop imports, split by domain so you only pull in what you need:

```rust
use rustyml::prelude::machine_learning::*; // ML models, traits, config enums
use rustyml::prelude::neural_network::*;   // layers, optimizers, losses
use rustyml::prelude::utils::*;            // PCA, t-SNE, scaling, splitting
use rustyml::prelude::metrics::*;          // evaluation metrics
use rustyml::prelude::math::*;             // math primitives
```

## Feature Flags

The crate uses feature flags for modular compilation:

| Feature | Description |
|---------|-------------|
| `machine_learning` | Classical ML algorithms (enables `math`) |
| `neural_network` | Neural-network framework |
| `utils` | Data preprocessing and dimensionality reduction (enables `math`) |
| `metrics` | Evaluation metrics (enables `math`) |
| `math` | Mathematical and statistical primitives |
| `default` | `machine_learning` + `neural_network` |
| `full` | All of the above modules |
| `show_progress` | Render training/iteration progress bars in the terminal |

## Reproducibility

Every randomized component (weight initialization, K-means++, Isolation Forest, t-SNE, dropout,
…) resolves its `random_state: Option<u64>` against a shared entry point. Set one global seed and
the whole crate becomes deterministic:

```rust
use rustyml::set_global_seed;

set_global_seed(42);
// ... train models; results are now reproducible across runs ...
```

A per-call `random_state` takes precedence over the global seed, which in turn takes precedence
over system entropy. See the [`random`](https://docs.rs/rustyml/latest/rustyml/random/index.html)
module for the full resolution rules.

## Error Handling

Outside the `metrics` and `math` leaf modules, every fallible operation returns
`RustymlResult<T>` (an alias for `Result<T, rustyml::error::Error>`). The `Error` type is structured into
category variants and groups domain-specific failures into nested `NnError`, `TreeError`, and
`IoError` sub-enums, so you can match on what actually went wrong instead of parsing strings.

## Project Status

RustyML is under active development. The API is stabilizing, but breaking changes may still land
in minor releases before `1.0.0`.

## Contributing

Contributions are welcome! If you'd like to help build a robust ML ecosystem in Rust, you can:

1. Open issues for bugs or feature requests
2. Submit pull requests for improvements
3. Share feedback on the API design
4. Improve the documentation and examples

Please also review the [Code of Conduct](https://github.com/SomeB1oody/RustyML/blob/master/CODE_OF_CONDUCT.md).

## Author

SomeB1oody — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)

## License

Licensed under the [MIT License](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE). See the LICENSE file for details.
