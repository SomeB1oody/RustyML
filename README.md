[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

A high-performance machine learning and deep learning library written in **pure Rust**.

[![rustc](https://img.shields.io/badge/rustc-1.89%2B-brown)](https://www.rust-lang.org/)
[![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)

[![fmt](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/fmt.yml?branch=master&label=fmt)](https://github.com/SomeB1oody/RustyML/actions/workflows/fmt.yml)
[![clippy](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/clippy.yml?branch=master&label=clippy)](https://github.com/SomeB1oody/RustyML/actions/workflows/clippy.yml)
[![test](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/test.yml?branch=master&label=test)](https://github.com/SomeB1oody/RustyML/actions/workflows/test.yml)
[![doc](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/doc.yml?branch=master&label=doc)](https://github.com/SomeB1oody/RustyML/actions/workflows/doc.yml)

> 📖 **[RustyML User Guide](https://someb1oody.github.io/RustyML/en/)** — a hands-on book covering every module, from your first model to performance tuning.

## Overview

RustyML is a complete ecosystem for machine learning and deep learning, built end to end in
Rust with no C/C++ dependencies. It covers the full workflow — from data preprocessing and
feature engineering, through model training, to evaluation — while leaning on Rust's memory
safety, fearless concurrency, and zero-cost abstractions.

Everything is organized into five feature-gated modules, so you compile only what you use:
`machine_learning`, `neural_network`, `utils`, `metrics`, and `math`, plus a shared `prelude`.

## Highlights

- **Pure Rust, no FFI** — memory-safe and portable, with nothing to link against.
- **Parallelized by default** — heavy kernels use [Rayon](https://github.com/rayon-rs/rayon) for multi-threaded computation.
- **Broad algorithm coverage** — classical supervised/unsupervised learning, anomaly detection, and a full neural-network framework.
- **Unified, structured error handling** — every fallible call returns `RustymlResult<T>`; errors are grouped into clear category variants instead of opaque strings.
- **Reproducible by design** — a single `set_global_seed` call makes every randomized component on the calling thread deterministic; per-component `random_state` covers the rest.
- **Model persistence** — save and load trained models and network weights as compact binary via [Serde](https://serde.rs/) and [postcard](https://docs.rs/postcard/).
- **Rich evaluation metrics** — regression, classification (binary & multiclass), and clustering, mirroring scikit-learn conventions.
- **Checked against scikit-learn** — estimator defaults, score signs, and metric output conventions are verified numerically against scikit-learn 1.9, so a ported pipeline gives the same answers. Deliberate departures are documented where they occur.
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
let mut model = LinearRegression::new(true)
    .with_solver(LeastSquaresSolver::GradientDescent { learning_rate: 0.01, max_iter: 1000, tol: 1e-6 }).unwrap();

let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
let y = array![6.0, 9.0, 12.0];

model.fit(&x, &y).unwrap();
let predictions = model.predict(&x).unwrap();
println!("{:?}", predictions);

// Persist and reload the trained model
model.save_to_path("linear_regression.bin").unwrap();
let restored = LinearRegression::load_from_path("linear_regression.bin").unwrap();
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
    .add(Dense::new(784, 128, Activation::ReLU).unwrap())
    .add(Dense::new(128, 64, Activation::ReLU).unwrap())
    .add(Dense::new(64, 10, Activation::Softmax).unwrap())
    .compile(
        Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        CategoricalCrossEntropy::new(false),
    );

model.summary(); // print the architecture
model.fit(&x, &y, 10).unwrap();

let predictions = model.predict(&x).unwrap();
println!("Predictions shape: {:?}", predictions.shape());

// Save the trained weights, then load them into a fresh model
model.save_to_path("model.bin").unwrap();
```

### Evaluating a Model

```rust
use rustyml::metrics::*;
use ndarray::array;

// Arguments are always (y_true, y_pred), matching scikit-learn
// ConfusionMatrix::new takes hard 0.0/1.0 labels (new_with_labels covers other pairs)
let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

// The two arguments carry independent storage types, so an owned array and a view mix
let cm = ConfusionMatrix::new(&y_true, &y_pred.view());
println!("Accuracy: {:.3}", cm.accuracy());
println!("F1 score: {:.3}", cm.f1_score());
```

## Modules

### `machine_learning`

Classical supervised and unsupervised algorithms, all with parallel-processing optimization,
input validation, and binary persistence.

| Category | Algorithms |
|----------|------------|
| **Regression** | Linear Regression (closed-form OLS by default, or gradient descent; optional L1/L2 regularization) |
| **Classification** | Logistic Regression, K-Nearest Neighbors, Decision Tree (ID3 / C4.5 / CART), SVC (kernel SMO), Linear SVC, Linear Discriminant Analysis |
| **Clustering** | KMeans (K-means++ init, `n_init` restarts), DBSCAN, MeanShift (flat kernel) |
| **Dimensionality Reduction** | PCA (multiple SVD solvers), KernelPCA (RBF / Linear / Poly / Sigmoid / Cosine kernels), t-SNE |
| **Anomaly Detection** | Isolation Forest |

All three clustering estimators return `Array1<isize>` labels with `-1` for noise or unassigned points,
which is also what every clustering metric consumes — so any of them feeds any metric directly.
`IsolationForest` follows scikit-learn's sign convention: `score_samples` returns values in `[-1, 0)`
where lower is more anomalous, and `predict` returns `-1` (outlier) / `+1` (inlier).

Shared config types live in [`machine_learning::types`](https://docs.rs/rustyml/latest/rustyml/machine_learning/types/index.html):
`RegularizationType` (L1 / L2), `Gamma`, and `KernelType` (Linear / Poly / RBF / Sigmoid / Cosine).
`RegularizationType`'s documentation carries a conversion table for scikit-learn's penalty strengths
(`Lasso`/`Ridge` `alpha`, `LogisticRegression` `C`); L1 is applied as a proximal soft-threshold, so
penalized coefficients reach exactly zero.
The `DistanceCalculationMetric` (Euclidean / Manhattan / Minkowski) dispatcher lives in
[`math`](https://docs.rs/rustyml/latest/rustyml/math/index.html) and is re-exported at the
`machine_learning` root. Predictive models implement the
unified `Fit` and `Predict` traits; the dimensionality-reduction transformers
([`decomposition`](https://docs.rs/rustyml/latest/rustyml/machine_learning/decomposition/index.html)
and [`manifold`](https://docs.rs/rustyml/latest/rustyml/machine_learning/manifold/index.html))
implement `Transform` / `FitTransform`. All four traits live at the crate root in
[`traits`](https://docs.rs/rustyml/latest/rustyml/traits/index.html), shared with the stateful
preprocessing transformers in `utils`.

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
- **Optimizers** — `SGD` (with momentum), `Adam`, `AdamW`, `RMSprop`, `AdaGrad`
- **Losses** — `MeanSquaredError`, `MeanAbsoluteError`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`

Training supports full-batch (`fit`) and mini-batch (`fit_with_batches`) loops, weight
inspection (`get_weights`), and binary serialization (`save_to_path` / `load_from_path`).

### `utils`

Data preprocessing and dataset splitting. (Dimensionality reduction — `PCA`, `KernelPCA`,
`TSNE` — now lives in `machine_learning` under `decomposition` and `manifold`.)

- **Scaling (one-shot)** — `standardize` (z-score), `normalize` (configurable axis & order)
- **Scaling (stateful)** — `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`
  (median / IQR, outlier-resistant), and `Normalizer`: scikit-learn-style transformers with
  `fit` / `transform` / `fit_transform` / `inverse_transform` (and `partial_fit` on all but
  `RobustScaler`), storing the training statistics so later batches go through the same map,
  persistable with `save_to_path`
- **Label encoding** — `to_categorical`, `to_categorical_with_mapping`, `to_sparse_categorical`
- **Splitting** — `train_test_split` and `train_test_split_stratified`, with a configurable ratio

### `metrics`

A broad evaluation suite. All functions take `(y_true, y_pred)` and panic on precondition
violations (mismatched lengths, empty input, an out-of-domain label) rather than returning `Result`,
keeping this leaf module dependency-light.

- **Regression** — MSE, RMSE, MAE, median absolute error, MAPE, R², explained variance
- **Classification** — accuracy, `ConfusionMatrix` & `MulticlassConfusionMatrix`, ROC AUC, log loss, Cohen's κ, top-k accuracy, average precision, ROC & precision-recall curves
- **Clustering** — Adjusted Rand Index, Normalized / Adjusted Mutual Information, homogeneity / completeness / V-measure, Fowlkes–Mallows, silhouette, Davies–Bouldin, Calinski–Harabasz

Curve outputs follow scikit-learn's point ordering; `roc_curve` is the one deliberate difference,
always returning the full threshold sweep (scikit-learn's `drop_intermediate=False`), which can yield
more points than Python's default while tracing the same curve and the same `roc_auc`. The clustering
metrics take `isize` label arrays, matching what the clustering estimators return.

### `math`

Pure, stateless numerical primitives shared across the crate: `gemmkit`-backed matrix products
(GEMM / GEMV), deterministic blocked parallel reductions, and pairwise distances
(`squared_euclidean_distance_row`, `manhattan_distance_row`, `minkowski_distance_row`) plus the
`DistanceCalculationMetric` (Euclidean / Manhattan / Minkowski) dispatcher.

### `prelude`

One-stop imports, split by domain so you only pull in what you need:

```rust
use rustyml::prelude::machine_learning::*; // ML models (incl. PCA/KernelPCA/t-SNE), traits, config enums
use rustyml::prelude::neural_network::*;   // layers, optimizers, losses
use rustyml::prelude::utils::*;            // scaling, label encoding, splitting
use rustyml::prelude::metrics::*;          // evaluation metrics
```

## Feature Flags

The crate uses feature flags for modular compilation:

| Feature | Description |
|---------|-------------|
| `machine_learning` | Classical ML algorithms (enables `math`) |
| `neural_network` | Neural-network framework (enables `math`) |
| `utils` | Data preprocessing and dataset splitting (enables `math`) |
| `metrics` | Evaluation metrics (enables `math`) |
| `math` | Numerical primitives (distances, matrix products, parallel reductions) |
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
module for the full resolution rules. `KMeans` restarts its fit `n_init` times (10 by default),
deriving each restart's k-means++ seed deterministically from `random_state`, so a seeded fit stays
reproducible; the seed only reaches `IsolationForest`'s per-tree RNGs through an explicit
`random_state`, since those are built on Rayon worker threads.

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
