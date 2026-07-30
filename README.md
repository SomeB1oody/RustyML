[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

A machine learning and deep learning library written in **pure Rust**.

[![rustc](https://img.shields.io/badge/rustc-1.89%2B-brown)](https://www.rust-lang.org/)
[![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)

[![fmt](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/fmt.yml?branch=master&label=fmt)](https://github.com/SomeB1oody/RustyML/actions/workflows/fmt.yml)
[![clippy](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/clippy.yml?branch=master&label=clippy)](https://github.com/SomeB1oody/RustyML/actions/workflows/clippy.yml)
[![test](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/test.yml?branch=master&label=test)](https://github.com/SomeB1oody/RustyML/actions/workflows/test.yml)
[![doc](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/doc.yml?branch=master&label=doc)](https://github.com/SomeB1oody/RustyML/actions/workflows/doc.yml)

> **[RustyML User Guide](https://someb1oody.github.io/RustyML/en/)** covers every module, from your first model to performance tuning.

## Overview

RustyML is a machine learning and deep learning library, built end to end in Rust with no C or
C++ dependencies. It covers the full workflow: data preprocessing, feature engineering, model
training, and evaluation. It uses Rust's memory safety, safe concurrency, and zero-cost
abstractions.

RustyML splits into 5 feature-gated modules, so you compile only what you use:
`machine_learning`, `neural_network`, `utils`, `metrics`, and `math`, plus a shared `prelude`.

## Highlights

- **Pure Rust, no FFI**: memory-safe and portable, with nothing to link against.
- **Parallelized by default**: heavy kernels use [Rayon](https://github.com/rayon-rs/rayon) for multi-threaded computation.
- **Algorithm coverage**: classical supervised and unsupervised learning, anomaly detection, and a neural-network framework.
- **Structured error handling**: every fallible call returns `RustymlResult<T>`. The `Error` type groups failures into category variants instead of plain strings.
- **Reproducible**: a single `set_global_seed` call makes every randomized component on the calling thread deterministic. A per-component `random_state` covers the rest.
- **Model persistence**: save and load trained models and network weights as compact binary, using [Serde](https://serde.rs/) and [postcard](https://docs.rs/postcard/).
- **Evaluation metrics**: regression, classification (binary and multiclass), and clustering, matching scikit-learn conventions.
- **Checked against scikit-learn**: numeric tests verify estimator defaults, score signs, and metric output conventions against scikit-learn 1.9. A ported pipeline gives the same answers. Where the crate departs from scikit-learn on purpose, the item's own documentation says so.
- **Modular features**: the whole crate is on by default. Opt out and add only `metrics`, only `math`, or any subset you need.

## Installation

Add RustyML to your `Cargo.toml`:

```toml
[dependencies]
rustyml = "*"
ndarray = "0.17"
```

The default feature set is `full`, so every module is there. A ported scikit-learn script reaches
across all of them anyway (`utils::train_test_split` -> `machine_learning` -> `metrics`). To slim
the build, opt out of the default and name what you need:

```toml
# Everything (ml, nn, utils, metrics, math)
rustyml = "*"

# Just the neural-network framework
rustyml = { version = "*", default-features = false, features = ["neural_network"] }

# Just the evaluation metrics
rustyml = { version = "*", default-features = false, features = ["metrics"] }

# Show training progress bars in the terminal
rustyml = { version = "*", features = ["show_progress"] }
```

Cargo features are additive. Naming one does not disable the rest. `features = ["metrics"]` alone
still compiles the full crate. The `default-features = false` line does the trimming.

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

// 1 loss value per epoch, each measured while that epoch ran, not after it
let history = model.fit(&x, &y, 10).unwrap();
println!("Per-epoch loss: {:?}", history.loss());

// Score the weights the model holds now: inference mode, updates nothing
println!("Loss after training: {}", model.evaluate(&x, &y).unwrap());

let predictions = model.predict(&x).unwrap();
println!("Predictions shape: {:?}", predictions.shape());

// Save the trained weights to a file
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

Classical supervised and unsupervised algorithms, with parallel-processing optimization, input
validation, and binary persistence.

| Category | Algorithms |
|----------|------------|
| **Regression** | Linear Regression (closed-form OLS by default, or gradient descent, with optional L1/L2 regularization) |
| **Classification** | Logistic Regression, K-Nearest Neighbors, Decision Tree (ID3 / C4.5 / CART), SVC (kernel SMO), Linear SVC, Linear Discriminant Analysis |
| **Clustering** | KMeans (K-means++ init, `n_init` restarts), DBSCAN, MeanShift (flat kernel) |
| **Dimensionality Reduction** | PCA (multiple SVD solvers), KernelPCA (RBF / Linear / Poly / Sigmoid / Cosine kernels), t-SNE |
| **Anomaly Detection** | Isolation Forest |

All 3 clustering estimators return `Array1<isize>` labels, with `-1` for noise or unassigned
points. Every clustering metric consumes that same type, so any estimator feeds any metric
directly. `IsolationForest` follows scikit-learn's sign convention. `score_samples` returns
values in `[-1, 0)`, where lower means more anomalous. `predict` returns `-1` for an outlier and
`+1` for an inlier.

Shared config types live in [`machine_learning::types`](https://docs.rs/rustyml/latest/rustyml/machine_learning/types/index.html):
`RegularizationType` (L1 or L2), `Gamma`, and `KernelType` (Linear, Poly, RBF, Sigmoid, or Cosine).
The `RegularizationType` documentation carries a conversion table for scikit-learn's penalty
strengths (`Lasso` or `Ridge` `alpha`, `LogisticRegression` `C`). L1 uses a proximal
soft-threshold, so a penalized coefficient reaches exactly zero.

The `DistanceCalculationMetric` (Euclidean, Manhattan, or Minkowski) dispatcher lives in
[`math`](https://docs.rs/rustyml/latest/rustyml/math/index.html) and is re-exported at the
`machine_learning` root. Predictive models implement the shared `Fit` and `Predict` traits. The
dimensionality-reduction transformers ([`decomposition`](https://docs.rs/rustyml/latest/rustyml/machine_learning/decomposition/index.html)
and [`manifold`](https://docs.rs/rustyml/latest/rustyml/machine_learning/manifold/index.html))
implement `Transform` and `FitTransform`. All 4 traits live at the crate root in
[`traits`](https://docs.rs/rustyml/latest/rustyml/traits/index.html), shared with the stateful
preprocessing transformers in `utils`.

### `neural_network`

A framework for building, training, and serializing feed-forward, convolutional, and recurrent
networks with a Keras-style `Sequential` API. Tensors are channels-last, and kernels use the same
shapes as Keras. A layout you already know from Keras carries over unchanged.

- **Core layers**: `Dense`, `Flatten`
- **Activations**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Linear` (as the `Activation` enum or standalone layers)
- **Convolution**: `Conv1D`, `Conv2D`, `Conv3D`, `DepthwiseConv2D`, `SeparableConv2D`
- **Pooling**: max and average pooling for 1D, 2D, and 3D, plus their global variants
- **Recurrent**: `SimpleRNN`, `LSTM`, `GRU`
- **Regularization**: `Dropout`, `SpatialDropout{1,2,3}D`, `GaussianNoise`, `GaussianDropout`
- **Normalization**: `BatchNormalization`, `LayerNormalization`, `InstanceNormalization`, `GroupNormalization`
- **Optimizers**: `SGD` (with momentum), `Adam`, `AdamW`, `RMSprop`, `AdaGrad`
- **Losses**: `MeanSquaredError`, `MeanAbsoluteError`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`

Training supports a full-batch loop (`fit`) and a mini-batch loop (`fit_with_batches`), weight
inspection (`get_weights`), and binary serialization (`save_to_path` and `load_from_path`).

Both loops return a `History` holding one loss value per epoch. Training measures each value
during that epoch, not after it, matching Keras. Each value comes from the forward pass that
precedes each batch's own weight update. So an entry describes the weights the epoch ran with,
not the weights it ends with.

Use `evaluate` to score the model you are holding now. That call is one inference-mode pass. It
touches no gradients, no parameters, and no batch-norm running statistics.

`train_batch` (Keras' `train_on_batch`) is also public. A custom loop can therefore own the epoch
structure. It does not have to reimplement the forward pass, loss, backward pass, clipping, and
update steps.

### `utils`

Data preprocessing and dataset splitting. Dimensionality reduction (`PCA`, `KernelPCA`, `TSNE`)
now lives in `machine_learning`, under `decomposition` and `manifold`.

- **Scaling (one-shot)**: `standardize` (z-score), `normalize` (configurable axis and order)
- **Scaling (stateful)**: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler` (median and IQR, resists outliers), and `Normalizer` (per-sample norm). Every scaler supports `fit`, `transform`, and `fit_transform`, and stores its training statistics for later batches. `StandardScaler`, `MinMaxScaler`, and `MaxAbsScaler` also support `partial_fit` and `inverse_transform`. `RobustScaler` supports `inverse_transform` but not `partial_fit`. `Normalizer` supports neither. All 5 scalers persist with `save_to_path`.
- **Label encoding**: `to_categorical`, `to_categorical_with_mapping`, `to_sparse_categorical`
- **Splitting**: `train_test_split` and `train_test_split_stratified`, with a configurable ratio

### `metrics`

An evaluation suite for regression, classification, and clustering. Every function takes
`(y_true, y_pred)` and panics on a precondition violation, such as mismatched lengths, empty
input, or an out-of-domain label, instead of returning `Result`. This keeps the leaf module
dependency-light.

- **Regression**: MSE, RMSE, MAE, median absolute error, MAPE, R^2, explained variance
- **Classification**: accuracy, `ConfusionMatrix` and `MulticlassConfusionMatrix`, ROC AUC, log loss, Cohen's kappa, top-k accuracy, average precision, and ROC and precision-recall curves
- **Clustering**: Adjusted Rand Index, normalized and adjusted mutual information, homogeneity, completeness, V-measure, Fowlkes-Mallows, silhouette, Davies-Bouldin, and Calinski-Harabasz

Curve outputs follow scikit-learn's point ordering. `roc_curve` is the only deliberate
difference. It always returns the full threshold sweep (scikit-learn's `drop_intermediate=False`).
That can yield more points than Python's default, while it traces the same curve and the same
`roc_auc`. The clustering metrics take `isize` label arrays, matching what the clustering
estimators return.

### `math`

Pure, stateless numerical primitives shared across the crate. `gemmkit`-backed matrix products
cover GEMM and GEMV. Deterministic blocked parallel reductions are also here. Pairwise distances
(`squared_euclidean_distance_row`, `manhattan_distance_row`, `minkowski_distance_row`) round out
the module, along with the `DistanceCalculationMetric` (Euclidean, Manhattan, or Minkowski)
dispatcher.

### `prelude`

Single-import access, split by domain, so you import only what you need:

```rust
use rustyml::prelude::machine_learning::*; // ML models (including PCA, KernelPCA, and t-SNE), traits, and config enums
use rustyml::prelude::neural_network::*; // Sequential and History, layers, optimizers, and losses
use rustyml::prelude::utils::*; // scaling, label encoding, and splitting
use rustyml::prelude::metrics::*; // evaluation metrics
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
| `full` | All of the above modules |
| `default` | `full` |
| `show_progress` | Render training/iteration progress bars in the terminal |

## Reproducibility

Every randomized component (weight initialization, K-means++, Isolation Forest, t-SNE, dropout,
and more) resolves its `random_state: Option<u64>` against a shared entry point. Set a single
global seed, and the whole crate becomes deterministic:

```rust
use rustyml::set_global_seed;

set_global_seed(42);
// ... train models. Results are now reproducible across runs ...
```

A per-call `random_state` takes precedence over the global seed. The global seed in turn takes
precedence over system entropy. See the [`random`](https://docs.rs/rustyml/latest/rustyml/random/index.html)
module for the full resolution rules.

`KMeans` restarts its fit `n_init` times (10 by default). Each restart derives its k-means++ seed
deterministically from `random_state`, so a seeded fit stays reproducible. `IsolationForest`
builds its per-tree RNGs on Rayon worker threads, so only an explicit `random_state` reaches them
there.

## Error Handling

Outside the `metrics` and `math` leaf modules, every fallible operation returns
`RustymlResult<T>` (an alias for `Result<T, rustyml::error::Error>`). The `Error` type groups
failures into category variants. It nests domain-specific failures in `NnError` and `TreeError`,
and shared I/O and serialization failures in `IoError`. You can match on what went wrong instead
of parsing strings.

## Project Status

RustyML is under active development. The API is stabilizing, but breaking changes can still
appear in minor releases before `1.0.0`.

## Contributing

Contributions are welcome. To help build the Rust ML library, you can:

1. Open issues for bugs or feature requests
2. Submit pull requests for improvements
3. Share feedback on the API design
4. Improve the documentation and examples

Please also review the [Code of Conduct](https://github.com/SomeB1oody/RustyML/blob/master/CODE_OF_CONDUCT.md).

## Author

SomeB1oody ([stanyin64@gmail.com](mailto:stanyin64@gmail.com))

## License

The [MIT License](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE) covers this project. See the LICENSE file for details.
