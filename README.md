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

> Read **[RustyML User Guide](https://someb1oody.github.io/RustyML/en/)** for detailed documentation and tutorials of RustyML.

## Overview

RustyML is a machine learning and deep learning library, built end to end in Rust with no C or
C++ dependencies. It covers the full workflow: data preprocessing, feature engineering, model
training, and evaluation. It uses Rust's memory safety, safe concurrency, and zero-cost
abstractions.

## Highlights

- **Pure Rust, no FFI**: memory-safe and portable, with nothing to link against.
- **Parallelized by default**: heavy kernels use [Rayon](https://github.com/rayon-rs/rayon) for multi-threaded computation.
- **Algorithm coverage**: classical supervised and unsupervised learning, anomaly detection, and a neural-network framework.
- **Reproducible**: a single `set_global_seed` call makes every randomized component on the calling thread deterministic. A per-component `random_state` covers the rest.
- **Model persistence**: save and load trained models and network weights as compact binary, using [Serde](https://serde.rs/) and [postcard](https://docs.rs/postcard/).
- **Evaluation metrics**: regression, classification (binary and multiclass), and clustering, matching scikit-learn conventions.
## Installation

Add RustyML to your `Cargo.toml`:

```toml
[dependencies]
rustyml = "*"
ndarray = "0.17"
```

To slim the build, opt out of the default and name what you need:

```toml, ignore
# Everything (ml, nn, utils, metrics, math)
rustyml = "*"

# Just the neural-network framework
rustyml = { version = "*", default-features = false, features = ["neural_network"] }

# Just the evaluation metrics
rustyml = { version = "*", default-features = false, features = ["metrics"] }

# Show training progress bars in the terminal
rustyml = { version = "*", features = ["show_progress"] }
```

**MSRV:** Rust 1.89+ (edition 2024).

## Quick Start

### Classical Machine Learning

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

fn main() {
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
}
```

### Neural Networks

```rust
use rustyml::prelude::neural_network::*;
use ndarray::Array;

fn main() {
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
}
```

### Evaluating a Model

```rust
use rustyml::metrics::*;
use ndarray::array;

fn main() {
    // Arguments are always (y_true, y_pred)
    // ConfusionMatrix::new takes hard 0.0/1.0 labels (new_with_labels covers other pairs)
    let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

    // The two arguments carry independent storage types, so an owned array and a view mix
    let cm = ConfusionMatrix::new(&y_true, &y_pred.view());
    println!("Accuracy: {:.3}", cm.accuracy());
    println!("F1 score: {:.3}", cm.f1_score());
}
```

## Modules

See at [docs.rs](https://docs.rs/rustyml/latest/rustyml/index.html#architecture)

## Feature Flags

The crate uses feature flags for modular compilation:

| Feature            | Description                                                            |
|--------------------|------------------------------------------------------------------------|
| `machine_learning` | Classical ML algorithms (enables `math`)                               |
| `neural_network`   | Neural-network framework (enables `math`)                              |
| `utils`            | Data preprocessing and dataset splitting (enables `math`)              |
| `metrics`          | Evaluation metrics (enables `math`)                                    |
| `math`             | Numerical primitives (distances, matrix products, parallel reductions) |
| `full`             | All of the above modules                                               |
| `default`          | `full`                                                                 |
| `show_progress`    | Render training/iteration progress bars in the terminal                |

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
