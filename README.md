[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

A machine learning and deep learning library written in **pure Rust**.

[![rustc](https://img.shields.io/badge/rustc-1.89%2B-brown)](https://www.rust-lang.org/)
[![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)
[![ci](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/ci.yml?branch=master&label=ci)](https://github.com/SomeB1oody/RustyML/actions/workflows/ci.yml)

> Read the **[RustyML User Guide](https://someb1oody.github.io/RustyML/en/)** for instructions and examples.

## Overview

RustyML provides data preprocessing, machine learning models, neural networks, and evaluation metrics.
It uses `ndarray` arrays and Rust numeric kernels.

## Features

- **Parallel kernels**: selected kernels use [Rayon](https://github.com/rayon-rs/rayon). Work thresholds control the choice between serial and parallel paths.
- **Models**: supervised learning, unsupervised learning, anomaly detection, sequential networks, and graph networks.
- **Random seeds**: `set_global_seed` sets a seed stream on the calling thread. Local `random_state` values override that stream where supported.
- **Binary files**: supported models use [Serde](https://serde.rs/) and [postcard](https://docs.rs/postcard/) to save and load fitted state.
- **Network checkpoints**: save model arrays, then load them into a compatible built model. Checkpoints do not restore the optimizer or loss.
- **Metrics**: regression, binary classification, multiclass classification, and clustering. Read each metric's input rules and score definition.

Reset the global seed before repeating the same sequence of random operations.
Other threads have separate seed streams.
Some decomposition solvers use independent seeds.

## Installation

Add RustyML to your `Cargo.toml`:

```toml
[dependencies]
rustyml = "0.17.0"
ndarray = "0.17"
```

Disable default features to select individual modules:

```toml,ignore
# Everything (ml, nn, utils, metrics, math)
rustyml = "0.17.0"

# Just the neural-network framework
rustyml = { version = "0.17.0", default-features = false, features = ["neural_network"] }

# Just the evaluation metrics
rustyml = { version = "0.17.0", default-features = false, features = ["metrics"] }

# Show training progress bars in the terminal
rustyml = { version = "0.17.0", features = ["show_progress"] }
```

**MSRV:** Rust 1.89+ (edition 2024).

## Quick Start

### Classical Machine Learning

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

fn main() {
    // Train a linear regression model without regularization.
    let mut model = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent { learning_rate: 0.01, max_iter: 1000, tol: 1e-6 }).unwrap();

    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![6.0, 9.0, 12.0];

    model.fit(&x, &y).unwrap();
    let predictions = model.predict(&x).unwrap();
    println!("{:?}", predictions);

    // Save and reload the trained model.
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
    let mut y = Array::zeros((32, 10));
    y.column_mut(0).fill(1.0);
    let y = y.into_dyn();

    // Build the layers from the input shape.
    let mut model = SequentialBuilder::new()
        .add(Dense::new(128, Activation::ReLU).unwrap())
        .add(Dense::new(64, Activation::ReLU).unwrap())
        .add(Dense::new(10, Activation::Softmax { axis: -1 }).unwrap())
        .build(&Shape::known(x.shape()))
        .unwrap();

    model.compile(
        Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        CategoricalCrossEntropy::new(false),
    );

    model.summary(); // Print the model structure.

    // Record 1 loss value during each epoch.
    let history = model.fit(&x, &y, 10).unwrap();
    println!("Per-epoch loss: {:?}", history.loss());

    // Evaluate the current weights in inference mode.
    println!("Loss after training: {}", model.evaluate(&x, &y).unwrap());

    let predictions = model.predict(&x).unwrap();
    println!("Predictions shape: {:?}", predictions.shape());

    // Save all model arrays to a file.
    model.save_to_path("model.bin").unwrap();
}
```

### Graph Models

A sequential model passes each layer's output to the next layer.
`GraphBuilder` connects layer calls as nodes in a graph.
A graph can have multiple inputs, multiple outputs, shared layers, and residual connections.
The merge layers join branches: `Add`, `Subtract`, `Multiply`, `Average`, `Maximum`, `Minimum`, and `Concatenate`.

```rust
use rustyml::prelude::neural_network::*;
use ndarray::Array;

fn main() {
    // 4 samples of 8 features, and 2 output values per sample
    let x = Array::ones((4, 8)).into_dyn();
    let y = Array::ones((4, 2)).into_dyn();

    // A residual block: the input of the block reaches the sum and the hidden layer alike
    let mut builder = GraphBuilder::new();
    let input = builder.input(Shape::known(&[4, 8]));
    let hidden = builder.add(Dense::new(8, Activation::ReLU).unwrap(), &[input]);
    let sum = builder.add(Add::new(), &[input, hidden]);
    let head = builder.add(Dense::new(2, Activation::Linear).unwrap(), &[sum]);
    let mut model = builder.build(&[head]).unwrap();

    model.compile(
        SGD::new(0.01, 0.0, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    // 1 tensor per inlet, and 1 target per outlet
    model.fit(&[&x], &[&y], 10).unwrap();

    let predictions = model.predict(&[&x]).unwrap();
    println!("Predictions shape: {:?}", predictions[0].shape());
}
```

### Evaluating a Model

```rust
use rustyml::metrics::*;
use ndarray::array;

fn main() {
    // Pass the true labels before the predicted labels.
    // Use hard 0.0/1.0 labels here. Use new_with_labels for other label pairs.
    let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

    // The arguments can use an owned array and a view.
    let cm = ConfusionMatrix::new(&y_true, &y_pred.view());
    println!("Accuracy: {:.3}", cm.accuracy());
    println!("F1 score: {:.3}", cm.f1_score());
}
```

## Modules

The public modules group the API by task:

- `machine_learning`: estimators for classification, regression, clustering, decomposition, and anomaly detection.
- `neural_network`: layers, model builders, losses, optimizers, and checkpoints.
- `utils`: preprocessing, label conversion, scalers, and dataset splits.
- `metrics`: classification, regression, and clustering scores.
- `math`: distance functions, matrix operations, and reductions.
- `error`: shared error types and result aliases.
- `random`: thread-local seed controls.
- `traits`: common estimator contracts.
- `tuning`: runtime thresholds for serial and parallel work.
- `prelude`: grouped imports for public types and traits.

See the [API reference](https://docs.rs/rustyml/latest/rustyml/).

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

RustyML is under active development. Minor releases can change the API before `1.0.0`.

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
