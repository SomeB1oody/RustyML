# Introduction

RustyML is a machine learning and deep learning library written entirely in Rust. It needs no external library to link against, no Python interpreter, and no FFI boundary to move arrays across. This guide is the hands-on companion to the library. It takes you from a fresh `cargo new` to training neural networks. It also covers tuning the parallel and serial thresholds inside the hot kernels, and shows what lands on disk when you serialize a model. This guide tracks RustyML **0.15**. Every complete example compiles against that version with the `full` and `show_progress` features on, so what you read is what the compiler accepts.

## What this guide is

The [API reference on docs.rs](https://docs.rs/rustyml) is the source of truth for every signature, every trait bound, and every enum variant. This guide does not duplicate that reference. It explains the decisions a signature does not state. docs.rs lists the arguments to `Adam::new`. This guide explains which optimizer to choose. It explains why a per-call `random_state` overrides the global seed instead of the reverse. It also explains what happens when you load a saved network into a model whose layer shapes no longer match. The guide stays concrete, states an opinion where the evidence supports one, and names the pitfalls in specific estimators.

The classical-ML half of the crate follows one shape almost everywhere: construct, `fit`, `predict`. That is the same `(&x, &y)`-then-`&x` rhythm scikit-learn uses.

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

fn main() {
    // A tiny linear relationship: y = 3 * x
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![3.0, 6.0, 9.0, 12.0];

    // new(fit_intercept). The default solver is the exact closed form
    let mut model = LinearRegression::new(true);
    model.fit(&x, &y).unwrap();

    let predictions = model.predict(&x).unwrap();
    println!("predictions: {:?}", predictions);
}
```

The rest of this guide unpacks that pattern, one module at a time. Models share the `Fit` and `Predict` traits. Metrics always take `(y_true, y_pred)` in that order. A single `set_global_seed` call makes the whole crate deterministic.

## Who it is for

This guide serves 2 kinds of readers. The first is a **Rust developer** who wants machine learning without leaving the Rust ecosystem. That reader needs no `pip`, no linked BLAS, and no unsafe FFI to audit. A `cargo add` command adds a crate that follows Rust's rules on ownership, `Send`/`Sync`, and error handling. The second is an **ML practitioner coming from Python**, fluent in scikit-learn and Keras. That reader wants the same mental model in Rust. RustyML gives it: `fit`/`predict` methods and a Keras-style `Sequential` model built with `.add(...)` and `.compile(...)`. Its confusion matrices and silhouette scores mean what they mean in scikit-learn. RustyML expresses these ideas as compiled, statically typed, parallel-by-default Rust. Where RustyML departs from scikit-learn or Keras, this guide states the difference and the reason for it.

You do not need prior experience with Rust numerical code. You should be comfortable reading Rust and running `cargo`. Data flows through [`ndarray`](https://docs.rs/ndarray) arrays throughout the crate, so [Working with ndarray](./Chapter-01/1.3._Working_with_ndarray.md) covers that library before you meet it in every later chapter.

## How the book is organized

The crate splits into 5 feature-gated modules: `machine_learning`, `neural_network`, `utils`, `metrics`, and `math`. It also has a domain-split `prelude`. The chapters follow that structure. Read the first chapter start to finish, then use the rest as reference material.

| Chapter | Covers | Maps to |
|---|---|---|
| [1. Getting Started](./Chapter-01/1.0._Getting_Started.md) | Installation, feature flags, ndarray, your first end-to-end model, the prelude, error handling | the on-ramp |
| [2. Classical Machine Learning](./Chapter-02/2.0._Classical_Machine_Learning.md) | Regression, classification, clustering, dimensionality reduction, anomaly detection | `machine_learning` |
| [3. Neural Networks](./Chapter-03/3.0._Neural_Networks.md) | The `Sequential` model, dense/conv/recurrent layers, losses, optimizers, saving weights | `neural_network` |
| [4. Data Preprocessing](./Chapter-04/4.0._Data_Preprocessing.md) | Train/test splitting, standardization and normalization, label encoding | `utils` |
| [5. Model Evaluation](./Chapter-05/5.0._Model_Evaluation.md) | Regression, classification, and clustering metrics | `metrics` |
| [6. Math Utilities](./Chapter-06/6.0._Math_Utilities.md) | Distance metrics, matrix multiplication, deterministic parallel reductions | `math` |
| [7. Advanced Topics](./Chapter-07/7.0._Advanced_Topics.md) | Reproducibility and seeds, model persistence internals, performance tuning, minimal builds | cross-cutting |

Chapters 2 and 3 cover the two large algorithm families and most of the crate's surface area. Chapters 4 through 6 support those families. Preprocessing runs before a model, metrics run after it, and both rest on the same numerical primitives. Chapter 7 covers cross-cutting advanced topics. It explains how `random_state` resolves against the global seed, what the postcard-serialized bytes of a saved model contain, and how the runtime-tunable parallelism gates work. It also shows how to compile a build that pulls in only `metrics` or only `math`.

## How to read it

Read [Getting Started](./Chapter-01/1.0._Getting_Started.md) once, in order, from start to end. It is the only chapter that assumes you read the chapters before it. It installs the crate, sets up ndarray, and builds a complete model, so every later chapter can build on that shared base. Every chapter after Chapter 1 is reference material, and you can read those chapters in any order. Jump straight to [Support Vector Machines](./Chapter-02/2.5._Support_Vector_Machines.md), [Optimizers](./Chapter-03/3.4._Optimizers.md), or [Clustering Metrics](./Chapter-05/5.3._Clustering_Metrics.md) as your problem demands. Follow the inline cross-links when one page depends on a concept from another page. If RustyML already compiles for you and a model already trains, Chapter 1 has done its job. Treat the rest of the guide as a manual: open the page you need.

## Versions, feedback, and conventions

This edition tracks the current crate version, **0.15**. RustyML is pre-1.0 and under active development. The API is stabilizing, but breaking changes can still land in minor releases. If a signature in this guide differs from what the compiler accepts, trust [docs.rs for the exact version](https://docs.rs/rustyml) in use. Send bug reports, feature requests, and corrections to this guide to the [GitHub repository](https://github.com/SomeB1oody/RustyML). Issues and pull requests are welcome there.

This guide follows 2 conventions. First, a complete example is a self-contained program with a `main` function, a tiny inline dataset, and few iterations. Paste a complete example into a `full`-feature project and run it as written. A fragment or a signature is marked as such. Second, outside the leaf `metrics` and `math` modules, every fallible call returns `RustymlResult<T>`, an alias for `Result<T, rustyml::error::Error>` over a structured, matchable error enum. The examples reach for `.unwrap()` only to stay short. [Error Handling](./Chapter-01/1.6._Error_Handling.md) shows what to write instead in real code.
