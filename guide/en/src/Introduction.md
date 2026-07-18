# Introduction

RustyML is a machine learning and deep learning library written entirely in Rust — nothing to link against, no Python interpreter in the loop, no FFI boundary to marshal arrays across. This guide is the hands-on companion to that library: a book that takes you from a fresh `cargo new` to training neural networks, tuning the parallel/serial thresholds inside the hot kernels, and reasoning about exactly what lands on disk when you serialize a model. It tracks RustyML **0.13**; every complete example here is compiled against that version with the `full` and `show_progress` features enabled, so what you read is what the compiler accepts.

## What this guide is

The [API reference on docs.rs](https://docs.rs/rustyml) is the source of truth for every signature, every trait bound, every enum variant. This guide does not duplicate it — it explains the decisions the signatures leave unsaid. docs.rs tells you the arguments to `Adam::new`; this guide tells you which optimizer to reach for, why a per-call `random_state` overrides the global seed rather than the other way around, and what happens to a saved network when you load it into a model whose layer shapes no longer match. It is written the way an experienced colleague would walk you through a codebase they know well: concrete, opinionated where opinions are earned, and honest about the edges where a given estimator will bite you.

Here is the shape of nearly everything in the classical-ML half of the crate — construct, `fit`, `predict`, the same `(&x, &y)`-then-`&x` rhythm you know from scikit-learn:

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

fn main() {
    // A tiny linear relationship: y = 3 * x
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![3.0, 6.0, 9.0, 12.0];

    // (fit_intercept, learning_rate, max_iter, tolerance)
    let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6).unwrap();
    model.fit(&x, &y).unwrap();

    let predictions = model.predict(&x).unwrap();
    println!("predictions: {:?}", predictions);
}
```

That fluency — models sharing the `Fit` / `Predict` traits, metrics that always take `(y_true, y_pred)` in that order, a single `set_global_seed` that makes the whole crate deterministic — is what the rest of the book unpacks, one module at a time.

## Who it is for

Two readers, coming from opposite directions, will both feel at home here. The first is a **Rust developer** who wants machine learning without leaving the ecosystem: no `pip`, no linked BLAS, no unsafe FFI to audit, just a `cargo add` and a crate that plays by Rust's rules on ownership, `Send`/`Sync`, and error handling. The second is an **ML practitioner arriving from Python**, fluent in scikit-learn and Keras, who wants the same mental models — `fit`/`predict`, a Keras-style `Sequential` you `.add(...)` layers to and `.compile(...)`, confusion matrices and silhouette scores that mean what they mean in sklearn — but expressed as compiled, statically typed, parallel-by-default Rust. Where RustyML deliberately diverges from those tools, the guide says so and says why.

You do not need prior Rust experience with numerical code, but you should be comfortable reading Rust and running `cargo`. Data flows through [`ndarray`](https://docs.rs/ndarray) arrays throughout, so [Working with ndarray](./Chapter-01/1.3._Working_with_ndarray.md) gives that library the short, focused treatment it deserves before you meet it everywhere else.

## How the book is organized

The crate splits into five feature-gated modules — `machine_learning`, `neural_network`, `utils`, `metrics`, and `math` — plus a domain-split `prelude`. The chapters follow that structure, front-loaded with the one chapter you should read start to finish.

| Chapter | Covers | Maps to |
|---|---|---|
| [1. Getting Started](./Chapter-01/1.0._Getting_Started.md) | Installation, feature flags, ndarray, your first end-to-end model, the prelude, error handling | the on-ramp |
| [2. Classical Machine Learning](./Chapter-02/2.0._Classical_Machine_Learning.md) | Regression, classification, clustering, dimensionality reduction, anomaly detection | `machine_learning` |
| [3. Neural Networks](./Chapter-03/3.0._Neural_Networks.md) | The `Sequential` model, dense/conv/recurrent layers, losses, optimizers, saving weights | `neural_network` |
| [4. Data Preprocessing](./Chapter-04/4.0._Data_Preprocessing.md) | Train/test splitting, standardization and normalization, label encoding | `utils` |
| [5. Model Evaluation](./Chapter-05/5.0._Model_Evaluation.md) | Regression, classification, and clustering metrics | `metrics` |
| [6. Math Utilities](./Chapter-06/6.0._Math_Utilities.md) | Distance metrics, matrix multiplication, deterministic parallel reductions | `math` |
| [7. Advanced Topics](./Chapter-07/7.0._Advanced_Topics.md) | Reproducibility and seeds, model persistence internals, performance tuning, minimal builds | cross-cutting |

Chapters 2 and 3 are the two large algorithm families and carry most of the crate's surface area. Chapters 4 through 6 are the supporting cast — the preprocessing you run before a model, the metrics you run after, and the numerical primitives both are built on. Chapter 7 is where the guide earns the word *advanced*: how `random_state` resolves against the global seed, what the postcard-serialized bytes of a saved model actually contain, how the runtime-tunable parallelism gates work, and how to compile a build that pulls in only `metrics` or only `math`.

## How to read it

Read [Getting Started](./Chapter-01/1.0._Getting_Started.md) in order, once, front to back. It is the only chapter that assumes you have read what came before it: it installs the crate, sets up ndarray, and builds a complete model so that every later chapter can lean on that shared footing instead of re-deriving it. Everything after Chapter 1 is reference-style and deliberately order-independent — jump straight to [Support Vector Machines](./Chapter-02/2.5._Support_Vector_Machines.md), [Optimizers](./Chapter-03/3.4._Optimizers.md), or [Clustering Metrics](./Chapter-05/5.3._Clustering_Metrics.md) as your problem demands, and follow the inline cross-links when one page depends on a concept another page owns. If you already have RustyML compiling and a model training, you have finished Chapter 1's job; treat the rest as a manual you open at the page you need.

## Versions, feedback, and conventions

This edition tracks the current crate version, **0.13**. RustyML is pre-1.0 and under active development: the API is stabilizing but breaking changes can still land in minor releases, so if a signature in the guide has drifted from what your compiler sees, trust [docs.rs for your exact version](https://docs.rs/rustyml) and let us know. Bug reports, feature requests, and corrections to this guide all belong in the same place — the [GitHub repository](https://github.com/SomeB1oody/RustyML), where issues and pull requests are welcome.

Two conventions worth internalizing before you go further. First, complete examples in this book are self-contained programs with a `main`, tiny inline datasets, and few iterations, so you can paste one into a `full`-feature project and run it as-is; fragments and signatures are marked as such. Second, outside the leaf `metrics` and `math` modules, every fallible call returns `RustymlResult<T>` — an alias for `Result<T, rustyml::error::Error>` over a structured, matchable error enum — and the examples reach for `.unwrap()` only to stay short. [Error Handling](./Chapter-01/1.6._Error_Handling.md) shows what you would write instead in real code.
