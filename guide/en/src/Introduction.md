# RustyML User Guide

RustyML provides machine learning models, neural networks, data utilities, and evaluation metrics for Rust.
This book explains the main public APIs in RustyML 0.17.0.
It assumes basic Rust and Cargo knowledge.

Start with [Installation](installation.md), then run [Your first model](first-model.md).
Read the other chapters when you need their APIs.

| Task | Chapter |
| --- | --- |
| Split data, scale features, or encode labels | [Prepare data](data.md) |
| Predict values or class labels | [Supervised models](supervised.md) |
| Find clusters or reduce feature counts | [Unsupervised models](unsupervised.md) |
| Build and train a neural network | [Neural networks](neural-networks.md) |
| Select layers and tensor shapes | [Neural network layers](neural-network-layers.md) |
| Measure model results | [Evaluate models](evaluation.md) |
| Save and load a classical model | [Model files](model-files.md) |
| Handle failed operations | [Errors](errors.md) |
| Repeat randomized operations | [Random seeds](random-seeds.md) |

The examples use small arrays to show API use.
Their scores do not measure performance on real datasets.
For complete type and method details, use the [API reference](https://docs.rs/rustyml/0.17.0/rustyml/).
