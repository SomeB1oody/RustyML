[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

一个用**纯 Rust** 编写的机器学习与深度学习库。

[![rustc](https://img.shields.io/badge/rustc-1.89%2B-brown)](https://www.rust-lang.org/)
[![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)

[![fmt](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/fmt.yml?branch=master&label=fmt)](https://github.com/SomeB1oody/RustyML/actions/workflows/fmt.yml)
[![clippy](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/clippy.yml?branch=master&label=clippy)](https://github.com/SomeB1oody/RustyML/actions/workflows/clippy.yml)
[![test](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/test.yml?branch=master&label=test)](https://github.com/SomeB1oody/RustyML/actions/workflows/test.yml)
[![doc](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/doc.yml?branch=master&label=doc)](https://github.com/SomeB1oody/RustyML/actions/workflows/doc.yml)

> 阅读 **[RustyML 使用指南](https://someb1oody.github.io/RustyML/zh-Hans/)**，获取 RustyML 的详细文档与教程。

## 概述

RustyML 是一个机器学习与深度学习库，完全用 Rust 端到端实现，不依赖任何 C 或 C++ 代码。它覆盖完整的
工作流程：数据预处理、特征工程、模型训练和评估。它利用了 Rust 的内存安全、安全并发和零成本抽象。

## 核心亮点

- **纯 Rust，无 FFI**：内存安全、可移植，无需链接任何外部库。
- **默认并行**：计算密集的内核使用 [Rayon](https://github.com/rayon-rs/rayon) 进行多线程计算。
- **算法覆盖**：经典的监督与无监督学习、异常检测，以及一个神经网络框架。
- **可复现**：一次 `set_global_seed` 调用即可让调用线程上所有随机化组件变得确定。按组件设置的 `random_state` 覆盖其余情形。
- **模型持久化**：通过 [Serde](https://serde.rs/) 和 [postcard](https://docs.rs/postcard/) 将训练好的模型和网络权重保存为紧凑的二进制格式。
- **评估指标**：回归、分类（二分类与多分类）、聚类，遵循 scikit-learn 的约定。

## 安装

在 `Cargo.toml` 中添加 RustyML：

```toml
[dependencies]
rustyml = "*"
ndarray = "0.17"
```

想把构建裁小，就关掉默认并显式列出所需模块：

```toml, ignore
# 全部模块（ml、nn、utils、metrics、math）
rustyml = "*"

# 仅神经网络框架
rustyml = { version = "*", default-features = false, features = ["neural_network"] }

# 仅评估指标
rustyml = { version = "*", default-features = false, features = ["metrics"] }

# 训练时在终端显示进度条
rustyml = { version = "*", features = ["show_progress"] }
```

**最低支持 Rust 版本（MSRV）：** Rust 1.89+（edition 2024）。

## 快速上手

### 经典机器学习

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

fn main() {
    // 训练一个不带正则化的线性回归模型
    let mut model = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent { learning_rate: 0.01, max_iter: 1000, tol: 1e-6 }).unwrap();

    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let y = array![6.0, 9.0, 12.0];

    model.fit(&x, &y).unwrap();
    let predictions = model.predict(&x).unwrap();
    println!("{:?}", predictions);

    // 保存并重新加载训练好的模型
    model.save_to_path("linear_regression.bin").unwrap();
    let restored = LinearRegression::load_from_path("linear_regression.bin").unwrap();
}
```

### 神经网络

```rust
use rustyml::prelude::neural_network::*;
use ndarray::Array;

fn main() {
    // 32 个样本，784 个输入特征，10 个输出类别
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

    model.summary(); // 打印网络结构

    // 每个 epoch 1 个损失值，测的是这个 epoch 运行期间的权重，而不是结束时的
    let history = model.fit(&x, &y, 10).unwrap();
    println!("Per-epoch loss: {:?}", history.loss());

    // 给模型此刻持有的权重打分：推理模式，什么都不更新
    println!("Loss after training: {}", model.evaluate(&x, &y).unwrap());

    let predictions = model.predict(&x).unwrap();
    println!("Predictions shape: {:?}", predictions.shape());

    // 保存训练好的权重
    model.save_to_path("model.bin").unwrap();
}
```

### 评估模型

```rust
use rustyml::metrics::*;
use ndarray::array;

fn main() {
    // 参数顺序始终是 (y_true, y_pred)
    // ConfusionMatrix::new 只接受 0.0/1.0 硬标签（其他标签取值对用 new_with_labels）
    let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

    // 两个参数的存储类型相互独立，持有所有权的数组和视图可以混用
    let cm = ConfusionMatrix::new(&y_true, &y_pred.view());
    println!("Accuracy: {:.3}", cm.accuracy());
    println!("F1 score: {:.3}", cm.f1_score());
}
```

## 模块

参见 [docs.rs](https://docs.rs/rustyml/latest/rustyml/index.html#architecture)

## 特性标志（Feature Flags）

该 crate 使用 feature 进行模块化编译：

| Feature            | 说明                                  |
|--------------------|---------------------------------------|
| `machine_learning` | 经典机器学习算法（启用 `math`）       |
| `neural_network`   | 神经网络框架（启用 `math`）           |
| `utils`            | 数据预处理与数据集划分（启用 `math`） |
| `metrics`          | 评估指标（启用 `math`）               |
| `math`             | 数值原语（距离、矩阵乘积、并行归约）  |
| `full`             | 以上全部模块                          |
| `default`          | `full`                                |
| `show_progress`    | 在终端渲染训练/迭代进度条             |

## 项目状态

RustyML 正在积极开发中。API 正在趋于稳定，但在 `1.0.0` 之前，次要版本更新中仍可能出现破坏性更改。

## 贡献

欢迎贡献。如果你想帮助构建这个 Rust 机器学习库，可以：

1. 提交 issue 反馈 bug 或功能需求
2. 提交 pull request 改进代码
3. 就 API 设计提供反馈
4. 完善文档与示例

也请阅读[行为准则](https://github.com/SomeB1oody/RustyML/blob/master/CODE_OF_CONDUCT.md)。

## 作者

SomeB1oody（[stanyin64@gmail.com](mailto:stanyin64@gmail.com)）

## 许可证

本项目遵循 [MIT 许可证](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)。详情请参阅 LICENSE 文件。
