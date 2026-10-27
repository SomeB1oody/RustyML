[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

一个用**纯 Rust** 编写的机器学习与深度学习库。

[![rustc](https://img.shields.io/badge/rustc-1.89%2B-brown)](https://www.rust-lang.org/)
[![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
[![crates.io](https://img.shields.io/crates/v/rustyml.svg)](https://crates.io/crates/rustyml)
[![ci](https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/ci.yml?branch=master&label=ci)](https://github.com/SomeB1oody/RustyML/actions/workflows/ci.yml)

> 阅读 **[RustyML 使用指南](https://someb1oody.github.io/RustyML/zh-Hans/)**，获取 RustyML 的详细文档与教程。

## 概述

RustyML 提供数据预处理、机器学习模型、神经网络和评估指标。
它使用 `ndarray` 数组和 Rust 数值内核。

## 功能

- **并行内核**：部分内核使用 [Rayon](https://github.com/rayon-rs/rayon)。工作量阈值决定使用串行还是并行路径。
- **模型**：监督学习、无监督学习、异常检测、顺序网络和图网络。
- **随机种子**：`set_global_seed` 设置调用线程的种子流。支持 `random_state` 的组件可用本地种子覆盖此设置。
- **二进制文件**：支持持久化的模型使用 [Serde](https://serde.rs/) 和 [postcard](https://docs.rs/postcard/) 保存及加载拟合状态。
- **网络检查点**：保存模型数组，再将其加载到兼容的已构建模型中。检查点不恢复优化器或损失函数。
- **评估指标**：回归、二分类、多分类和聚类。各指标的输入规则和分数定义见 API 文档。

重复相同的随机操作序列前，请重置全局种子。
其他线程有独立的种子流。
部分分解求解器使用独立的种子。

## 安装

在 `Cargo.toml` 中添加 RustyML：

```toml
[dependencies]
rustyml = "0.17.0"
ndarray = "0.17"
```

关闭默认特性后，可单独选择所需模块：

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

**最低支持 Rust 版本（MSRV）：** Rust 1.89+（edition 2024）。

## 快速上手

### 经典机器学习

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

### 神经网络

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

### 图模型

顺序模型将每层的输出传给下一层。
`GraphBuilder` 将层调用连接为图节点。
图模型支持多个输入、多个输出、共享层和残差连接。
合并层负责汇合分支：`Add`、`Subtract`、`Multiply`、`Average`、`Maximum`、`Minimum` 和 `Concatenate`。

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

### 评估模型

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

## 模块

公开模块按任务划分 API：

- `machine_learning`：分类、回归、聚类、分解和异常检测模型。
- `neural_network`：层、模型构建器、损失函数、优化器和检查点。
- `utils`：预处理、标签转换、缩放器和数据集划分。
- `metrics`：分类、回归和聚类指标。
- `math`：距离函数、矩阵运算和归约。
- `error`：共用的错误类型和结果别名。
- `random`：线程本地种子控制。
- `traits`：模型共用的接口。
- `tuning`：串行与并行工作的运行时阈值。
- `prelude`：公开类型和接口的分组导入。

参见 [API 文档](https://docs.rs/rustyml/latest/rustyml/)。

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

RustyML 正在开发中。在 `1.0.0` 之前，次要版本更新仍可能更改 API。

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
