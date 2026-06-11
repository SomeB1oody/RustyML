[简体中文](https://github.com/SomeB1oody/RustyML/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/RustyML/blob/master/README.md)

# RustyML

一个用**纯 Rust** 编写的高性能机器学习与深度学习库。

<p align="center">
  <a href="https://www.rust-lang.org/"><img alt="rustc" src="https://img.shields.io/badge/rustc-1.89%2B-brown"></a>
  <a href="https://doc.rust-lang.org/edition-guide/"><img alt="edition" src="https://img.shields.io/badge/edition-2024-orange"></a>
  <a href="https://github.com/SomeB1oody/RustyML/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="https://crates.io/crates/rustyml"><img alt="crates.io" src="https://img.shields.io/crates/v/rustyml.svg"></a>
  <br>
  <a href="https://github.com/SomeB1oody/RustyML/actions/workflows/fmt.yml"><img alt="fmt" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/fmt.yml?branch=master&label=fmt"></a>
  <a href="https://github.com/SomeB1oody/RustyML/actions/workflows/clippy.yml"><img alt="clippy" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/clippy.yml?branch=master&label=clippy"></a>
  <a href="https://github.com/SomeB1oody/RustyML/actions/workflows/test.yml"><img alt="test" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/test.yml?branch=master&label=test"></a>
  <a href="https://github.com/SomeB1oody/RustyML/actions/workflows/doc.yml"><img alt="doc" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/RustyML/doc.yml?branch=master&label=doc"></a>
</p>

## 概述

RustyML 是一个完整的机器学习与深度学习生态，完全用 Rust 端到端实现，不依赖任何 C/C++ 代码。
它覆盖从数据预处理、特征工程，到模型训练、评估的全流程，同时充分利用 Rust 的内存安全、无畏并发
和零成本抽象。

整个库被划分为六个由 feature 控制的模块，你只需编译用得上的部分：
`machine_learning`、`neural_network`、`utils`、`metrics`、`math`，以及共享的 `prelude`。

## 核心亮点

- **纯 Rust，无 FFI**——内存安全、可移植，无需链接任何外部库。
- **默认并行**——计算密集的内核使用 [Rayon](https://github.com/rayon-rs/rayon) 进行多线程计算。
- **算法覆盖广**——经典的监督/无监督学习、异常检测，以及完整的神经网络框架。
- **统一的结构化错误处理**——所有可能失败的调用都返回 `RustymlResult<T>`；错误被归类为清晰的类别变体，而非难以解析的字符串。
- **可复现性**——一次 `set_global_seed` 调用即可让所有随机化组件变得确定。
- **模型持久化**——通过 [Serde](https://serde.rs/) 将训练好的模型和网络权重以 JSON 形式保存与加载。
- **丰富的评估指标**——回归、分类（二分类与多分类）、聚类，遵循 scikit-learn 的约定。
- **模块化 feature**——可以只引入 `metrics`、只引入 `math`、引入 `default` 学习栈，或引入 `full` 全量。

## 安装

在 `Cargo.toml` 中添加 RustyML：

```toml
[dependencies]
rustyml = { version = "0.12", features = ["full"] }
ndarray = "0.17"
```

按需选择 feature 组合：

```toml
# 默认：经典机器学习 + 神经网络
rustyml = "0.12"

# 仅神经网络框架
rustyml = { version = "0.12", features = ["neural_network"] }

# 全部模块（ml、nn、utils、metrics、math）
rustyml = { version = "0.12", features = ["full"] }

# 训练时在终端显示进度条
rustyml = { version = "0.12", features = ["full", "show_progress"] }
```

> **最低支持 Rust 版本（MSRV）：** Rust 1.89+（edition 2024）。

## 快速上手

### 经典机器学习

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

// 训练一个不带正则化的线性回归模型
let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None).unwrap();

let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
let y = array![6.0, 9.0, 12.0];

model.fit(&x, &y).unwrap();
let predictions = model.predict(&x).unwrap();
println!("{:?}", predictions);

// 保存并重新加载训练好的模型
model.save_to_path("linear_regression.json").unwrap();
let restored = LinearRegression::load_from_path("linear_regression.json").unwrap();
```

### 神经网络

```rust
use rustyml::neural_network::sequential::Sequential;
use rustyml::prelude::neural_network::*;
use ndarray::Array;

// 32 个样本，784 个输入特征，10 个输出类别
let x = Array::ones((32, 784)).into_dyn();
let y = Array::ones((32, 10)).into_dyn();

let mut model = Sequential::new();
model
    .add(Dense::new(784, 128, Activation::ReLU, None).unwrap())
    .add(Dense::new(128, 64, Activation::ReLU, None).unwrap())
    .add(Dense::new(64, 10, Activation::Softmax, None).unwrap())
    .compile(
        Adam::new(0.001, 0.9, 0.999, 1e-8, None).unwrap(),
        CategoricalCrossEntropy::new(),
    );

model.summary(); // 打印网络结构
model.fit(&x, &y, 10).unwrap();

let predictions = model.predict(&x).unwrap();
println!("预测结果形状: {:?}", predictions.shape());

// 保存训练好的权重，之后可加载到新模型中
model.save_to_path("model.json").unwrap();
```

### 评估模型

```rust
use rustyml::metrics::*;
use ndarray::array;

// 参数顺序始终是 (y_true, y_pred)，与 scikit-learn 一致
let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view());
println!("准确率: {:.3}", cm.accuracy());
println!("F1 分数: {:.3}", cm.f1_score());
```

## 模块

### `machine_learning`

经典的监督与无监督学习算法，均带有并行优化、输入校验和 JSON 持久化能力。

| 类别 | 算法 |
|------|------|
| **回归** | 线性回归（可选 L1/L2 正则化） |
| **分类** | 逻辑回归、K 近邻、决策树（ID3 / C4.5 / CART）、SVC（核 SMO）、Linear SVC、线性判别分析（LDA） |
| **聚类** | KMeans（K-means++ 初始化）、DBSCAN、MeanShift |
| **异常检测** | 隔离森林（Isolation Forest） |

共享的配置类型位于 [`types`](https://docs.rs/rustyml/latest/rustyml/types/index.html) 模块：
`DistanceCalculationMetric`（欧几里得 / 曼哈顿 / 闵可夫斯基）、`RegularizationType`（L1 / L2）、
以及 `KernelType`（Linear / Poly / RBF / Sigmoid / Cosine）。所有模型都实现统一的 `Fit` 与
`Predict` trait。

### `neural_network`

一个完整的框架，通过 Keras 风格的 `Sequential` API 构建、训练并序列化前馈、卷积及循环网络。

- **核心层** - `Dense`、`Flatten`
- **激活** - `ReLU`、`Sigmoid`、`Tanh`、`Softmax`、`Linear`（可用 `Activation` 枚举或独立的激活层）
- **卷积** - `Conv1D`、`Conv2D`、`Conv3D`、`DepthwiseConv2D`、`SeparableConv2D`
- **池化** - 1D / 2D / 3D 的最大池化与平均池化，以及它们对应的全局变体
- **循环** - `SimpleRNN`、`LSTM`、`GRU`
- **正则化** - `Dropout`、`SpatialDropout{1,2,3}D`、`GaussianNoise`、`GaussianDropout`
- **归一化** - `BatchNormalization`、`LayerNormalization`、`InstanceNormalization`、`GroupNormalization`
- **优化器** - `SGD`（支持动量）、`Adam`、`RMSprop`、`AdaGrad`
- **损失函数** - `MeanSquaredError`、`MeanAbsoluteError`、`BinaryCrossEntropy`、`CategoricalCrossEntropy`、`SparseCategoricalCrossEntropy`

训练支持全批量（`fit`）与小批量（`fit_with_batches`）循环、权重查看（`get_weights`），
以及 JSON 序列化（`save_to_path` / `load_from_path`）。

### `utils`

数据预处理与降维。

- **降维** - `PCA`（多种 SVD 求解器）、`KernelPCA`（RBF / Linear / Poly / Sigmoid / Cosine 核）、`TSNE`
- **缩放** - `standardize`（z-score 标准化）、`normalize`（可配置轴与范数阶数）
- **标签编码** - `to_categorical`、`to_categorical_with_mapping`、`to_sparse_categorical`
- **数据划分** - `train_test_split`，比例可配置

### `metrics`

一套广泛的评估指标。所有函数都以 `(y_true, y_pred)` 为参数，并在违反前置条件时（长度不匹配、
输入为空）直接 panic 而非返回 `Result`，从而让这个叶子模块保持轻量、依赖极少。

- **回归** - MSE、RMSE、MAE、中位数绝对误差、MAPE、R²、可解释方差
- **分类** - 准确率、`ConfusionMatrix` 与 `MulticlassConfusionMatrix`、ROC AUC、对数损失、Cohen's κ、top-k 准确率、平均精度、ROC 与精确率-召回率曲线
- **聚类** - 调整兰德指数、标准化 / 调整互信息、同质性 / 完整性 / V-measure、Fowlkes–Mallows、轮廓系数、Davies–Bouldin、Calinski–Harabasz

### `math`

整个库共享的纯函数式数值原语：不纯度度量（`entropy`、`gini`）、距离
（`squared_euclidean_distance_row`、`manhattan_distance_row`、`minkowski_distance_row`）、
统计量（`variance`、`standard_deviation`、`sum_of_square_total`、`sum_of_squared_errors`），
以及激活/损失辅助函数（`sigmoid`、`logistic_loss`、`hinge_loss`）。

### `prelude`

按领域拆分的一站式导入，让你只引入需要的部分：

```rust
use rustyml::prelude::machine_learning::*; // 机器学习模型、trait、配置枚举
use rustyml::prelude::neural_network::*;   // 层、优化器、损失函数
use rustyml::prelude::utils::*;            // PCA、t-SNE、缩放、数据划分
use rustyml::prelude::metrics::*;          // 评估指标
use rustyml::prelude::math::*;             // 数学原语
```

## 特性标志（Feature Flags）

该 crate 使用 feature 进行模块化编译：

| Feature | 说明 |
|---------|------|
| `machine_learning` | 经典机器学习算法（启用 `math`） |
| `neural_network` | 神经网络框架 |
| `utils` | 数据预处理与降维（启用 `math`） |
| `metrics` | 评估指标（启用 `math`） |
| `math` | 数学与统计原语 |
| `default` | `machine_learning` + `neural_network` |
| `full` | 以上全部模块 |
| `show_progress` | 在终端渲染训练/迭代进度条 |

## 可复现性

每个随机化组件（权重初始化、K-means++、隔离森林、t-SNE、dropout……）都会将其
`random_state: Option<u64>` 解析到一个共享入口。只需设置一个全局种子，整个库即变得确定：

```rust
use rustyml::set_global_seed;

set_global_seed(42);
// ……训练模型；结果在多次运行间可复现...
```

单次调用传入的 `random_state` 优先级高于全局种子，全局种子又高于系统熵。完整的解析规则请见
[`random`](https://docs.rs/rustyml/latest/rustyml/random/index.html) 模块。

## 错误处理

除 `metrics` 和 `math` 这两个叶子模块外，所有可能失败的操作都返回 `RustymlResult<T>`
（即 `Result<T, error::Error>` 的别名）。`Error` 类型被组织为多个类别变体，并将领域相关的失败
归入嵌套的 `NnError`、`TreeError`、`IoError` 子枚举，因此你可以精确匹配出错原因，而无需解析字符串。

## 项目状态

RustyML 正在积极开发中。API 正在趋于稳定，但在 `1.0.0` 之前，次要版本更新中仍可能出现破坏性更改。

## 贡献

欢迎贡献！如果你有兴趣帮助构建一个强大的 Rust 机器学习生态，你可以：

1. 提交 issue 反馈 bug 或功能需求
2. 提交 pull request 改进代码
3. 就 API 设计提供反馈
4. 完善文档与示例

也请阅读[行为准则](https://github.com/SomeB1oody/RustyML/blob/master/CODE_OF_CONDUCT.md)。

## 作者

SomeB1oody — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)

## 许可证

基于 [MIT 许可证](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)授权。详情请参阅 LICENSE 文件。
