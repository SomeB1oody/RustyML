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

> **[RustyML 使用指南](https://someb1oody.github.io/RustyML/zh-Hans/)** 覆盖所有模块，从第一个模型到性能调优。

## 概述

RustyML 是一个机器学习与深度学习库，完全用 Rust 端到端实现，不依赖任何 C 或 C++ 代码。它覆盖完整的
工作流程：数据预处理、特征工程、模型训练和评估。它利用了 Rust 的内存安全、安全并发和零成本抽象。

RustyML 被划分为 5 个由 feature 控制的模块，你只需编译用得上的部分：
`machine_learning`、`neural_network`、`utils`、`metrics`、`math`，外加一个共享的 `prelude`。

## 核心亮点

- **纯 Rust，无 FFI**：内存安全、可移植，无需链接任何外部库。
- **默认并行**：计算密集的内核使用 [Rayon](https://github.com/rayon-rs/rayon) 进行多线程计算。
- **算法覆盖**：经典的监督与无监督学习、异常检测，以及一个神经网络框架。
- **结构化错误处理**：所有可能失败的调用都返回 `RustymlResult<T>`。错误被归类到不同的类别变体中，而不是普通字符串。
- **可复现**：一次 `set_global_seed` 调用即可让调用线程上所有随机化组件变得确定。按组件设置的 `random_state` 覆盖其余情形。
- **模型持久化**：通过 [Serde](https://serde.rs/) 和 [postcard](https://docs.rs/postcard/) 将训练好的模型和网络权重保存为紧凑的二进制格式。
- **评估指标**：回归、分类（二分类与多分类）、聚类，遵循 scikit-learn 的约定。
- **与 scikit-learn 校验**：估计器的默认配置、评分符号和指标的输出约定都与 scikit-learn 1.9 做了数值比对。移植过来的流水线能给出相同的结果。有意保留的差异都在相应位置注明。
- **模块化 feature**：默认打开整个 crate。关掉默认后，只添加 `metrics`、只添加 `math`，或任意你需要的子集。

## 安装

在 `Cargo.toml` 中添加 RustyML：

```toml
[dependencies]
rustyml = "*"
ndarray = "0.17"
```

默认 feature 集是 `full`，所以所有模块都在。一段从 scikit-learn 移植过来的脚本本来就会横跨它们
（`utils::train_test_split` -> `machine_learning` -> `metrics`）。想把构建裁小，就关掉默认并显式列出所需模块：

```toml
# 全部模块（ml、nn、utils、metrics、math）
rustyml = "*"

# 仅神经网络框架
rustyml = { version = "*", default-features = false, features = ["neural_network"] }

# 仅评估指标
rustyml = { version = "*", default-features = false, features = ["metrics"] }

# 训练时在终端显示进度条
rustyml = { version = "*", features = ["show_progress"] }
```

Cargo 的 feature 是叠加的。写上一个并不会关掉其他的。单写 `features = ["metrics"]` 编译出来的
仍然是完整的 crate。真正起裁剪作用的是 `default-features = false` 这一行。

> **最低支持 Rust 版本（MSRV）：** Rust 1.89+（edition 2024）。

## 快速上手

### 经典机器学习

```rust
use rustyml::prelude::machine_learning::*;
use ndarray::array;

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
```

### 神经网络

```rust
use rustyml::prelude::neural_network::*;
use ndarray::Array;

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
```

### 评估模型

```rust
use rustyml::metrics::*;
use ndarray::array;

// 参数顺序始终是 (y_true, y_pred)，与 scikit-learn 一致
// ConfusionMatrix::new 只接受 0.0/1.0 硬标签（其他标签取值对用 new_with_labels）
let y_true = array![1.0, 0.0, 0.0, 1.0, 1.0];
let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0];

// 两个参数的存储类型相互独立，持有所有权的数组和视图可以混用
let cm = ConfusionMatrix::new(&y_true, &y_pred.view());
println!("Accuracy: {:.3}", cm.accuracy());
println!("F1 score: {:.3}", cm.f1_score());
```

## 模块

### `machine_learning`

经典的监督与无监督学习算法，带有并行优化、输入校验和二进制持久化能力。

| 类别 | 算法 |
|------|------|
| **回归** | 线性回归（默认闭式 OLS，也可切换梯度下降，并可选 L1/L2 正则化） |
| **分类** | 逻辑回归、K 近邻、决策树（ID3 / C4.5 / CART）、SVC（核 SMO）、Linear SVC、线性判别分析（LDA） |
| **聚类** | KMeans（K-means++ 初始化，`n_init` 次重启）、DBSCAN、MeanShift（平坦核） |
| **降维** | PCA（多种 SVD 求解器）、KernelPCA（RBF / Linear / Poly / Sigmoid / Cosine 核）、t-SNE |
| **异常检测** | 隔离森林（Isolation Forest） |

3 个聚类估计器返回的都是 `Array1<isize>` 标签，`-1` 表示噪声或未归属的点。每个聚类指标接收的也正是
这个类型，所以任意估计器都能直接喂进任意指标。`IsolationForest` 遵循 scikit-learn 的符号约定。
`score_samples` 返回 `[-1, 0)` 区间内的值，越低越异常。`predict` 对离群点返回 `-1`，对正常点返回 `+1`。

共享的配置类型位于 [`machine_learning::types`](https://docs.rs/rustyml/latest/rustyml/machine_learning/types/index.html) 模块：
`RegularizationType`（L1 或 L2）、`Gamma`，以及 `KernelType`（Linear、Poly、RBF、Sigmoid 或 Cosine）。
`RegularizationType` 的文档里带有一张换算表，说明如何把 scikit-learn 的惩罚强度
（`Lasso` 或 `Ridge` 的 `alpha`、`LogisticRegression` 的 `C`）折算过来。L1 走的是邻近算子软阈值，
所以被惩罚的系数会恰好归零。

`DistanceCalculationMetric`（欧几里得、曼哈顿或闵可夫斯基）调度器定义于 `math` 模块，并在
`machine_learning` 根重新导出。预测类模型实现共用的 `Fit` 与 `Predict` trait。降维变换器
（[`decomposition`](https://docs.rs/rustyml/latest/rustyml/machine_learning/decomposition/index.html)
与 [`manifold`](https://docs.rs/rustyml/latest/rustyml/machine_learning/manifold/index.html)）
则实现 `Transform` 与 `FitTransform`。这 4 个 trait 统一定义在 crate 根的
[`traits`](https://docs.rs/rustyml/latest/rustyml/traits/index.html) 模块，与 `utils` 中的
有状态预处理变换器共用。

### `neural_network`

一个框架，通过 Keras 风格的 `Sequential` API 构建、训练并序列化前馈、卷积和循环网络。张量采用
channels-last，卷积核形状与 Keras 相同。你在 Keras 里熟悉的那套布局可以原样搬过来。

- **核心层**：`Dense`、`Flatten`
- **激活**：`ReLU`、`Sigmoid`、`Tanh`、`Softmax`、`Linear`（可用 `Activation` 枚举或独立的激活层）
- **卷积**：`Conv1D`、`Conv2D`、`Conv3D`、`DepthwiseConv2D`、`SeparableConv2D`
- **池化**：1D / 2D / 3D 的最大池化与平均池化，以及它们对应的全局变体
- **循环**：`SimpleRNN`、`LSTM`、`GRU`
- **正则化**：`Dropout`、`SpatialDropout{1,2,3}D`、`GaussianNoise`、`GaussianDropout`
- **归一化**：`BatchNormalization`、`LayerNormalization`、`InstanceNormalization`、`GroupNormalization`
- **优化器**：`SGD`（支持动量）、`Adam`、`AdamW`、`RMSprop`、`AdaGrad`
- **损失函数**：`MeanSquaredError`、`MeanAbsoluteError`、`BinaryCrossEntropy`、`CategoricalCrossEntropy`、`SparseCategoricalCrossEntropy`

训练支持全批量循环（`fit`）与小批量循环（`fit_with_batches`）、权重查看（`get_weights`），以及
二进制序列化（`save_to_path` 与 `load_from_path`）。

两个循环都返回一个 `History`，其中每个 epoch 对应 1 个损失值。每个值都是在该 epoch 进行中测得的，
而不是结束后，这与 Keras 一致。它取自每个批次自身权重更新之前的那次前向传播。所以每一项描述的是
该 epoch 运行时所持有的权重，而不是它结束时的权重。

要给手上这个模型打分，请使用 `evaluate`。这个调用是一次推理模式的前向传播。它不涉及梯度、参数，
也不涉及批归一化的滑动统计量。

`train_batch`（即 Keras 的 `train_on_batch`）也是公开的。因此自定义循环可以自己掌控 epoch 结构。
它不需要重新实现前向传播、损失计算、反向传播、梯度裁剪和参数更新这些步骤。

### `utils`

数据预处理与数据集划分。降维（`PCA`、`KernelPCA`、`TSNE`）现已迁至 `machine_learning` 下的
`decomposition` 与 `manifold`。

- **缩放（一次性）**：`standardize`（z-score 标准化）、`normalize`（可配置轴与范数阶数）
- **缩放（有状态）**：`StandardScaler`、`MinMaxScaler`、`MaxAbsScaler`、`RobustScaler`（中位数与四分位距，抗离群点）与 `Normalizer`（按样本归一化）。每个缩放器都支持 `fit`、`transform`、`fit_transform`，并会保存训练集的统计量供后续批次使用。`StandardScaler`、`MinMaxScaler`、`MaxAbsScaler` 还额外支持 `partial_fit` 与 `inverse_transform`。`RobustScaler` 支持 `inverse_transform`，但不支持 `partial_fit`。`Normalizer` 两者都不支持。这 5 个缩放器都可以用 `save_to_path` 持久化。
- **标签编码**：`to_categorical`、`to_categorical_with_mapping`、`to_sparse_categorical`
- **数据划分**：`train_test_split` 与 `train_test_split_stratified`，比例可配置

### `metrics`

一套用于回归、分类和聚类的评估指标。所有函数都以 `(y_true, y_pred)` 为参数，并在违反前置条件时
（例如长度不匹配、输入为空、标签超出取值范围）直接 panic，而不是返回 `Result`，从而让这个叶子模块
保持轻量、依赖极少。

- **回归**：MSE、RMSE、MAE、中位数绝对误差、MAPE、R^2、可解释方差
- **分类**：准确率、`ConfusionMatrix` 与 `MulticlassConfusionMatrix`、ROC AUC、对数损失、Cohen's kappa、top-k 准确率、平均精度、ROC 与精确率-召回率曲线
- **聚类**：调整兰德指数、标准化与调整互信息、同质性、完整性、V-measure、Fowlkes-Mallows、轮廓系数、Davies-Bouldin、Calinski-Harabasz

曲线函数的输出点采用 scikit-learn 的排列顺序。`roc_curve` 是唯一一处有意的差异。它总是返回完整的
阈值扫描（相当于 scikit-learn 的 `drop_intermediate=False`）。这可能比 Python 的默认值产生更多点，
但描出的曲线和 `roc_auc` 完全相同。聚类指标接收 `isize` 标签数组，与聚类估计器的返回类型一致。

### `math`

整个库共享的纯函数式数值原语。`gemmkit` 支持的矩阵乘积覆盖 GEMM 与 GEMV。确定性分块并行归约也在
这里。成对距离函数（`squared_euclidean_distance_row`、`manhattan_distance_row`、
`minkowski_distance_row`）补全了这个模块，此外还有 `DistanceCalculationMetric`（欧几里得、曼哈顿
或闵可夫斯基）调度器。

### `prelude`

按领域拆分的单一导入入口，让你只导入需要的部分：

```rust
use rustyml::prelude::machine_learning::*; // 机器学习模型（含 PCA、KernelPCA、t-SNE）、trait、配置枚举
use rustyml::prelude::neural_network::*; // Sequential 与 History、层、优化器、损失函数
use rustyml::prelude::utils::*; // 缩放、标签编码、数据划分
use rustyml::prelude::metrics::*; // 评估指标
```

## 特性标志（Feature Flags）

该 crate 使用 feature 进行模块化编译：

| Feature | 说明 |
|---------|------|
| `machine_learning` | 经典机器学习算法（启用 `math`） |
| `neural_network` | 神经网络框架（启用 `math`） |
| `utils` | 数据预处理与数据集划分（启用 `math`） |
| `metrics` | 评估指标（启用 `math`） |
| `math` | 数值原语（距离、矩阵乘积、并行归约） |
| `full` | 以上全部模块 |
| `default` | `full` |
| `show_progress` | 在终端渲染训练/迭代进度条 |

## 可复现性

每个随机化组件（权重初始化、K-means++、隔离森林、t-SNE、dropout 等）都会将其
`random_state: Option<u64>` 解析到一个共享入口。只需设置一个全局种子，整个库即变得确定：

```rust
use rustyml::set_global_seed;

set_global_seed(42);
// ……训练模型。结果在多次运行间可复现……
```

单次调用传入的 `random_state` 优先级高于全局种子，全局种子又高于系统熵。完整的解析规则请见
[`random`](https://docs.rs/rustyml/latest/rustyml/random/index.html) 模块。

`KMeans` 会把拟合重启 `n_init` 次（默认 10 次）。每次重启都会根据 `random_state` 确定性地派生出
对应的 k-means++ 种子，所以已播种的拟合依然可复现。`IsolationForest` 每棵树的 RNG 建在 Rayon
worker 线程上，所以只有显式的 `random_state` 才能影响它们。

## 错误处理

除 `metrics` 和 `math` 这两个叶子模块外，所有可能失败的操作都返回 `RustymlResult<T>`
（即 `Result<T, rustyml::error::Error>` 的别名）。`Error` 类型把失败归类到不同的类别变体中。它把
领域相关的失败嵌套进 `NnError` 与 `TreeError`，把共享的 I/O 与序列化失败嵌套进 `IoError`。你可以
据此精确匹配出错原因，而无需解析字符串。

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
