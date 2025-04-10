# RustyML
A comprehensive machine learning and deep learning library written in pure Rust.
一个用纯Rust编写的全面机器学习和深度学习库。

## Overview | 概述
Rust AI aims to be a feature-rich machine learning and deep learning framework that leverages Rust's performance, memory safety, and concurrency features. While currently in early development stages with foundational components, the project's long-term vision is to provide a complete ecosystem for machine learning, deep learning, and transformer-based models.
Rust AI 旨在成为一个功能丰富的机器学习和深度学习框架，充分利用Rust的性能、内存安全性和并发特性。虽然目前处于早期开发阶段，只实现了基础组件，但项目的长期愿景是提供一个完整的机器学习、深度学习和基于transformer架构的模型生态系统。

## Current Features | 当前功能
- **Mathematical Utilities | 数学工具**:
  - Sum of square total (SST) for measuring data variability | 总平方和(SST)，用于衡量数据变异性
  - Sum of squared errors (SSE) for evaluating prediction errors | 误差平方和(SSE)，用于评估预测误差
  - Sigmoid function for logistic regression and neural networks | Sigmoid函数，用于逻辑回归和神经网络
  - Logistic loss (log loss) for binary classification models | 逻辑损失函数(对数损失)，用于二元分类模型
  - Accuracy score for classification model evaluation | 准确率分数，用于分类模型评估
  - Calculate the squared Euclidean distance between two points | 计算两点之间的欧几里得距离平方
  - Calculate the Manhattan distance between two points ｜ 计算两点间的曼哈顿距离
  - Calculate the Minkowski distance between two points ｜ 计算两点间的闵可夫斯基距离
  - Calculate the variance | 计算方差
  - Calculates the entropy of a label set | 计算标签集的熵
  - Calculates the Gini impurity of a label set | 计算标签集的基尼不纯度
  - Calculates the information gain when splitting a dataset | 计算数据集分割时的信息增益
  - Calculates the gain ratio for a dataset split | 计算数据集分割的增益率
  - Calculates the leaf node adjustment factor c(n) | 计算叶节点调整因子 c(n)
  - Calculates the standard deviation of a set of values | 计算标准差


- **Metric Utilities | 度量工具**:
  - R-squared (R²) score for assessing model fit quality | R平方(R²)分数，用于评估模型拟合质量
  - Calculates the Root Mean Squared Error (RMSE) between predicted and actual values | 计算均方误差根
  - Calculates the Mean Squared Error (MSE) of a set of values | 计算一组值的均方误差(MSE)
  - Calculates the Mean Absolute Error (MAE) between predicted and actual values | 计算平均绝对误差
  - Confusion Matrix for binary classification evaluation | 混淆矩阵及相关计算公式
  - Calculates the Normalized Mutual Information (NMI) between two cluster label assignments.(Unit: nat) | 计算NMI值(单位: nat)
  - Calculates the Adjusted Mutual Information (AMI) between two cluster label assignments.(Unit: nat) | 计算AMI值(单位: nat)
  - Calculates the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) | 计算AUC-ROC值
  - Calculates the Mean Squared Error (MSE) of a set of values | 计算一组值的均方误差(MSE)


- **Machine Learning Models | 机器学习模型**:
  - Supervised Learning | 监督式学习:
    - Linear Regression | 线性回归
    - Logistic Regression | 逻辑回归
    - KNN | K邻近值聚类
    - Decision Tree | 决策树
    - SVC(Support Vector Classification) | 支持向量机分类
    - Linear SVC | 线性向量机分类
    - LDA(Linear Discriminant Analysis) | 线性判别分析
    - Kernel PCA | 核主成分分析

  - Unsupervised Learning | 无监督学习:
    - KMeans | K均值聚类
    - MeanShift | MeanShift聚类
    - DBSCAN | DBSCAN聚类
    - Isolation Forest | 隔离森林


- **Utilities | 工具**:
  - PCA(Principal Component Analysis) | 主成分分析
  - standardize data($\mu = 0$, $\sigma = 1$) | 标准化数据(确保均值$\mu = 0$, 协方差$\sigma = 1$)
  - Split dataset for training and dataset for test | 分离训练集和测试集
  - LDA(Linear Discriminant Analysis) | 线性判别分析
  - t-Distributed Stochastic Neighbor Embedding (t-SNE) | t分布随机邻域嵌入


- **Datasets | 数据集**:
  - Iris dataset | 鸢尾花分类数据集
  - Diabetes dataset | 糖尿病回归数据集


- **Neural Network(initial implementation) | 神经网络(初步实现)**:

  - model | 模型:
    - Sequential | 顺序模型

  - Layer | 层:
    - Dense | 全连接层

  - Loss function | 损失计算函数:
    - Mean Squared Error | 均方误差
    - Mean Absolute Error | 平均绝对误差
    - Binary Cross Entropy | 二元交叉熵
    - Categorical Cross Entropy | 分类交叉熵
    - Sparse Categorical Cross Entropy | 稀疏分类交叉熵

  - Optimizer | 优化器:
    - SGD | 随机梯度下降
    - Adam | 自适应矩估计
    - RMSprop | 均方根传播

  - Activation Function | 激活函数:
    - ReLU | 修正线性单元（线性整流函数）
    - Tanh | 双曲正切函数
    - Sigmoid | 逻辑函数
    - Softmax | 归一化指数函数

## Vision | 愿景
While the library is in its early stages, Rust AI aims to evolve into a comprehensive crate that includes:
虽然该库处于早期阶段，但Rust AI旨在发展成为一个包含以下内容的综合性crate：
- **Classical Machine Learning Algorithms | 经典机器学习算法**:
    - Linear and Logistic Regression | 线性和逻辑回归
    - Decision Trees | 决策树
    - Clustering algorithms (K-means, MeanShift, KNN) | 聚类算法(K均值, MeanShift, KNN)
    - Dimensionality reduction technique (PCA) | 降维技术(PCA)

- **Deep Learning | 深度学习**:
    - Neural network building blocks | 神经网络构建模块
    - Convolutional neural networks | 卷积神经网络
    - Recurrent neural networks | 循环神经网络
    - Optimization algorithms | 优化算法

- **Transformer Architecture | Transformer架构**:
    - Self-attention mechanisms | 自注意力机制
    - Multi-head attention | 多头注意力
    - Encoder-decoder architectures | 编码器-解码器架构
    - Pre-training and fine-tuning capabilities | 预训练和微调能力

- **Utilities | 实用工具**:
    - Data preprocessing | 数据预处理
    - Cross-validation | 交叉验证
    - Performance metrics | 性能指标
    - Visualization helpers | 可视化辅助工具

## Dependencies | 依赖
- [ndarray](https://crates.io/crates/ndarray) (0.16.1): N-dimensional array library for Rust | Rust的N维数组库
- [rand](https://crates.io/crates/rand) (0.9.0): Random number generators and other randomness functionality for Rust | Rust的随机数生成器和其他随机性功能
- [ndarray-linalg](https://crates.io/crates/ndarray-linalg)(0.17.0): Linear algebra package for rust-ndarray | 给 Rust ndarray使用的线性代数包
- [statrs](https://crates.io/crates/statrs)(0.18.0): A host of statistical utilities for Rust scientific computing | Rust 科学计算的一整套统计工具
- [rand_distr](https://crates.io/crates/rand_distr)(0.5.1) Sampling from random number distributions | 从随机数分布中采样
- [rayon](https://crates.io/crates/rayon)(1.10.0) a data-parallelism library for Rust | Rust的数据并行库

## Getting Started | 开始使用
Add the library to your `Cargo.toml`:
将库添加到您的`Cargo.toml`文件中：
``` toml
[dependencies]
rustyml = "0.4.0"
```
Example usage | 使用示例:
``` rust
use rustyml::machine_learning::linear_regression::LinearRegression;
use ndarray::{Array1, Array2, array};

// Create a linear regression model
let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6);

// Prepare training data
let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
let raw_y = vec![6.0, 9.0, 12.0];

// Convert Vec to ndarray types
let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
let y = Array1::from_vec(raw_y);

// Train the model
model.fit(x.view(), y.view()).unwrap();

// Make predictions
let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
let predictions = model.predict(new_data.view());

// Since Clone is implemented, the model can be easily cloned
let model_copy = model.clone();

// Since Debug is implemented, detailed model information can be printed
println!("{:?}", model);
```

For Chinese mainland users, I provide mandarin video tutorial on Bilibili, follow [@SomeB1oody](https://space.bilibili.com/1349872478)
对于中国大陆用户,我在B站提供视频教程,关注[@SomeB1oody](https://space.bilibili.com/1349872478)

## Project Status | 项目状态
This project is in the **early development stage**. Currently, only a small subset of the planned functionality has been implemented. The API is unstable and subject to significant changes as the project evolves.
该项目处于**早期开发阶段**。目前，仅实现了计划功能的一小部分。API不稳定，随着项目的发展可能会有重大变化。

## Contribution | 贡献
Contributions are welcome! If you're interested in helping build a robust machine learning ecosystem in Rust, please feel free to:
欢迎贡献！如果您有兴趣帮助构建Rust中的强大机器学习生态系统，请随时：
1. Submit issues for bugs or feature requests | 提交bug或功能请求
2. Create pull requests for improvements | 创建改进的拉取请求
3. Provide feedback on the API design | 提供API设计的反馈意见
4. Help with documentation and examples | 帮助完善文档和示例

## License | 许可证
[MIT - License](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)
## Authors | 作者
- SomeB1oody (stanyin64@gmail.com)

_RustyML - Bringing the power and safety of Rust to machine learning and AI._
_RustyML - 将Rust的强大性能和安全特性引入机器学习和人工智能领域。_