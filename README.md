# RustyML
A comprehensive machine learning and deep learning library written in pure Rust.  
一个用纯Rust编写的全面机器学习和深度学习库。

[![Rust Version](https://img.shields.io/badge/Rust-v.1.85-brown)](https://www.rust-lang.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)  
[![ndarray](https://img.shields.io/badge/ndrarray-0.16.1-blue)](https://crates.io/crates/ndarray)  
[![rand](https://img.shields.io/badge/rand-0.9.2-blue)](https://crates.io/crates/rand)  
[![nalgebra](https://img.shields.io/badge/nalgebra-0.34.0-blue)](https://crates.io/crates/nalgebra)  
[![statrs](https://img.shields.io/badge/statrs-0.18.0-blue)](https://crates.io/crates/statrs)  
[![rand_distr](https://img.shields.io/badge/rand_distr-0.5.1-blue)](https://crates.io/crates/rand_distr)  
[![rayon](https://img.shields.io/badge/rayon-1.11.0-blue)](https://crates.io/crates/rayon)  
[![ndarray-rand](https://img.shields.io/badge/ndarray_rand-0.15.0-blue)](https://crates.io/crates/ndarray-rand)  
[![ahash](https://img.shields.io/badge/ahash-0.8.12-blue)](https://crates.io/crates/ahash)

## Overview | 概述
RustyML aims to be a feature-rich machine learning and deep learning framework that leverages Rust's performance, memory safety, and concurrency features. While currently in early development stages, the project's long-term vision is to provide a complete ecosystem for machine learning, deep learning, and transformer-based models.  
RustyML 旨在成为一个功能丰富的机器学习和深度学习框架，充分利用Rust的性能、内存安全性和并发特性。虽然目前处于早期开发阶段，但项目的长期愿景是提供一个完整的机器学习、深度学习和基于transformer架构的模型生态系统。

## Key Features | 核心特性
- **Pure Rust Implementation** | 纯Rust实现: No external C/C++ dependencies, ensuring memory safety and portability | 无需外部C/C++依赖，确保内存安全性和可移植性
- **Parallel Processing** | 并行处理: Leverages Rayon for efficient multi-threaded computation | 利用Rayon进行高效多线程计算
- **Rich Algorithm Collection** | 丰富的算法集合: Supervised, unsupervised learning, and neural networks | 监督学习、无监督学习和神经网络
- **Comprehensive Metrics** | 全面的评估指标: Evaluation tools for regression, classification, and clustering | 回归、分类和聚类的评估工具
- **Model Persistence** | 模型持久化: Save and load trained models with JSON serialization | 通过JSON序列化保存和加载训练好的模型
- **Standard Datasets** | 标准数据集: Built-in access to benchmark datasets for experimentation | 内置基准数据集用于实验
- **Feature Gated** | 特性门控: Modular design allowing you to compile only what you need | 模块化设计，只编译您需要的功能

## Architecture | 架构

### Machine Learning | 机器学习 (`features = ["machine_learning"]`)
Classical machine learning algorithms for supervised and unsupervised learning:  
经典机器学习算法，用于监督和无监督学习：

- **Regression | 回归**:
  - Linear Regression with L1/L2 regularization | 带L1/L2正则化的线性回归

- **Classification | 分类**:
  - Logistic Regression | 逻辑回归
  - KNN (K-Nearest Neighbors) | K近邻
  - Decision Tree (ID3, C4.5, CART) | 决策树（ID3、C4.5、CART）
  - SVC (Support Vector Classification) | 支持向量分类
  - Linear SVC | 线性支持向量分类
  - LDA (Linear Discriminant Analysis) | 线性判别分析

- **Clustering | 聚类**:
  - KMeans with K-means++ initialization | K均值（带K-means++初始化）
  - DBSCAN (Density-based clustering) | DBSCAN（基于密度的聚类）
  - MeanShift | 均值漂移

- **Anomaly Detection | 异常检测**:
  - Isolation Forest | 隔离森林

### Neural Network | 神经网络 (`features = ["neural_network"]`)
Complete neural network framework with flexible architecture design:  
完整的神经网络框架，具有灵活的架构设计：

- **Layers | 层**:
  - Dense (Fully connected) | 全连接层
  - RNN (Recurrent Neural Network) | 循环神经网络
  - LSTM (Long Short-Term Memory) | 长短期记忆网络
  - Convolution | 卷积层
  - Pooling (Max Pooling 1D/2D/3D, Global Pooling) | 池化层（最大池化1D/2D/3D，全局池化）
  - Dropout | Dropout层

- **Optimizers | 优化器**:
  - SGD (Stochastic Gradient Descent) | 随机梯度下降
  - Adam | Adam优化器
  - RMSProp | RMSProp优化器
  - AdaGrad | AdaGrad优化器

- **Loss Functions | 损失函数**:
  - MSE (Mean Squared Error) | 均方误差
  - MAE (Mean Absolute Error) | 平均绝对误差
  - Binary Cross-Entropy | 二元交叉熵
  - Categorical Cross-Entropy | 分类交叉熵
  - Sparse Categorical Cross-Entropy | 稀疏分类交叉熵

- **Models | 模型**:
  - Sequential architecture for feed-forward networks | 用于前馈网络的顺序架构

- **Activation Functions | 激活函数**:
  - ReLU, Tanh, Sigmoid, Softmax, Linear | ReLU、双曲正切、Sigmoid、Softmax、线性

### Utility | 工具 (`features = ["utility"]`)
Data preprocessing and dimensionality reduction utilities:  
数据预处理和降维工具：

- **Dimensionality Reduction | 降维**:
  - PCA (Principal Component Analysis) | 主成分分析
  - Kernel PCA (with RBF, Linear, Polynomial, Sigmoid kernels) | 核主成分分析（支持RBF、线性、多项式、Sigmoid核）
  - LDA (Linear Discriminant Analysis) | 线性判别分析
  - t-SNE (t-Distributed Stochastic Neighbor Embedding) | t-分布随机邻域嵌入

- **Preprocessing | 预处理**:
  - Standardization (z-score normalization) | 标准化（z分数归一化）
  - Train-test splitting | 训练测试集分割

- **Kernel Functions | 核函数**:
  - RBF, Linear, Polynomial, Sigmoid | RBF、线性、多项式、Sigmoid

### Metric | 评估指标 (`features = ["metric"]`)
Comprehensive evaluation metrics for model performance assessment:  
用于模型性能评估的全面评估指标：

- **Regression Metrics | 回归指标**:
  - MSE (Mean Squared Error) | 均方误差
  - RMSE (Root Mean Squared Error) | 均方根误差
  - MAE (Mean Absolute Error) | 平均绝对误差
  - R² score | R²分数

- **Classification Metrics | 分类指标**:
  - Accuracy | 准确率
  - Confusion Matrix (with TP, FP, TN, FN, precision, recall, F1-score) | 混淆矩阵（包含TP、FP、TN、FN、精确率、召回率、F1分数）
  - AUC-ROC | AUC-ROC曲线下面积

- **Clustering Metrics | 聚类指标**:
  - Adjusted Rand Index (ARI) | 调整兰德指数
  - Normalized Mutual Information (NMI) | 标准化互信息
  - Adjusted Mutual Information (AMI) | 调整互信息
  - Silhouette Score | 轮廓系数

### Math | 数学工具 (`features = ["math"]`)
Mathematical utilities and statistical functions:  
数学工具和统计函数：

- **Distance Metrics | 距离度量**:
  - Euclidean distance | 欧几里得距离
  - Manhattan distance | 曼哈顿距离
  - Minkowski distance | 闵可夫斯基距离

- **Impurity Measures | 不纯度度量**:
  - Entropy | 熵
  - Gini impurity | 基尼不纯度
  - Information Gain | 信息增益
  - Gain Ratio | 增益率

- **Statistical Functions | 统计函数**:
  - Variance | 方差
  - Standard deviation | 标准差
  - SST (Sum of Squares Total) | 总平方和
  - SSE (Sum of Squared Errors) | 误差平方和

- **Activation Functions | 激活函数**:
  - Sigmoid | Sigmoid函数
  - Logistic loss | 逻辑损失

### Dataset | 数据集 (`features = ["dataset"]`)
Access to standardized datasets for experimentation:  
用于实验的标准化数据集：

- Iris (150 samples, 4 features, 3 classes) | 鸢尾花数据集（150个样本，4个特征，3个类别）
- Diabetes (442 samples, 10 features) | 糖尿病数据集（442个样本，10个特征）
- Boston Housing | 波士顿房价数据集
- Wine Quality (red and white wines) | 葡萄酒质量数据集（红葡萄酒和白葡萄酒）
- Titanic | 泰坦尼克号数据集

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

## Getting Started | 开始使用

### Machine Learning Example | 机器学习示例

Add the library to your `Cargo.toml`:  
将库添加到您的`Cargo.toml`文件中：
``` toml
[dependencies]
rustyml = {version = "0.8.0", features = ["machine_learning"]} 
# or use `features = ["full"]` to enable all features
```

In your Rust code, write: 
在你的Rust代码里写：
``` rust
use rustyml::machine_learning::linear_regression::*; 
// or just use `rustyml::prelude::*;`  
use ndarray::{Array1, Array2, array};  
  
// Create a linear regression model  
let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6, None);  
  
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
  
// Save the trained model to a file   
model.save_to_path("linear_regression_model.json").unwrap();  
  
// Load the model from the file   
let loaded_model = LinearRegression::load_from_path("linear_regression_model.json").unwrap();  
  
// Use the loaded model for predictions    
let loaded_predictions = loaded_model.predict(new_data.view());  
  
// Since Clone is implemented, the model can be easily cloned  
let model_copy = model.clone();  
  
// Since Debug is implemented, detailed model information can be printed  
println!("{:?}", model);  
```

### Neural Network Example | 神经网络示例

Add the library to your `Cargo.toml`:  
将库添加到您的`Cargo.toml`文件中：
``` toml
[dependencies]
rustyml = {version = "0.8.0", features = ["neural_network"]} 
# or use `features = ["full"]` to enable all features
```

In your Rust code, write:
在你的Rust代码里写：
``` rust
use rustyml::prelude::*;  
use ndarray::Array;  
  
// Create training data   
let x = Array::ones((32, 784)).into_dyn(); // 32 samples, 784 features 
let y = Array::ones((32, 10)).into_dyn();  // 32 samples, 10 classes
  
// Build a neural network   
let mut model = Sequential::new();  
model  
    .add(Dense::new(784, 128, Activation::ReLU))    
    .add(Dense::new(128, 64, Activation::ReLU))    
    .add(Dense::new(64, 10, Activation::Softmax))    
    .compile(Adam::new(0.001, 0.9, 0.999, 1e-8), CategoricalCrossEntropy::new());  
// Display model structure   
model.summary();  
  
// Train the model  
model.fit(&x, &y, 10).unwrap();  
  
// Save model weights to file  
model.save_to_path("model.json").unwrap();  
  
// Create a new model with the same architecture  
let mut new_model = Sequential::new();
new_model  
    .add(Dense::new(784, 128, Activation::ReLU))    
    .add(Dense::new(128, 64, Activation::ReLU))    
    .add(Dense::new(64, 10, Activation::Softmax));
  
// Load weights from file  
new_model.load_from_path("model.json").unwrap();  
  
// Compile before using (required for training, optional for prediction)    
new_model.compile(Adam::new(0.001, 0.9, 0.999, 1e-8), CategoricalCrossEntropy::new());  
  
// Make predictions with loaded model  
let predictions = new_model.predict(&x);  
println!("Predictions shape: {:?}", predictions.shape());  
```

For Chinese mainland users, a mandarin video tutorial is provided on Bilibili, follow [@SomeB1oody](https://space.bilibili.com/1349872478)
对于中国大陆用户，B站有视频教程，关注[@SomeB1oody](https://space.bilibili.com/1349872478)

## Feature Flags | 特性标志

The crate uses feature flags for modular compilation:  
该crate使用特性标志进行模块化编译：

| Feature            | Description                                            | 说明                            |  
|--------------------|--------------------------------------------------------|-------------------------------|  
| `machine_learning` | Classical ML algorithms (depends on `math`, `utility`) | 经典机器学习算法（依赖于`math`、`utility`） |  
| `neural_network`   | Neural network framework                               | 神经网络框架                        |  
| `utility`          | Data preprocessing and dimensionality reduction        | 数据预处理和降维                      |  
| `metric`           | Evaluation metrics                                     | 评估指标                          |  
| `math`             | Mathematical utilities                                 | 数学工具                          |  
| `dataset`          | Standard datasets                                      | 标准数据集                         |  
| `full`             | Enables all features                                   | 启用所有功能                        |  

## Design Principles | 设计原则

- **Safety First**: Extensive input validation and comprehensive error types | **安全第一**: 广泛的输入验证和全面的错误类型
- **Performance**: Parallel processing via Rayon, efficient memory layouts with ndarray | **性能**: 通过Rayon实现并行处理，使用ndarray实现高效内存布局
- **Ergonomics**: Intuitive API design following Rust conventions | **人体工学**: 遵循Rust约定的直观API设计
- **Flexibility**: Customizable hyperparameters and configuration options | **灵活性**: 可自定义的超参数和配置选项
- **Reliability**: Robust implementations tested against standard datasets | **可靠性**: 经过标准数据集测试的健壮实现

## Project Status | 项目状态
RustyML is in active development. While the API is stabilizing, breaking changes may occur in minor version updates until version 1.0.0.
RustyML正在积极开发中。虽然API正在稳定，但在1.0.0版本之前，次要版本更新中可能会出现破坏性更改。

## Contribution | 贡献
Contributions are welcome! If you're interested in helping build a robust machine learning ecosystem in Rust, please feel free to:  
欢迎贡献！如果您有兴趣帮助构建Rust中的强大机器学习生态系统，请随时：
1. Submit issues for bugs or feature requests | 提交bug或功能请求
2. Create pull requests for improvements | 创建改进的拉取请求
3. Provide feedback on the API design | 提供API设计的反馈意见
4. Help with documentation and examples | 帮助完善文档和示例

## Authors | 作者
- SomeB1oody (stanyin64@gmail.com)

## License | 许可证
Licensed under the [MIT License](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE). See LICENSE file for details.  
根据[MIT许可证](https://github.com/SomeB1oody/RustyML/blob/master/LICENSE)授权。有关详细信息，请参阅LICENSE文件。
  
---  

_RustyML - Bringing the power and safety of Rust to machine learning and AI._  
_RustyML - 将Rust的强大性能和安全特性引入机器学习和人工智能领域。_