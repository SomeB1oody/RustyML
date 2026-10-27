# RustyML 使用指南

RustyML 为 Rust 提供机器学习模型、神经网络、数据工具和评估指标。
本书介绍 RustyML 0.17.0 中主要公开 API 的使用方法。
阅读本书需要基本的 Rust 和 Cargo 知识。

先阅读[安装](installation.md)，再运行[第一个模型](first-model.md)。
其余章节可以按需阅读。

| 任务 | 章节 |
| --- | --- |
| 划分数据、缩放特征、转换标签 | [准备数据](data.md) |
| 预测数值或类别标签 | [监督学习模型](supervised.md) |
| 聚类或减少特征数量 | [无监督学习模型](unsupervised.md) |
| 构建和训练神经网络 | [神经网络](neural-networks.md) |
| 选择层并确定张量形状 | [神经网络层](neural-network-layers.md) |
| 评估模型结果 | [评估模型](evaluation.md) |
| 保存和加载传统模型 | [模型文件](model-files.md) |
| 处理操作失败 | [错误处理](errors.md) |
| 重复随机操作 | [随机种子](random-seeds.md) |

示例使用小型数组说明 API 的用法。
示例中的分数不代表模型在实际数据集上的表现。
完整的类型和方法说明见 [API 文档](https://docs.rs/rustyml/0.17.0/rustyml/)。
