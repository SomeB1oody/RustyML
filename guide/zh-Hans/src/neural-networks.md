# 神经网络

当每层接收上一层的输出时，使用 `SequentialBuilder`。
通过 `use rustyml::prelude::neural_network::*` 导入神经网络常用类型。
启用 Cargo 的 `neural_network` 功能。
输入数组使用 `ndarray = "0.17.2"`。

## 准备张量

`Tensor` 是 `ndarray::ArrayD<f32>` 的别名。
使用 `.into_dyn()` 转换数组，并使用 `f32` 数值。
第零轴表示样本。
表格输入使用 `[samples, features]`。
回归目标使用 `[samples, outputs]`。
训练输入与目标必须具有相同的样本数。

训练和预测必须采用一致的特征缩放方式。
将评估样本与训练样本分开。

## 构建与训练

1. 创建各层，按顺序添加。
2. 调用 `build(&Shape::known(x.shape()))`，验证形状并初始化权重。
3. 训练或评估前，调用 `compile(optimizer, loss)`。
4. 调用 `fit`、`fit_with_batches` 或 `train_batch`。
5. 调用 `predict` 生成输出张量。

`fit(&x, &y, epochs)` 每轮使用完整数据集更新一次。
`fit_with_batches(&x, &y, epochs, batch_size)` 每轮打乱样本，然后逐批更新。
批次大小必须大于零，且不能超过样本数。
`train_batch(&x, &y)` 更新一次，不拆分数据，也不打乱顺序。

两种拟合方法均返回 `History`。
`history.loss()` 按轮次记录训练过程中测得的损失。
`evaluate(&x, &y)` 测量当前模型损失，不更新权重。
`predict(&x)` 和 `evaluate` 对 dropout 与归一化层使用推理行为。
`predict` 无需调用 `compile`。
构建后可以改变批次大小，但其他必要维度必须保持兼容。

## 选择输出和损失

| 任务 | 最后一层激活 | 损失 | 目标形状 |
| --- | --- | --- | --- |
| 回归 | `Activation::Linear` | `MeanSquaredError::new()` 或 `MeanAbsoluteError::new()` | 与预测相同 |
| 二分类 | `Activation::Sigmoid` | `BinaryCrossEntropy::new()` | `[samples, 1]`，值为 0 或 1 |
| 类别概率 | `Activation::Softmax { axis: -1 }` | `CategoricalCrossEntropy::new(false)` | `[samples, classes]`，独热值 |
| 类别 logits | `Activation::Linear` | `SparseCategoricalCrossEntropy::new(true)` | `[samples, 1]`，类别索引 |

稀疏标签仍使用 `f32`，例如 `0.0` 和 `2.0`。
类别索引必须为零至 `classes - 1` 范围内的整数。
对于类别交叉熵，`true` 表示输入原始 logits，`false` 表示输入概率。
配置 logits 损失时，不要先应用 softmax。

可从 `SGD` 开始，设置学习率和可选的动量。
`SGD::new(0.05, 0.0, false, 0.0)` 关闭动量、Nesterov 更新与权重衰减。
`Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0)` 提供自适应更新。
`AdamW` 提供解耦权重衰减。
`RMSprop` 和 `AdaGrad` 提供其他自适应更新方式。
使用 `set_learning_rate` 修改已编译优化器的学习率。

## 训练、保存与恢复

此示例通过四个样本学习 `y = 2x + 1`。
它检查训练进展，并验证加载后的预测完全相同。
源文件位于 `guide/examples/nn_regression.rs`。

```rust
use ndarray::array;
use rustyml::prelude::neural_network::*;

fn build_model() -> Result<Sequential, rustyml::error::Error> {
    SequentialBuilder::new_with_seed(42)
        .add(Dense::new(1, Activation::Linear)?.with_random_state(7))
        .build(&Shape::known(&[4, 1]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[-1.0_f32], [0.0], [1.0], [2.0]].into_dyn();
    let y = array![[-1.0_f32], [1.0], [3.0], [5.0]].into_dyn();
    let mut model = build_model()?;
    model.compile(SGD::new(0.05, 0.0, false, 0.0)?, MeanSquaredError::new());
    let initial_loss = model.evaluate(&x, &y)?;
    let history = model.fit_with_batches(&x, &y, 200, 2)?;
    let final_loss = model.evaluate(&x, &y)?;
    assert!(final_loss < initial_loss);
    assert!(final_loss < 0.0001);
    println!("epochs: {}", history.loss().len());
    println!("loss decreased: {}", final_loss < initial_loss);

    let path = std::env::temp_dir().join(format!("rustyml-nn-{}.bin", std::process::id()));
    model.save_to_path(&path)?;
    let mut restored = build_model()?;
    restored.load_from_path(&path)?;
    let prediction = restored.predict(&x)?;
    assert_eq!(model.predict(&x)?, prediction);
    println!("prediction shape: {:?}", prediction.shape());
    println!("restored predictions match: true");
    std::fs::remove_file(path)?;
    Ok(())
}
```

预期输出：

```text
epochs: 200
loss decreased: true
prediction shape: [4, 1]
restored predictions match: true
```

`save_to_path` 保存可训练数组和不可训练数组，包括归一化统计量。
它不保存架构、层配置、优化器或损失。
调用 `load_from_path` 前，重新构建相同架构。
严格加载器检查层类型、数组名称、数组形状及构建形状的兼容性。
检查点允许使用不同的批次大小。
请保持层配置相同，因为检查点不会恢复配置。

加载后，先调用 `compile`，再训练或评估。
训练使用新的优化器状态，因此加载不会恢复优化器历史。
如示例所示，预测无需编译。
仅在有意加载匹配数组时使用 `load_partial_from_path`，并检查其报告。

## 重复训练

构建前，通过 `with_random_state` 设置各随机层的种子。
通过 `SequentialBuilder::new_with_seed` 或 `model.set_seed` 设置批次打乱种子。
构建器种子只控制打乱顺序，不控制层初始化。
整批 `fit` 不打乱样本。

也可以在创建随机组件前调用 `rustyml::random::set_global_seed`。
在同一线程重复相同操作序列前，重置该种子。
没有局部种子的随机操作共享该线程的随机流。
保持输入、预处理、操作顺序和训练设置一致。
不同硬件和执行环境可能产生不同的浮点结果。

继续阅读[层形状与常用组合](neural-network-layers.md)。
