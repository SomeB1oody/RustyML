# 神经网络层

根据样本形状和任务所需的输出选择层。
所有张量使用 `f32`，并将批次轴放在首位。
神经网络 prelude 导出下列层。

## 常用形状规则

| 层类型 | 输入 | 输出 |
| --- | --- | --- |
| `Dense(units)` | `[batch, ..., features]`，至少二维 | `[batch, ..., units]` |
| `Conv1D` | `[batch, length, channels]` | `[batch, output_length, filters]` |
| `Conv2D` | `[batch, height, width, channels]` | `[batch, output_height, output_width, filters]` |
| `Conv3D` | `[batch, depth, height, width, channels]` | 轴顺序不变，空间尺寸与滤波器数量改变 |
| 局部池化 | 卷积使用的轴顺序 | 空间轴缩小，通道数不变 |
| `Flatten` | `[batch, ...]` | `[batch, product_of_remaining_axes]` |
| 循环层，默认 | `[batch, time, features]` | `[batch, units]` |
| 循环层，返回序列 | `[batch, time, features]` | `[batch, time, units]` |

这些行说明形状行为，并非构造函数语法。
图像和卷积张量将通道放在最后。
灰度图像仍需保留大小为一的通道维度。
单张图像也必须包含批次轴。

## 全连接层

使用 `Dense::new(units, activation)?` 创建全连接层。
简单的隐藏层可使用 `Activation::ReLU`。
最后一层的激活必须与损失匹配。
参见[输出和损失选择](neural-networks.md#选择输出和损失)。

Dense 只替换最后一轴，保留其他前置轴。
例如，四个单元将 `[8, 12, 6]` 转换为 `[8, 12, 4]`。
需要每张图像对应一个向量时，先添加 `Flatten::new()`。
不展平时，Dense 分别处理各个前置位置。

## 卷积与池化

`Conv2D::new(filters, kernel_size, strides, activation)?` 配置二维卷积。
默认填充为 `PaddingType::Valid`。
调用 `.with_padding(PaddingType::Same)`，可在步长为一时保持空间尺寸。
需要更大的有效卷积核时，使用 `.with_dilation_rate(...)`。

对每个空间轴，定义 `effective_kernel = dilation * (kernel - 1) + 1`。
有效填充的输出大小为 `floor((input - effective_kernel) / stride) + 1`。
使用有效填充时，有效卷积核必须能放入输入。
相同填充的输出大小为 `ceil(input / stride)`。
分别对各空间轴应用这些规则。

`MaxPooling2D::new((2, 2))` 保留各窗口的最大值。
`AveragePooling2D` 使用窗口平均值。
局部池化保留通道数。
默认步长等于池化窗口大小，默认使用有效填充。
通过 `with_strides(...)?` 和 `with_padding(...)` 覆盖默认设置。
全局池化将所有空间位置缩减为每通道一个值。

下面的图像示例按以下步骤转换 `[2, 8, 8, 1]`：

1. 有效卷积产生 `[2, 6, 6, 4]`。
2. 池化产生 `[2, 3, 3, 4]`。
3. 展平产生 `[2, 36]`。
4. 全连接层产生 `[2, 3]`。

## 循环层

使用 `SimpleRNN`、`LSTM` 或 `GRU` 处理有序序列。
构造函数接收单元数和激活，例如 `LSTM::new(4, Activation::Tanh)?`。
输入必须为 `[batch, time, features]`。
默认输出包含最后一个隐藏状态，形状为 `[batch, units]`。
调用 `.with_return_sequences(true)`，返回每个时间步的隐藏状态。

堆叠另一个循环层前，启用序列输出。
后续 Dense 层将分别转换每个时间步。
需要每个序列对应一个预测时，保留最后一个循环层的默认输出。
调用 `.with_go_backwards(true)`，按相反顺序处理时间步。

## 运行图像与序列组合

预测无需优化器或损失。
此示例使用零输入检查两种组合的形状。
源文件位于 `guide/examples/nn_shapes.rs`。

```rust
use ndarray::Array;
use rustyml::prelude::neural_network::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images = Array::<f32, _>::zeros((2, 8, 8, 1)).into_dyn();
    let cnn = SequentialBuilder::new()
        .add(Conv2D::new(4, (3, 3), (1, 1), Activation::ReLU)?.with_random_state(11))
        .add(MaxPooling2D::new((2, 2)))
        .add(Flatten::new())
        .add(Dense::new(3, Activation::Linear)?.with_random_state(12))
        .build(&Shape::known(images.shape()))?;
    println!("image predictions: {:?}", cnn.predict(&images)?.shape());

    let sequence = Array::<f32, _>::zeros((2, 5, 3)).into_dyn();
    let rnn = SequentialBuilder::new()
        .add(
            LSTM::new(4, Activation::Tanh)?
                .with_random_state(13)
                .with_return_sequences(true),
        )
        .add(Dense::new(2, Activation::Linear)?.with_random_state(14))
        .build(&Shape::known(sequence.shape()))?;
    println!(
        "sequence predictions: {:?}",
        rnn.predict(&sequence)?.shape()
    );
    Ok(())
}
```

预期输出：

```text
image predictions: [2, 3]
sequence predictions: [2, 5, 2]
```

## 修正形状错误

逐层比较输出与下一层的输入要求。
使用 `model.summary()` 检查完整组合。
训练前，根据实际输入张量形状构建模型。
卷积拒绝图像输入时，检查通道顺序。
循环层拒绝前一层输出时，检查 `with_return_sequences`。
拟合前，检查目标维度是否符合所选损失。
