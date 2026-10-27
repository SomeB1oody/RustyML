# 错误处理

大多数模型和数据工具方法返回 `Result<T, rustyml::error::Error>`。
使用 `?` 将错误返回给调用者。
调用者能够处理特定问题时，使用 `match`。

## 处理模型状态错误

这个程序处理训练前预测产生的错误。
随后，它训练模型并再次预测。

```rust
use ndarray::array;
use rustyml::error::{Error, RustymlResult};
use rustyml::machine_learning::LinearRegression;

fn main() -> RustymlResult<()> {
    let x = array![[1.0], [2.0]];
    let mut model = LinearRegression::new(true);
    match model.predict(&x) {
        Err(Error::NotFitted(_)) => println!("fit the model before prediction"),
        Err(error) => return Err(error),
        Ok(_) => println!("prediction succeeded"),
    }
    model.fit(&x, &array![3.0, 5.0])?;
    println!("prediction: {:.1}", model.predict(&array![[3.0]])?[0]);
    Ok(())
}
```

输出：

```text
fit the model before prediction
prediction: 7.0
```

`Error` 标有 `non_exhaustive`。
匹配其变体时，需要包含兜底分支。

## 处理常见错误

| 错误 | 处理方法 |
| --- | --- |
| `EmptyInput` | 提供非空数组。 |
| `DimensionMismatch` | 检查样本数量和特征数量。 |
| `ShapeMismatch` | 检查完整的张量形状。 |
| `InvalidParameter` | 检查构造函数或方法的参数。 |
| `InvalidInput` | 检查数组维数、标签值和其他输入要求。 |
| `NonFinite` | 检查 NaN、无穷值或数值溢出。 |
| `NotFitted` | 在预测或转换之前调用 `fit`。 |
| `NotConverged` | 检查数据缩放和求解器设置。 |
| `NeuralNetwork(NnError::NotCompiled(_))` | 在训练或评估前调用 `compile`。 |
| `Io` | 检查文件路径、权限和文件格式。 |

`NnError` 位于 `rustyml::neural_network`。
`neural_network` feature 启用 `NeuralNetwork` 错误变体。
每个方法的 API 文档列出该方法可能返回的错误。

## 指标的输入检查

指标直接返回分数。
无效的指标输入可能导致 panic。
调用指标前，检查数组长度、标签值和该指标的输入要求。
常用指标的输入要求见[评估模型](evaluation.md)。
