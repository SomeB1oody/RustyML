# 第一个模型

这个程序训练一个线性回归模型，并用测试数据评估它。
数据满足 `y = 2 * x + 1`。
每行是一个样本，每个样本有一个特征。

将完整程序复制到 `src/main.rs`。
依赖配置见[安装](installation.md)。

```rust
use ndarray::array;
use rustyml::error::RustymlResult;
use rustyml::machine_learning::LinearRegression;
use rustyml::metrics::mean_absolute_error;
use rustyml::utils::train_test_split::train_test_split;

fn main() -> RustymlResult<()> {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.33), Some(42))?;

    let mut model = LinearRegression::new(true);
    model.fit(&x_train, &y_train)?;
    let predictions = model.predict(&x_test)?;
    println!("training rows: {}", x_train.nrows());
    println!("test rows: {}", x_test.nrows());
    println!(
        "test MAE: {:.6}",
        mean_absolute_error(&y_test, &predictions)
    );

    let query = array![[6.0]];
    println!("prediction for 6: {:.3}", model.predict(&query)?[0]);
    Ok(())
}
```

运行 `cargo run`。

```text
training rows: 4
test rows: 2
test MAE: 0.000000
prediction for 6: 13.000
```

## 理解程序

`train_test_split` 将特征和目标划分成一一对应的训练数组和测试数组。
显式种子让重复运行使用相同的划分。
该函数会取得原始数组的所有权。

`LinearRegression::new(true)` 创建带截距的模型。
`fit` 从训练数组中学习特征与目标的关系。
`predict` 为每个输入行返回一个预测值。

`mean_absolute_error` 比较预测值和测试目标。
测试数组中的样本不参与训练。
最后一次查询使用与训练数据相同的特征数量和顺序。

这组数据满足精确的线性关系。
实际数据集通常会产生非零误差。
特征缩放和标签转换见[准备数据](data.md)。
其他模型的选择方法见[监督学习模型](supervised.md)。
