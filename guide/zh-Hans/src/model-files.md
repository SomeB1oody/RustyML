# 传统模型文件

保存已拟合的传统模型，然后通过其具体模型类型恢复。

本章介绍 `LinearRegression`、`KNN`、`KMeans`、`DBSCAN` 和 `PCA` 的模型文件。

启用 `machine_learning` 功能。此示例还需要 `ndarray = "0.17.2"`。

## 保存和恢复

`save_to_path(&self, path: &str) -> Result<(), Error>` 写入模型状态。

`Model::load_from_path(path: &str) -> Result<Model, Error>` 返回恢复后的模型。

保存操作会替换已有文件。如果需要保留旧模型，请选择其他路径。

此示例检查未拟合模型，保存训练后的模型，再恢复预测。最后删除临时文件。

```rust
use ndarray::array;
use rustyml::error::Error;
use rustyml::prelude::machine_learning::LinearRegression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![1.0, 3.0, 5.0];
    let mut model = LinearRegression::new(true);
    assert!(matches!(model.predict(&x), Err(Error::NotFitted(_))));
    model.fit(&x, &y)?;

    let path = std::env::temp_dir().join(format!("rustyml-guide-{}.bin", std::process::id()));
    let filename = path.to_str().ok_or("temporary path is not UTF-8")?;
    model.save_to_path(filename)?;
    let restored = LinearRegression::load_from_path(filename)?;
    let predictions = restored.predict(&array![[3.0]])?;
    println!("restored prediction: {:.1}", predictions[0]);
    std::fs::remove_file(path)?;
    Ok(())
}
```

输出：

```text
restored prediction: 7.0
```

## 保持输入兼容

恢复后的模型仍然要求原来的特征数量和顺序。单独保存标签含义和预处理设置。

保存未拟合模型不会训练它。预测仍然需要先训练。

使用创建文件时的具体模型类型加载。对于 KNN，还需要保持标签类型一致，例如 `KNN<i32>`。

KNN 持久化要求标签实现 serde 序列化和反序列化。整数标签满足这些要求。

为每个已保存模型记录 crate 版本。加载其他版本生成的文件前，请检查兼容性。

## 处理文件错误

文件访问和序列化失败返回 `Error::Io`。传播错误，或在报告错误时附上文件路径。

加载成功会恢复已存储的状态，但不会检查应用中的特征名称或预处理选择。

保留训练数据和配置，以便重新训练模型。
