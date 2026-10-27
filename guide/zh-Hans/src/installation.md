# 安装

RustyML 0.17.0 要求 Rust 1.89.0 或更高版本。
示例使用 Rust 2024 edition 和 ndarray 0.17.2。

## 创建项目

1. 创建 Cargo 项目。

   ```sh
   cargo new rustyml-example
   ```

2. 进入项目目录。

   ```sh
   cd rustyml-example
   ```

3. 在 `Cargo.toml` 中添加以下依赖。

   ```toml
   [dependencies]
   rustyml = "=0.17.0"
   ndarray = "=0.17.2"
   ```

4. 用[第一个模型](first-model.md)中的程序替换 `src/main.rs`。

5. 运行程序。

   ```sh
   cargo run
   ```

RustyML 默认启用全部公开模块。
示例会直接导入 ndarray 的数组和宏，因此需要单独声明 ndarray 依赖。

## 选择 feature

需要按模块选择功能时，禁用默认 feature，并列出所需的 feature。
以下配置支持传统机器学习模型、数据工具和评估指标。

```toml
[dependencies]
rustyml = { version = "=0.17.0", default-features = false, features = ["machine_learning", "utils", "metrics"] }
ndarray = "=0.17.2"
```

| Feature | 公开 API |
| --- | --- |
| `machine_learning` | 回归、分类、聚类和降维 |
| `neural_network` | 模型、层、损失函数和优化器 |
| `utils` | 数据划分、scaler 和标签转换 |
| `metrics` | 回归、分类和聚类指标 |
| `math` | 距离函数和归约函数 |
| `full` | 以上五个模块 |
| `show_progress` | 支持该功能的训练操作的进度条 |

默认 feature 启用 `full`。
`machine_learning`、`neural_network`、`utils` 和 `metrics` 都会同时启用 `math`。
Cargo feature 会向构建添加 API。
`show_progress` 不会启用模型模块。

## 导入 API

示例使用模块路径标明 API 的来源。
需要缩短导入语句时，可以使用分类 prelude。

```rust
use ndarray::array;
use rustyml::prelude::machine_learning::*;

fn main() -> Result<(), rustyml::error::Error> {
    let x = array![[1.0], [2.0]];
    let y = array![3.0, 5.0];
    let mut model = LinearRegression::new(true);
    model.fit(&x, &y)?;
    println!("prediction: {:.1}", model.predict(&array![[3.0]])?[0]);
    Ok(())
}
```

输出：

```text
prediction: 7.0
```

其他分类 prelude 包括 `rustyml::prelude` 下的 `utils`、`metrics` 和 `neural_network`。
根 prelude 导出所有已启用分类的 API。
启用 `metrics` 时，根 prelude 中的 `Average` 指标枚举用于指定平均方式。
合并层请使用 `rustyml::prelude::neural_network::Average`。
