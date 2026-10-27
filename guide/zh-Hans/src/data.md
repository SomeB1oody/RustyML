# 准备数据

本章需要启用 `utils` 功能。在依赖中添加 `ndarray = "=0.17.2"`。
示例也支持 RustyML 的默认功能。

## 选择数组类型

传统模型的特征使用 `Array2<f64>`。每行代表一个样本，每列代表一个特征。
形状 `(100, 4)` 表示 100 个样本和四个特征。

回归目标使用 `Array1<f64>`。分类标签的类型取决于接收数据的 API。
目标顺序必须与特征行顺序一致。

| 数据 | 类型或形状 | 用途 |
| --- | --- | --- |
| 特征矩阵 | `Array2<f64>`，`(samples, features)` | 传统模型和数据划分工具 |
| 回归目标 | `Array1<f64>`，`(samples,)` | 回归模型和指标 |
| 划分标签 | `Array1<A>` | 任意可克隆的标签类型 |
| 独热标签 | `Array2<f64>`，`(samples, classes)` | 分类目标 |
| 稀疏编码标签 | `Array1<i32>` | `to_categorical` 输入及 `to_sparse_categorical` 输出 |

`array!` 可以直接创建小数组。`Array2::from_shape_vec` 按行顺序从一维数据创建矩阵。
形状必须与元素数量一致。向 RustyML 传递数据前，请处理形状错误。

使用 `.view()` 可以借用数组，无需复制元素。
许多预处理函数通过泛型数组引用同时支持拥有数据的数组和视图。

划分函数会取得数组所有权。只有需要保留原始数组时才克隆。

神经网络层具有各自的张量形状要求。重塑数据前，请检查对应层。

## 先划分，再拟合预处理器

`train_test_split(x, y, test_size, random_state)` 在 `Result` 中返回 `(x_train, x_test, y_train, y_test)`。
传入 `Some(0.33)` 可将约三分之一的数据留作测试。`None` 使用默认测试比例 `0.3`。

传入 `Some(42)` 可固定随机种子。`None` 使用已设置的全局种子流，否则使用系统熵源。
详见[随机种子](random-seeds.md)。
划分会对测试样本数取整，并保证两侧各有至少一个样本。

分类任务需要两侧都保留各个类别时，使用 `train_test_split_stratified`。
标签必须实现 `Clone + Eq + Hash`。整数和字符串标签满足这些要求。
浮点标签没有实现 `Eq`。

每个类别至少需要两个样本。函数分别划分每个类别。
取整可能导致总测试比例与请求比例不同。

仅使用训练特征拟合 `StandardScaler`。使用同一个已拟合缩放器转换训练特征和留出特征。
这个顺序可避免测试信息改变训练数据的转换方式。
保存模型时，也应保留缩放器，以便后续预测。

缩放器保存每个特征的总体均值和方差。
默认转换会减去均值，再除以标准差。
常量或近似常量特征使用 `1.0` 作为除数。

`fit` 会替换已有统计量。`fit_transform` 会拟合并转换同一份输入。
`partial_fit` 可以使用额外的训练批次更新统计量。

无状态的 `standardize` 函数每次都会根据当前输入重新计算统计量。
需要跨数据集保持一致转换时，请使用有状态缩放器。

运行 `data_prepare.rs`：

```rust
use ndarray::array;
use rustyml::utils::{StandardScaler, train_test_split, train_test_split_stratified};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0]
    ];
    let y = array![0, 0, 0, 1, 1, 1];
    let (x_train, x_test, y_train, y_test) =
        train_test_split(x.clone(), y.clone(), Some(0.33), Some(42))?;
    println!("train: {:?}, labels: {}", x_train.dim(), y_train.len());
    println!("test: {:?}, labels: {}", x_test.dim(), y_test.len());
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train)?;
    let z_train = scaler.transform(&x_train)?;
    let z_test = scaler.transform(&x_test)?;
    println!("scaled: {:?}, {:?}", z_train.dim(), z_test.dim());
    let (_, _, strat_train, strat_test) = train_test_split_stratified(x, y, Some(0.33), Some(42))?;
    println!(
        "class 0 counts: train {}, test {}",
        strat_train.iter().filter(|&&v| v == 0).count(),
        strat_test.iter().filter(|&&v| v == 0).count()
    );
    Ok(())
}
```

```text
train: (4, 2), labels: 4
test: (2, 2), labels: 2
scaled: (4, 2), (2, 2)
class 0 counts: train 2, test 1
```

## 保持标签编码一致

`to_categorical` 接收非负 `i32` 标签。标签值直接选择对应列。
传入固定的类别数，可保证各个子集具有相同列数。
省略类别数时，函数使用最大标签值加一。

`to_categorical_with_mapping` 接收可哈希标签的切片，并按首次出现顺序分配索引。
保存返回的映射，编码后续数据时使用相同索引。
单独对测试标签调用此函数，可能改变各列含义。
应用需要自行决定如何处理训练映射中不存在的标签。

`to_sparse_categorical` 选择每行的最大值，并将列索引作为 `i32` 标签返回。
并列最大值选择最小列索引。使用保存的映射恢复原始字符串标签。
此函数也支持所有值均有限的分数矩阵。

运行 `data_labels.rs`：

```rust
use ndarray::array;
use rustyml::utils::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = array![0_i32, 2, 1];
    let encoded = to_categorical(&labels, Some(3))?;
    println!("one-hot shape: {:?}", encoded.dim());
    println!("first row: {:?}", encoded.row(0).to_vec());
    println!("decoded: {:?}", to_sparse_categorical(&encoded)?.to_vec());
    let (encoded, mapping) = to_categorical_with_mapping(&["cat", "dog", "cat"], None)?;
    println!("cat: {}, dog: {}", mapping["cat"], mapping["dog"]);
    println!("mapped shape: {:?}", encoded.dim());
    Ok(())
}
```

```text
one-hot shape: (3, 3)
first row: [1.0, 0.0, 0.0]
decoded: [0, 2, 1]
cat: 0, dog: 1
mapped shape: (3, 2)
```

## 处理预处理错误

示例通过 `?` 返回错误。生产代码可以匹配 `rustyml::error::Error` 的各个变体。
这个枚举允许增加新变体，因此匹配时需要添加兜底分支。

| 操作 | 无效输入 | 结果 |
| --- | --- | --- |
| 两种划分 | 样本为空 | `Error::EmptyInput` |
| 两种划分 | 特征行数与标签数不同 | `Error::DimensionMismatch` |
| 两种划分 | 测试比例不在开区间 `(0, 1)` 内 | `Error::InvalidParameter` |
| 普通划分 | 样本少于两个 | 单样本返回 `Error::InvalidInput` |
| 分层划分 | 任一类别少于两个样本 | `Error::InvalidInput` |
| 缩放器拟合 | 空矩阵或非有限值 | `Error::EmptyInput` 或 `Error::NonFinite` |
| 缩放器转换 | 尚未拟合 | `Error::NotFitted` |
| 缩放器转换 | 特征数不同 | `Error::DimensionMismatch` |
| 整数标签编码 | 负标签 | `Error::InvalidInput` |
| 两种独热编码 | 类别数不足 | `Error::InvalidParameter` |
| 稀疏解码 | 矩阵含非有限值 | `Error::NonFinite` |

训练前请检查数据值。划分工具不会全面验证特征值。
