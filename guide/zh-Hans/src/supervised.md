# 监督学习

使用回归预测数值。使用分类预测标签。

启用 `machine_learning` 功能。这些示例还需要 `ndarray = "0.17.2"`。

## 准备输入

使用形状为 `(samples, features)` 的 `Array2<f64>` 存储特征。每行表示一个样本。

使用 `Array1` 存储目标，每个样本对应一个目标。训练和预测时必须保持特征顺序一致。

使用独立的训练集和测试集衡量性能。只用训练样本拟合预处理步骤。

## 选择模型

使用线性回归预测数值目标。从默认的 `LeastSquaresSolver::Normal` 求解器开始。

使用逻辑回归处理二分类目标，并获取正类概率。使用 KNN 处理多类别标签和基于距离的分类决策。

使用留出的样本比较模型。当特征范围不同时，在 KNN 或基于梯度的训练前缩放特征。

## 训练和预测

`LinearRegression::new(true)` 启用截距。其目标和预测值使用 `f64` 类型。

`KNN::<i32>::new(k)` 创建整数标签分类器。`k` 必须介于一和训练样本数量之间。

此示例使用一个邻居，并避免距离相同的情况。

```rust
use ndarray::array;
use rustyml::prelude::machine_learning::{KNN, LinearRegression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let y = array![1.0, 3.0, 5.0, 7.0];
    let mut regression = LinearRegression::new(true);
    regression.fit(&x, &y)?;
    let predictions = regression.predict(&array![[4.0], [5.0]])?;
    println!("regression: {:.1}, {:.1}", predictions[0], predictions[1]);

    let labels = array![0_i32, 0, 1, 1];
    let mut classifier = KNN::<i32>::new(1)?;
    classifier.fit(&x, &labels)?;
    let predictions = classifier.predict(&array![[0.1], [2.9]])?;
    println!("classes: {:?}", predictions.to_vec());
    Ok(())
}
```

输出：

```text
regression: 9.0, 11.0
classes: [0, 1]
```

## 配置逻辑回归和惩罚项

逻辑回归要求目标使用 `Array1<f64>`，且仅包含 `0.0` 和 `1.0`。

`LogisticRegression::new(fit_intercept, learning_rate, max_iterations, tolerance)` 配置训练。学习率和容差必须为有限正数。

`max_iterations` 必须大于零。如果训练需要更多迭代，可增加此值。如果训练不稳定，可减小学习率。

`predict_proba(&x)` 为每行返回一个正类概率。当概率至少为 `0.5` 时，`predict(&x)` 返回 `1.0`。

两个方法均返回 `Result<Array1<f64>, Error>`。逻辑回归的 `fit(&x, &y)` 输入形状与线性回归相同。

通过 `with_regularization(RegularizationType::L2(alpha))` 为任一回归模型添加 L2 正则化。惩罚强度必须有限且非负。

线性回归的默认求解器支持 L2。使用 L1 时，选择 `LeastSquaresSolver::GradientDescent`，并设置 `learning_rate`、`max_iter` 和 `tol`。

此示例配置逻辑回归的 L2 正则化和线性回归的 L1 正则化。

```rust
use ndarray::array;
use rustyml::prelude::machine_learning::{
    LeastSquaresSolver, LinearRegression, LogisticRegression, RegularizationType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[-2.0], [-1.0], [1.0], [2.0]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let mut classifier = LogisticRegression::new(true, 0.1, 10_000, 1e-8)?
        .with_regularization(RegularizationType::L2(0.1))?;
    classifier.fit(&x, &y)?;
    let queries = array![[-1.5], [1.5]];
    let probabilities = classifier.predict_proba(&queries)?;
    println!(
        "positive probabilities: {:.3}, {:.3}",
        probabilities[0], probabilities[1]
    );
    println!(
        "binary labels: {:?}",
        classifier.predict(&queries)?.to_vec()
    );

    let targets = array![-3.0, -1.0, 3.0, 5.0];
    let mut regression = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent {
            learning_rate: 0.05,
            max_iter: 10_000,
            tol: 1e-8,
        })?
        .with_regularization(RegularizationType::L1(0.01))?;
    regression.fit(&x, &targets)?;
    println!("L1 predictions: {}", regression.predict(&queries)?.len());
    Ok(())
}
```

输出：

```text
positive probabilities: 0.101, 0.899
binary labels: [0.0, 1.0]
L1 predictions: 2
```

## 方法约定

下表使用拥有数据的数组简化签名。这些方法也接受兼容的 ndarray 视图。

| 模型 | 调用 | 成功结果 |
| --- | --- | --- |
| 线性回归 | `fit(&mut self, x: &Array2<f64>, y: &Array1<f64>)` | `&mut Self` |
| 线性回归 | `predict(&self, x: &Array2<f64>)` | `Array1<f64>` |
| KNN 分类 | `fit(&mut self, x: &Array2<f64>, y: &Array1<T>)` | `&mut Self` |
| KNN 分类 | `predict(&self, x: &Array2<f64>)` | `Array1<T>` |

每次调用返回 `Result<_, Error>`。每行输入对应一个预测值。

KNN 标签必须实现 `Clone`、`Hash` 和 `Eq`。整数标签满足这些要求。

## 处理错误

先调用 `fit`，再调用 `predict`。未拟合模型的预测返回 `Error::NotFitted`。

使用有限特征值，并确保目标数量匹配。预测输入的特征数量必须与训练时相同。

处理 `EmptyInput`、`DimensionMismatch`、`NonFinite` 和无效参数错误。训练中的数值计算失败也可能返回错误。

像示例一样，使用 `?` 传播错误。参阅[模型文件](model-files.md)，保存已拟合模型。
