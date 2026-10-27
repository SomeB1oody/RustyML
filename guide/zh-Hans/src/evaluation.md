# 评估结果

本章需要启用 `metrics` 功能。在依赖中添加 `ndarray = "=0.17.2"`。
示例也支持 RustyML 的默认功能。

使用未参与拟合的数据评估预测。复用训练集上学到的预处理统计量。
比较模型时，使用相同样本和目标顺序。
回归指标和离散标签指标的参数顺序是先真实值，后预测值。
这些指标直接返回分数，不返回 `Result`。

## 选择回归指标

回归指标接收一维 `f64` 数组或视图。

| 函数 | 含义 | 更优方向 |
| --- | --- | --- |
| `mean_squared_error` | 残差平方的均值 | 越低越好 |
| `root_mean_squared_error` | MSE 的平方根，单位与目标一致 | 越低越好 |
| `mean_absolute_error` | 绝对残差的均值，单位与目标一致 | 越低越好 |
| `median_absolute_error` | 绝对残差的中位数 | 越低越好 |
| `r2_score` | 相对真实目标均值基线的改进程度 | 越高越好 |
| `explained_variance_score` | 解释的目标方差 | 越高越好 |
| `mean_absolute_percentage_error` | 绝对相对误差的均值，以比例表示 | 越低越好 |

MSE 对较大残差赋予更高权重。MAE 使用目标单位，更容易解释。

完美预测的 R2 为 `1.0`。负 R2 表示预测比均值基线更差。
目标为常量时，`r2_score` 对完全正确的预测返回 `1.0`，否则返回 `0.0`。

目标接近零时，MAPE 可能非常大。其输出为比例，乘以 100 可得到百分比。

## 选择分类指标

`accuracy` 和二分类 `ConfusionMatrix` 接收 `f64` 标签。
`ConfusionMatrix::new` 使用 `0.0` 作为负类，`1.0` 作为正类。
使用其他明确的负类和正类标签时，请调用 `new_with_labels`。
`get_counts` 返回 `(true_positives, false_positives, true_negatives, false_negatives)`。

精确率表示预测为正的样本中，真正为正的比例。
召回率表示实际正样本中，模型找到的比例。
F1 是精确率和召回率的调和平均数。

`MulticlassConfusionMatrix` 接收 `usize` 标签。矩阵行代表真实类别，列代表预测类别。
`labels()` 返回按升序排列的标签，对应矩阵索引。
`Average::Macro` 为每个类别赋予相同权重。`Average::Weighted` 按各类真实样本数加权。
`Average::Micro` 先合并各类计数，再计算指标。

`roc_auc` 接收布尔真实标签和 `f64` 正类分数。
请传入概率或排序分数，而不是经过阈值处理的类别预测。
完美 ROC AUC 为 `1.0`。数据必须同时包含正样本和负样本。
使用 `average_precision` 或 `precision_recall_curve` 可以检查正类排序表现。
`roc_curve` 返回 `(fpr, tpr, thresholds)` 数组。

`log_loss` 接收 `usize` 标签和形状为 `(samples, classes)` 的概率矩阵。
每个标签索引对应的概率列。请提供有限、非负的概率，且每行总和为正。
函数会归一化每行，并在取对数前裁剪所选概率。
对数损失越低越好。`top_k_accuracy` 检查真实类别是否位于分数最高的 `k` 列中。

运行 `metrics_scores.rs`：

```rust
use ndarray::array;
use rustyml::metrics::{
    Average, ConfusionMatrix, MulticlassConfusionMatrix, accuracy, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc, root_mean_squared_error,
};

fn main() {
    let actual = array![1.0, 2.0, 3.0];
    let predicted = array![1.0, 2.0, 4.0];
    println!("MSE: {:.3}", mean_squared_error(&actual, &predicted));
    println!("RMSE: {:.3}", root_mean_squared_error(&actual, &predicted));
    println!("MAE: {:.3}", mean_absolute_error(&actual, &predicted));
    println!("R2: {:.3}", r2_score(&actual, &predicted));
    let actual = array![0.0, 0.0, 1.0, 1.0];
    let predicted = array![0.0, 1.0, 1.0, 1.0];
    let matrix = ConfusionMatrix::new(&actual, &predicted);
    println!("accuracy: {:.3}", accuracy(&actual, &predicted));
    println!("TP, FP, TN, FN: {:?}", matrix.get_counts());
    println!(
        "precision: {:.3}, recall: {:.3}, F1: {:.3}",
        matrix.precision(),
        matrix.recall(),
        matrix.f1_score()
    );
    let labels = array![false, false, true, true];
    let scores = array![0.1, 0.4, 0.35, 0.8];
    println!("ROC AUC: {:.3}", roc_auc(&labels, &scores));
    let actual = array![0_usize, 1, 2, 2];
    let predicted = array![0_usize, 2, 2, 2];
    let matrix = MulticlassConfusionMatrix::new(&actual, &predicted);
    println!("macro F1: {:.3}", matrix.f1(Average::Macro));
}
```

```text
MSE: 0.333
RMSE: 0.577
MAE: 0.333
R2: 0.500
accuracy: 0.750
TP, FP, TN, FN: (2, 1, 1, 0)
precision: 0.667, recall: 1.000, F1: 0.800
ROC AUC: 0.750
macro F1: 0.600
```

## 评估聚类

外部聚类指标比较参考标签与预测簇标签。

它们使用一维 `isize` 数组。簇编号无需与参考类别编号相同。
对于等价划分，`adjusted_rand_index` 和 `normalized_mutual_info` 都返回 `1.0`。

ARI 对样本对的一致性进行随机机会校正。NMI 衡量共享标签信息，不进行机会校正。

`adjusted_mutual_info` 为互信息添加机会校正。
同质性、完整性、V-measure 和 Fowlkes-Mallows 提供其他外部比较方式。

内部指标使用特征和簇分配，不需要参考标签。
`silhouette_score` 接收显式的 `DistanceCalculationMetric`。

轮廓系数越大，通常表示分离程度越好。接近零表示簇存在重叠。
负值表示样本可能更适合另一个簇。

`davies_bouldin_score` 越低越好。`calinski_harabasz_score` 越高越好。
这些内部指标要求不同簇的数量介于二和 `samples - 1` 之间。

负标签会被当作普通簇标识符。如果评估策略需要排除噪声，请自行过滤相应样本。

运行 `metrics_clusters.rs`：

```rust
use ndarray::array;
use rustyml::math::distance::DistanceCalculationMetric;
use rustyml::metrics::{adjusted_rand_index, normalized_mutual_info, silhouette_score};

fn main() {
    let x = array![[0.0], [1.0], [10.0], [11.0]];
    let reference = array![0_isize, 0, 1, 1];
    let clusters = array![7_isize, 7, 3, 3];
    println!("ARI: {:.3}", adjusted_rand_index(&reference, &clusters));
    println!("NMI: {:.3}", normalized_mutual_info(&reference, &clusters));
    println!(
        "silhouette: {:.3}",
        silhouette_score(&x, &clusters, DistanceCalculationMetric::Euclidean)
    );
}
```

```text
ARI: 1.000
NMI: 1.000
silhouette: 0.900
```

## 评分前验证输入

指标遇到无效输入时会触发 panic。在数据处理服务中调用指标前，请先验证输入。

回归指标和成对标签指标要求输入长度相等且非零。
内部聚类指标要求每个特征行对应一个标签，且簇数量有效。
二分类混淆矩阵会拒绝指定标签对之外的标签。
ROC 指标会拒绝 `NaN` 分数。ROC AUC 要求同时存在两种布尔标签。

对数损失会验证类别索引。概率值和行总和需要自行验证。
回归运算可能传播非有限值。不要假设每个指标都会拒绝所有非有限输入。

保持预测和目标形状相等。编码标签时，保持概率列含义一致。
比较模型前先选定指标，避免只选择有利的结果。
