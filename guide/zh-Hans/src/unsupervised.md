# 无监督学习

使用聚类对样本分组。使用 PCA 减少特征数量。

启用 `machine_learning` 功能。这些示例还需要 `ndarray = "0.17.2"`。

## 准备输入

传入形状为 `(samples, features)` 的 `Array2<f64>`。训练不需要目标值。

当特征单位或范围不同时，应缩放特征。对新样本应用相同的已拟合预处理步骤。

## 聚类和降维

`KMeans::new(n_clusters, max_iterations, tolerance)` 配置聚类。通过 `with_random_state(seed)` 设置随机种子，使初始化可重复。

KMeans 簇编号用于区分分组，不表示排序或含义。

`DBSCAN::new(eps, min_samples)` 配置邻域半径和最小密度。DBSCAN 使用 `-1` 表示噪声。

DBSCAN 预测使用 `eps` 范围内最近的已拟合核心点。否则，预测返回 `-1`。

`PCA::new(n_components)` 设置输出列数。先拟合 PCA，再转换新样本。

```rust
use ndarray::array;
use rustyml::prelude::machine_learning::{DBSCAN, KMeans, PCA};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];
    let mut kmeans = KMeans::new(2, 100, 1e-6)?.with_random_state(42);
    let labels = kmeans.fit_predict(&x)?;
    println!("same first pair: {}", labels[0] == labels[1]);
    println!("different groups: {}", labels[0] != labels[2]);
    let queries = array![[0.05, 0.05], [5.05, 5.05]];
    println!("KMeans predictions: {}", kmeans.predict(&queries)?.len());

    let mut dbscan = DBSCAN::new(0.3, 2)?;
    let labels = dbscan.fit_predict(&x)?;
    println!("DBSCAN labels: {:?}", labels.to_vec());
    let noise = dbscan.predict(&array![[20.0, 20.0]])?;
    println!("DBSCAN noise: {}", noise[0]);

    let mut pca = PCA::new(1)?;
    let reduced = pca.fit_transform(&x)?;
    println!("PCA shape: {:?}", reduced.dim());
    println!("PCA query shape: {:?}", pca.transform(&queries)?.dim());
    Ok(())
}
```

输出：

```text
same first pair: true
different groups: true
KMeans predictions: 2
DBSCAN labels: [0, 0, 1, 1]
DBSCAN noise: -1
PCA shape: (4, 1)
PCA query shape: (2, 1)
```

## 方法约定

下表使用拥有数据的数组简化签名。这些方法也接受兼容的 ndarray 视图。

| 模型 | 调用 | 成功结果 |
| --- | --- | --- |
| KMeans、DBSCAN、PCA | `fit(&mut self, x: &Array2<f64>)` | `&mut Self` |
| KMeans、DBSCAN | `fit_predict(&mut self, x: &Array2<f64>)` | `Array1<isize>` |
| KMeans、DBSCAN | `predict(&self, x: &Array2<f64>)` | `Array1<isize>` |
| PCA | `fit_transform(&mut self, x: &Array2<f64>)` | `Array2<f64>` |
| PCA | `transform(&self, x: &Array2<f64>)` | `Array2<f64>` |

每次调用返回 `Result<_, Error>`。聚类输出为每行输入提供一个标签。

PCA 输出形状为 `(input_samples, n_components)`。PCA 至少需要两个训练样本。

`n_components` 必须介于一和 `min(samples, features)` 之间。KMeans 簇数量不能超过训练样本数量。

## 处理错误

使用非空且数值有限的输入，并设置有效的构造参数。新输入的特征数量必须与拟合时相同。

预测和转换需要已拟合的模型。否则，它们返回 `Error::NotFitted`。

训练可能报告无效输入、数值计算失败或收敛失败。使用 `?` 传播错误，并检查错误消息。

参阅[模型文件](model-files.md)，保存已拟合的聚类模型或 PCA 投影。
