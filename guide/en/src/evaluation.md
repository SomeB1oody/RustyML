# Evaluate results

Enable the `metrics` feature for this chapter. Add `ndarray = "=0.17.2"` to your dependencies.
The examples also work with the default RustyML features.

Evaluate predictions on data excluded from fitting. Reuse preprocessing statistics learned from the training set.
Compare models on the same samples and target order.
Pass true values first and predicted values second to regression and hard-label metrics.
These metrics return scores directly. They do not return `Result`.

## Select a regression metric

Regression metrics accept one-dimensional `f64` arrays or views.

| Function | Interpretation | Preferred direction |
| --- | --- | --- |
| `mean_squared_error` | Mean squared residual | Lower |
| `root_mean_squared_error` | Square root of MSE, in target units | Lower |
| `mean_absolute_error` | Mean absolute residual, in target units | Lower |
| `median_absolute_error` | Median absolute residual | Lower |
| `r2_score` | Improvement over the true target mean baseline | Higher |
| `explained_variance_score` | Explained target variance | Higher |
| `mean_absolute_percentage_error` | Mean absolute relative error as a fraction | Lower |

MSE gives large residuals more weight. MAE is easier to interpret in target units.

A perfect R2 score is `1.0`. Negative R2 scores indicate worse predictions than the mean baseline.
For constant targets, `r2_score` returns `1.0` for exact predictions and `0.0` otherwise.

MAPE can become very large near zero targets. Its output is a fraction, so multiply by 100 for percentages.

## Select a classification metric

`accuracy` and binary `ConfusionMatrix` accept `f64` labels.
`ConfusionMatrix::new` uses `0.0` for negative labels and `1.0` for positive labels.
Use `new_with_labels` for another explicit negative and positive label pair.
`get_counts` returns `(true_positives, false_positives, true_negatives, false_negatives)`.

Precision measures the fraction of predicted positives that are correct.
Recall measures the fraction of actual positives that the model finds.
F1 combines precision and recall through their harmonic mean.

`MulticlassConfusionMatrix` accepts `usize` labels. Its matrix rows represent true classes, and columns represent predicted classes.
`labels()` gives the sorted labels corresponding to matrix indices.
`Average::Macro` weights each class equally. `Average::Weighted` weights each class by its true support.
`Average::Micro` combines counts across classes before computing the score.

`roc_auc` accepts boolean true labels and `f64` positive-class scores.
Pass probabilities or ranking scores rather than thresholded class predictions.
A perfect ROC AUC is `1.0`. Both positive and negative samples must exist.
Use `average_precision` or `precision_recall_curve` to inspect positive-class ranking.
`roc_curve` returns `(fpr, tpr, thresholds)` arrays.

`log_loss` accepts `usize` labels and a `(samples, classes)` probability matrix.
Each label indexes a probability column. Supply finite, nonnegative probabilities with positive row sums.
The function normalizes each row and clips the selected probability before taking its logarithm.
Lower log loss is better. `top_k_accuracy` checks whether the true class appears among the top `k` columns.

Run `metrics_scores.rs`:

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

## Evaluate clusters

External clustering metrics compare reference labels with predicted cluster labels.

They use one-dimensional `isize` arrays. Cluster numbers need not match reference class numbers.
`adjusted_rand_index` and `normalized_mutual_info` both return `1.0` for equivalent partitions.

ARI adjusts pair agreement for chance. NMI measures shared label information without chance adjustment.

`adjusted_mutual_info` adds chance adjustment to mutual information.
Homogeneity, completeness, V-measure, and Fowlkes-Mallows provide other external comparisons.

Internal metrics use features and cluster assignments without reference labels.
`silhouette_score` accepts an explicit `DistanceCalculationMetric`.

Larger silhouette values indicate better separation. Values near zero indicate overlapping clusters.
Negative values suggest that samples may fit another cluster better.

`davies_bouldin_score` prefers lower values. `calinski_harabasz_score` prefers higher values.
These internal metrics require between two and `samples - 1` distinct clusters.

Negative labels count as ordinary cluster identifiers. Filter noise samples yourself when that matches your evaluation policy.

Run `metrics_clusters.rs`:

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

## Validate before scoring

Metric input checks panic on invalid inputs. Handle validation before calling metrics in a data processing service.

Regression and paired label metrics require equal nonzero lengths.
Internal clustering metrics require one label for every feature row and a valid cluster count.

Binary confusion matrices reject labels outside the selected label pair.
ROC metrics reject `NaN` scores. ROC AUC requires both label values.

Log loss validates class indices. Validate probability values and row sums yourself.

Regression arithmetic can propagate non-finite values. Do not assume every metric rejects all non-finite inputs.

Keep prediction and target shapes equal. Preserve probability column meanings when encoding labels.
Choose metrics before comparing models to avoid selecting only favorable results.
