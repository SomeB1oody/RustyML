# Unsupervised learning

Use clustering to group samples. Use PCA to reduce the number of features.

Enable the `machine_learning` feature. These examples also require `ndarray = "0.17.2"`.

## Prepare inputs

Pass an `Array2<f64>` with shape `(samples, features)`. Training does not require targets.

Scale features when their units or ranges differ. Apply the same fitted preprocessing to new samples.

## Cluster and reduce features

`KMeans::new(n_clusters, max_iterations, tolerance)` configures clustering. Set a random seed with `with_random_state(seed)` for repeatable initialization.

KMeans cluster numbers identify groups. They do not indicate rank or meaning.

`DBSCAN::new(eps, min_samples)` configures neighborhood radius and minimum density. DBSCAN uses `-1` for noise.

DBSCAN prediction uses the nearest fitted core point within `eps`. Otherwise, prediction returns `-1`.

`PCA::new(n_components)` selects the output width. Fit PCA before transforming new samples.

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

Output:

```text
same first pair: true
different groups: true
KMeans predictions: 2
DBSCAN labels: [0, 0, 1, 1]
DBSCAN noise: -1
PCA shape: (4, 1)
PCA query shape: (2, 1)
```

## Method contracts

The signatures below use owned arrays for clarity. These methods also accept compatible ndarray views.

| Models | Call | Successful result |
| --- | --- | --- |
| KMeans, DBSCAN, PCA | `fit(&mut self, x: &Array2<f64>)` | `&mut Self` |
| KMeans, DBSCAN | `fit_predict(&mut self, x: &Array2<f64>)` | `Array1<isize>` |
| KMeans, DBSCAN | `predict(&self, x: &Array2<f64>)` | `Array1<isize>` |
| PCA | `fit_transform(&mut self, x: &Array2<f64>)` | `Array2<f64>` |
| PCA | `transform(&self, x: &Array2<f64>)` | `Array2<f64>` |

Each call returns `Result<_, Error>`. Cluster output contains one label per input row.

PCA output has shape `(input_samples, n_components)`. PCA requires at least two training samples.

Choose `n_components` from one through `min(samples, features)`. Choose no more KMeans clusters than training samples.

## Handle errors

Use nonempty, finite inputs and valid constructor parameters. New inputs must have the fitted feature count.

Prediction and transformation require a fitted model. Otherwise, they return `Error::NotFitted`.

Training can report invalid inputs, numerical failures, or convergence failures. Propagate errors with `?` and inspect their messages.

See [Model files](model-files.md) to save the fitted clustering model or PCA projection.
