# Supervised learning

Use regression to predict numeric values. Use classification to predict labels.

Enable the `machine_learning` feature. These examples also require `ndarray = "0.17.2"`.

## Prepare inputs

Store features in an `Array2<f64>` with shape `(samples, features)`. Each row contains one sample.

Store targets in an `Array1` with one target per sample. Keep feature order consistent during training and prediction.

Use separate training and test samples to measure performance. Fit preprocessing only on training samples.

## Choose a model

Use linear regression for numeric targets. Start with the default `LeastSquaresSolver::Normal` solver.

Use logistic regression for binary targets and positive-class probabilities. Use KNN for labels with multiple classes and distance-based decisions.

Compare models on held-out samples. Scale features before KNN or gradient-based training when their ranges differ.

## Train and predict

`LinearRegression::new(true)` enables an intercept. Its targets and predictions contain `f64` values.

`KNN::<i32>::new(k)` creates a classifier with integer labels. Choose `k` between one and the training sample count.

The example uses one neighbor and avoids distance ties.

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

Output:

```text
regression: 9.0, 11.0
classes: [0, 1]
```

## Configure logistic regression and penalties

Logistic regression requires `Array1<f64>` targets containing only `0.0` and `1.0`.

`LogisticRegression::new(fit_intercept, learning_rate, max_iterations, tolerance)` configures training. Use positive, finite learning rates and tolerances.

Set `max_iterations` above zero. Increase it when training needs more iterations, and reduce the learning rate if training becomes unstable.

`predict_proba(&x)` returns one positive-class probability per row. `predict(&x)` returns `1.0` when that probability is at least `0.5`.

Both methods return `Result<Array1<f64>, Error>`. Logistic regression uses the same `fit(&x, &y)` input shapes as linear regression.

Add `with_regularization(RegularizationType::L2(alpha))` to either regression model. Use finite, nonnegative penalty strengths.

Linear regression's default solver supports L2. Select `LeastSquaresSolver::GradientDescent` for L1, with `learning_rate`, `max_iter`, and `tol` settings.

The example configures logistic L2 regularization and linear L1 regularization.

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

Output:

```text
positive probabilities: 0.101, 0.899
binary labels: [0.0, 1.0]
L1 predictions: 2
```

## Method contracts

The signatures below use owned arrays for clarity. These methods also accept compatible ndarray views.

| Model | Call | Successful result |
| --- | --- | --- |
| Linear regression | `fit(&mut self, x: &Array2<f64>, y: &Array1<f64>)` | `&mut Self` |
| Linear regression | `predict(&self, x: &Array2<f64>)` | `Array1<f64>` |
| KNN classification | `fit(&mut self, x: &Array2<f64>, y: &Array1<T>)` | `&mut Self` |
| KNN classification | `predict(&self, x: &Array2<f64>)` | `Array1<T>` |

Each call returns `Result<_, Error>`. Predictions contain one value per input row.

KNN labels must implement `Clone`, `Hash`, and `Eq`. Integer labels satisfy these requirements.

## Handle errors

Call `fit` before `predict`. Unfitted predictions return `Error::NotFitted`.

Use finite feature values and matching target counts. Prediction inputs must have the training feature count.

Handle `EmptyInput`, `DimensionMismatch`, `NonFinite`, and invalid parameter errors. Numerical training failures can also return errors.

Propagate failures with `?`, as the example does. See [Model files](model-files.md) to save fitted models.
