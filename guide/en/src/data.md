# Prepare data

Enable the `utils` feature for this chapter. Add `ndarray = "=0.17.2"` to your dependencies.
The examples also work with the default RustyML features.

## Choose array types

Use `Array2<f64>` for classical model features. Each row contains one sample. Each column contains one feature.
A matrix with shape `(100, 4)` contains 100 samples and four features.

Use `Array1<f64>` for regression targets. Class label types depend on the receiving API.
Keep target order aligned with feature rows.

| Value | Type or shape | Use |
| --- | --- | --- |
| Feature matrix | `Array2<f64>`, `(samples, features)` | Classical models and split utilities |
| Regression targets | `Array1<f64>`, `(samples,)` | Regression models and metrics |
| Split labels | `Array1<A>` | Any cloneable label type |
| One-hot labels | `Array2<f64>`, `(samples, classes)` | Categorical targets |
| Sparse encoding labels | `Array1<i32>` | `to_categorical` input and `to_sparse_categorical` output |

`array!` creates small arrays directly. `Array2::from_shape_vec` builds matrices from flat values in row order.
Its shape must match the number of values. Handle its shape error before passing data to RustyML.

Use `.view()` to borrow an array without copying its elements.
Many preprocessing functions accept owned arrays and views through generic array references.

Split functions take ownership of their arrays. Clone them only when you need the originals later.

Neural network layers use their own tensor shapes. Check each layer before reshaping data.

## Split before fitting preprocessing

`train_test_split(x, y, test_size, random_state)` returns `(x_train, x_test, y_train, y_test)` inside a `Result`.
Pass `Some(0.33)` to reserve about one third for testing. `None` selects the default test fraction, `0.3`.

Pass `Some(42)` for a fixed random seed. `None` uses the global seed stream when set, or system entropy.
See [Random seeds](random-seeds.md).
The split rounds the test count and keeps at least one sample on each side.

Use `train_test_split_stratified` for classification when both subsets need each class.
It requires labels that implement `Clone + Eq + Hash`. Integer and string labels support these bounds.
Floating-point labels do not implement `Eq`.

Each class must contain at least two samples. The function splits each class separately.
Rounding can make its total test fraction differ from the requested fraction.

Fit `StandardScaler` on training features only. Transform training and held-out features with the same fitted scaler.
This sequence prevents test information from changing the training transformation.
Keep the scaler with the model for later predictions.

The scaler stores each feature's population mean and variance.
It subtracts the mean and divides by the standard deviation by default.
Constant or near-constant features use a divisor of `1.0`.

`fit` replaces stored statistics. `fit_transform` fits and transforms the same input.
`partial_fit` updates statistics from additional training batches.

The stateless `standardize` function computes fresh statistics from every input.
Use the stateful scaler when you need consistent transformations across datasets.

Run `data_prepare.rs`:

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

## Encode labels consistently

`to_categorical` accepts nonnegative `i32` labels. Each label selects its corresponding column.
Pass a fixed class count to preserve the same width across subsets.
Without a class count, the function uses the largest label plus one.

`to_categorical_with_mapping` accepts a slice of hashable labels. It assigns indices in first-seen order.
Save the returned mapping. Use those indices when encoding later data.
Calling it independently on test labels can change column meanings.
Decide how your application handles labels absent from the training mapping.

`to_sparse_categorical` selects each row's largest value. It returns column indices as `i32` labels.
Ties select the lowest column index. Use your saved mapping to recover original string labels.
This function also accepts finite score matrices.

Run `data_labels.rs`:

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

## Handle preprocessing errors

The examples return errors with `?`. Production code can match `rustyml::error::Error` variants.
Include a fallback arm because this enum is non-exhaustive.

| Operation | Invalid input | Result |
| --- | --- | --- |
| Either split | Empty samples | `Error::EmptyInput` |
| Either split | Different feature and label counts | `Error::DimensionMismatch` |
| Either split | Test fraction at or outside `(0, 1)` | `Error::InvalidParameter` |
| Plain split | Fewer than two samples | `Error::InvalidInput` for one sample |
| Stratified split | Any class has fewer than two samples | `Error::InvalidInput` |
| Scaler fit | Empty matrix or non-finite values | `Error::EmptyInput` or `Error::NonFinite` |
| Scaler transform | Scaler has no fitted state | `Error::NotFitted` |
| Scaler transform | Different feature count | `Error::DimensionMismatch` |
| Integer encoding | Negative label | `Error::InvalidInput` |
| Either one-hot encoder | Class count is too small | `Error::InvalidParameter` |
| Sparse decoding | Non-finite matrix value | `Error::NonFinite` |

Check dataset values before training. Split utilities do not perform full feature validation.
