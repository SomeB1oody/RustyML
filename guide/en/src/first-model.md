# Your first model

This program trains a linear regression model and evaluates it on test data.
The data follows `y = 2 * x + 1`.
Each row contains one sample with one feature.

Copy the complete program into `src/main.rs`.
Use the dependencies from [Installation](installation.md).

```rust
use ndarray::array;
use rustyml::error::RustymlResult;
use rustyml::machine_learning::LinearRegression;
use rustyml::metrics::mean_absolute_error;
use rustyml::utils::train_test_split::train_test_split;

fn main() -> RustymlResult<()> {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.33), Some(42))?;

    let mut model = LinearRegression::new(true);
    model.fit(&x_train, &y_train)?;
    let predictions = model.predict(&x_test)?;
    println!("training rows: {}", x_train.nrows());
    println!("test rows: {}", x_test.nrows());
    println!(
        "test MAE: {:.6}",
        mean_absolute_error(&y_test, &predictions)
    );

    let query = array![[6.0]];
    println!("prediction for 6: {:.3}", model.predict(&query)?[0]);
    Ok(())
}
```

Run `cargo run`.

```text
training rows: 4
test rows: 2
test MAE: 0.000000
prediction for 6: 13.000
```

## Read the program

`train_test_split` divides features and targets into matching training and test arrays.
The explicit seed fixes the split for repeat runs.
The function consumes the original arrays.

`LinearRegression::new(true)` creates a model with an intercept.
`fit` learns the relation from the training arrays.
`predict` returns one value for each input row.

`mean_absolute_error` compares the predictions with the test targets.
The test arrays contain samples that the model did not use during training.
The final query uses the same feature count and feature order as the training data.

This dataset has an exact linear relation.
Real datasets usually produce a nonzero error.
Use [Prepare data](data.md) for feature scaling and label conversion.
Use [Supervised models](supervised.md) to select another model.
