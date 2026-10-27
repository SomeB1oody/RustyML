# Errors

Most model and data utility methods return `Result<T, rustyml::error::Error>`.
Use `?` to return an error to the caller.
Use `match` when the caller can correct a specific problem.

## Handle a model state error

This program handles prediction before training.
It then trains the model and repeats the prediction.

```rust
use ndarray::array;
use rustyml::error::{Error, RustymlResult};
use rustyml::machine_learning::LinearRegression;

fn main() -> RustymlResult<()> {
    let x = array![[1.0], [2.0]];
    let mut model = LinearRegression::new(true);
    match model.predict(&x) {
        Err(Error::NotFitted(_)) => println!("fit the model before prediction"),
        Err(error) => return Err(error),
        Ok(_) => println!("prediction succeeded"),
    }
    model.fit(&x, &array![3.0, 5.0])?;
    println!("prediction: {:.1}", model.predict(&array![[3.0]])?[0]);
    Ok(())
}
```

Output:

```text
fit the model before prediction
prediction: 7.0
```

`Error` is non-exhaustive.
Include a fallback arm when you match its variants.

## Correct common errors

| Error | Action |
| --- | --- |
| `EmptyInput` | Supply a nonempty array. |
| `DimensionMismatch` | Check sample counts and feature counts. |
| `ShapeMismatch` | Check the complete tensor shape. |
| `InvalidParameter` | Check the constructor or method parameters. |
| `InvalidInput` | Check array ranks, label values, and other input requirements. |
| `NonFinite` | Check for NaN, infinity, or numerical overflow. |
| `NotFitted` | Call `fit` before prediction or transformation. |
| `NotConverged` | Check data scaling and the selected solver settings. |
| `NeuralNetwork(NnError::NotCompiled(_))` | Call `compile` before training or evaluation. |
| `Io` | Check the file path, permissions, and file format. |

`NnError` belongs to `rustyml::neural_network`.
The neural network feature enables the `NeuralNetwork` error variant.
Each method documents the errors that it can return.

## Metric input checks

Metrics return scores directly.
Invalid metric inputs can cause a panic.
Check array lengths, label values, and metric requirements before you call them.
See [Evaluate models](evaluation.md) for common input requirements.
