# Installation

RustyML 0.17.0 requires Rust 1.89.0 or later.
The examples use Rust edition 2024 and ndarray 0.17.2.

## Create a project

1. Create a Cargo project.

   ```sh
   cargo new rustyml-example
   ```

2. Open the project directory.

   ```sh
   cd rustyml-example
   ```

3. Add these dependencies to `Cargo.toml`.

   ```toml
   [dependencies]
   rustyml = "=0.17.0"
   ndarray = "=0.17.2"
   ```

4. Replace `src/main.rs` with the program in [Your first model](first-model.md).

5. Run the program.

   ```sh
   cargo run
   ```

RustyML enables all public modules by default.
Declare ndarray directly because the examples import its arrays and macros.

## Select features

To select modules, disable default features and list the features you need.
For example, this dependency supports classical models, data utilities, and evaluation metrics.

```toml
[dependencies]
rustyml = { version = "=0.17.0", default-features = false, features = ["machine_learning", "utils", "metrics"] }
ndarray = "=0.17.2"
```

| Feature | Public APIs |
| --- | --- |
| `machine_learning` | Regression, classification, clustering, and dimensionality reduction |
| `neural_network` | Models, layers, losses, and optimizers |
| `utils` | Data splits, scalers, and label conversion |
| `metrics` | Regression, classification, and clustering scores |
| `math` | Distance functions and reduction functions |
| `full` | All five modules above |
| `show_progress` | Training progress bars in supported operations |

The default feature enables `full`.
Each of `machine_learning`, `neural_network`, `utils`, and `metrics` also enables `math`.
Cargo features add APIs to the build.
The `show_progress` feature does not enable a model module.

## Import APIs

The examples use module paths to identify each API.
For shorter imports, use a category prelude.

```rust
use ndarray::array;
use rustyml::prelude::machine_learning::*;

fn main() -> Result<(), rustyml::error::Error> {
    let x = array![[1.0], [2.0]];
    let y = array![3.0, 5.0];
    let mut model = LinearRegression::new(true);
    model.fit(&x, &y)?;
    println!("prediction: {:.1}", model.predict(&array![[3.0]])?[0]);
    Ok(())
}
```

Output:

```text
prediction: 7.0
```

Other category preludes are `utils`, `metrics`, and `neural_network` under `rustyml::prelude`.
The root prelude exports APIs from every enabled category.
With `metrics` enabled, its `Average` name refers to the metrics enum.
Use `rustyml::prelude::neural_network::Average` for the merge layer.
