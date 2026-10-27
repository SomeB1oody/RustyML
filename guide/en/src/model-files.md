# Classical model files

Save a fitted classical model, then restore it with its concrete model type.

This chapter covers `LinearRegression`, `KNN`, `KMeans`, `DBSCAN`, and `PCA` model files.

Enable the `machine_learning` feature. The example also requires `ndarray = "0.17.2"`.

## Save and restore

`save_to_path(&self, path: &str) -> Result<(), Error>` writes the model state.

`Model::load_from_path(path: &str) -> Result<Model, Error>` returns a restored model.

Saving replaces an existing file. Choose a separate path when you need to retain an earlier model.

The example checks an unfitted model, saves a trained model, and restores its predictions. It removes the temporary file afterward.

```rust
use ndarray::array;
use rustyml::error::Error;
use rustyml::prelude::machine_learning::LinearRegression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![1.0, 3.0, 5.0];
    let mut model = LinearRegression::new(true);
    assert!(matches!(model.predict(&x), Err(Error::NotFitted(_))));
    model.fit(&x, &y)?;

    let path = std::env::temp_dir().join(format!("rustyml-guide-{}.bin", std::process::id()));
    let filename = path.to_str().ok_or("temporary path is not UTF-8")?;
    model.save_to_path(filename)?;
    let restored = LinearRegression::load_from_path(filename)?;
    let predictions = restored.predict(&array![[3.0]])?;
    println!("restored prediction: {:.1}", predictions[0]);
    std::fs::remove_file(path)?;
    Ok(())
}
```

Output:

```text
restored prediction: 7.0
```

## Keep compatible inputs

A restored model still expects the original feature count and feature order. Preserve label meanings and preprocessing settings separately.

Saving an unfitted model does not train it. Its predictions still require training.

Load the same concrete type that created the file. For KNN, also preserve the label type, such as `KNN<i32>`.

KNN persistence requires labels that implement serde serialization and deserialization. Integer labels satisfy these requirements.

Keep the crate version with each saved model. Check compatibility before you load files from another version.

## Handle file errors

File access and serialization failures return `Error::Io`. Propagate the error or report it with the file path.

A successful load restores stored state. It does not validate your application's feature names or preprocessing choices.

Retain training data and configuration when you need to retrain a model.
