# Neural networks

Use `SequentialBuilder` when each layer takes the previous layer output.
Import the neural network prelude with `use rustyml::prelude::neural_network::*`.
Enable the `neural_network` Cargo feature.
Use `ndarray = "0.17.2"` for your input arrays.

## Prepare tensors

`Tensor` means `ndarray::ArrayD<f32>`.
Convert arrays with `.into_dyn()` and store values as `f32`.
Axis zero contains samples.
For tabular data, use `[samples, features]`.
Use `[samples, outputs]` for regression targets.
Training inputs and targets must contain the same number of samples.

Scale features consistently before training and prediction.
Keep evaluation samples separate from training samples.

## Build and train

1. Create layers and add them in order.
2. Call `build(&Shape::known(x.shape()))` to validate shapes and initialize weights.
3. Call `compile(optimizer, loss)` before training or evaluation.
4. Call `fit`, `fit_with_batches`, or `train_batch`.
5. Call `predict` to produce output tensors.

`fit(&x, &y, epochs)` uses the complete dataset for one update per epoch.
`fit_with_batches(&x, &y, epochs, batch_size)` shuffles samples each epoch and updates each batch.
Use a positive batch size that does not exceed the sample count.
`train_batch(&x, &y)` performs one update without splitting or shuffling.

Both fitting methods return `History`.
`history.loss()` contains one loss per epoch, measured during training.
`evaluate(&x, &y)` measures the current model loss without updating weights.
`predict(&x)` and `evaluate` use inference behavior for dropout and normalization.
`predict` works without `compile`.
The batch size can change after building when the other required dimensions remain compatible.

## Choose outputs and loss

| Task | Last activation | Loss | Target shape |
| --- | --- | --- | --- |
| Regression | `Activation::Linear` | `MeanSquaredError::new()` or `MeanAbsoluteError::new()` | Same as predictions |
| Binary classification | `Activation::Sigmoid` | `BinaryCrossEntropy::new()` | `[samples, 1]`, values 0 or 1 |
| Class probabilities | `Activation::Softmax { axis: -1 }` | `CategoricalCrossEntropy::new(false)` | `[samples, classes]`, one-hot values |
| Class logits | `Activation::Linear` | `SparseCategoricalCrossEntropy::new(true)` | `[samples, 1]`, class indices |

Sparse labels still use `f32`, such as `0.0` and `2.0`.
Use whole class indices from zero through `classes - 1`.
For categorical losses, `true` means raw logits and `false` means probabilities.
Do not apply softmax before a loss configured for logits.

Start with `SGD` for a simple learning rate and optional momentum.
`SGD::new(0.05, 0.0, false, 0.0)` disables momentum, Nesterov updates, and weight decay.
`Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0)` provides adaptive updates.
`AdamW` provides decoupled weight decay.
`RMSprop` and `AdaGrad` provide other adaptive update choices.
Use `set_learning_rate` to change the compiled optimizer learning rate.

## Train, save, and restore

This example learns `y = 2x + 1` from four samples.
It checks training progress and verifies exact predictions after loading.
The source also lives in `guide/examples/nn_regression.rs`.

```rust
use ndarray::array;
use rustyml::prelude::neural_network::*;

fn build_model() -> Result<Sequential, rustyml::error::Error> {
    SequentialBuilder::new_with_seed(42)
        .add(Dense::new(1, Activation::Linear)?.with_random_state(7))
        .build(&Shape::known(&[4, 1]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[-1.0_f32], [0.0], [1.0], [2.0]].into_dyn();
    let y = array![[-1.0_f32], [1.0], [3.0], [5.0]].into_dyn();
    let mut model = build_model()?;
    model.compile(SGD::new(0.05, 0.0, false, 0.0)?, MeanSquaredError::new());
    let initial_loss = model.evaluate(&x, &y)?;
    let history = model.fit_with_batches(&x, &y, 200, 2)?;
    let final_loss = model.evaluate(&x, &y)?;
    assert!(final_loss < initial_loss);
    assert!(final_loss < 0.0001);
    println!("epochs: {}", history.loss().len());
    println!("loss decreased: {}", final_loss < initial_loss);

    let path = std::env::temp_dir().join(format!("rustyml-nn-{}.bin", std::process::id()));
    model.save_to_path(&path)?;
    let mut restored = build_model()?;
    restored.load_from_path(&path)?;
    let prediction = restored.predict(&x)?;
    assert_eq!(model.predict(&x)?, prediction);
    println!("prediction shape: {:?}", prediction.shape());
    println!("restored predictions match: true");
    std::fs::remove_file(path)?;
    Ok(())
}
```

Expected output:

```text
epochs: 200
loss decreased: true
prediction shape: [4, 1]
restored predictions match: true
```

`save_to_path` writes trainable arrays and non-trainable arrays, including normalization statistics.
It does not save the architecture, layer configuration, optimizer, or loss.
Rebuild the same architecture before calling `load_from_path`.
The strict loader checks layer types, array names, array shapes, and compatible build shapes.
The checkpoint permits a different batch size.
Keep the same layer configuration because the checkpoint does not restore it.

Call `compile` after loading before training or evaluation.
Training starts with new optimizer state, so loading does not resume optimizer history.
Prediction needs no compilation, as the example shows.
Use `load_partial_from_path` only when you intend to load matching arrays and inspect its report.

## Repeat a training run

Set each randomized layer seed with `with_random_state` before building.
Set the batch shuffle seed with `SequentialBuilder::new_with_seed` or `model.set_seed`.
The builder seed controls shuffling, not layer initialization.
Full-batch `fit` does not shuffle.

Alternatively, call `rustyml::random::set_global_seed` before creating randomized components.
Reset that seed before repeating the same operation sequence on the same thread.
Random operations without local seeds share that thread-local stream.
Keep input data, preprocessing, operation order, and training settings fixed.
Floating-point results can vary across hardware and execution environments.

Continue with [layer shapes and common stacks](neural-network-layers.md).
