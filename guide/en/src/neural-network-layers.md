# Neural network layers

Choose layers by the shape of each sample and the output your task needs.
All tensors use `f32` values and place the batch axis first.
The neural network prelude exports the layers below.

## Common shape rules

| Layer family | Input | Output |
| --- | --- | --- |
| `Dense(units)` | `[batch, ..., features]`, rank at least two | `[batch, ..., units]` |
| `Conv1D` | `[batch, length, channels]` | `[batch, output_length, filters]` |
| `Conv2D` | `[batch, height, width, channels]` | `[batch, output_height, output_width, filters]` |
| `Conv3D` | `[batch, depth, height, width, channels]` | Same axis order with new spatial sizes and filters |
| Local pooling | Convolution axis order | Reduced spatial axes, unchanged channels |
| `Flatten` | `[batch, ...]` | `[batch, product_of_remaining_axes]` |
| Recurrent, default | `[batch, time, features]` | `[batch, units]` |
| Recurrent, return sequences | `[batch, time, features]` | `[batch, time, units]` |

These rows describe shape behavior, not constructor syntax.
Image and convolution tensors place channels last.
For grayscale images, keep the final channel dimension equal to one.
A single image still needs a batch axis.

## Dense layers

Create a dense layer with `Dense::new(units, activation)?`.
Use `Activation::ReLU` for a simple hidden layer.
Choose the final activation to match your loss.
See [output and loss choices](neural-networks.md#choose-outputs-and-loss).

Dense replaces only the last axis and preserves all leading axes.
For example, `[8, 12, 6]` becomes `[8, 12, 4]` with four units.
Add `Flatten::new()` first when you need one vector per image.
Without flattening, Dense processes each leading position separately.

## Convolution and pooling

`Conv2D::new(filters, kernel_size, strides, activation)?` configures a two-dimensional convolution.
Its default padding is `PaddingType::Valid`.
Call `.with_padding(PaddingType::Same)` to request padding that preserves spatial size when the stride equals one.
Use `.with_dilation_rate(...)` when you need a larger effective kernel.

For each spatial axis, let `effective_kernel = dilation * (kernel - 1) + 1`.
With valid padding, the output size is `floor((input - effective_kernel) / stride) + 1`.
The effective kernel must fit the input for valid padding.
With same padding, the output size is `ceil(input / stride)`.
Apply these rules separately to each spatial axis.

`MaxPooling2D::new((2, 2))` keeps the largest value in each window.
`AveragePooling2D` uses window averages.
Local pooling preserves the channel count.
Pool strides default to the pool size, with valid padding.
Override them using `with_strides(...)?` and `with_padding(...)`.
Global pooling reduces all spatial positions to one value per channel.

The image example below transforms `[2, 8, 8, 1]` through these shapes:

1. Valid convolution produces `[2, 6, 6, 4]`.
2. Pooling produces `[2, 3, 3, 4]`.
3. Flatten produces `[2, 36]`.
4. Dense produces `[2, 3]`.

## Recurrent layers

Use `SimpleRNN`, `LSTM`, or `GRU` for ordered sequences.
Each constructor takes `units` and an activation, such as `LSTM::new(4, Activation::Tanh)?`.
Each expects `[batch, time, features]` input.
The default output contains the final hidden state, shaped `[batch, units]`.
Call `.with_return_sequences(true)` to return one hidden state per timestep.

Enable sequence output before stacking another recurrent layer.
A following Dense layer then maps every timestep separately.
For one prediction per sequence, keep the final recurrent layer default output.
Call `.with_go_backwards(true)` to process timesteps in reverse order.

## Run image and sequence stacks

Prediction does not require an optimizer or loss.
This example uses zero inputs to verify both stack shapes.
The source also lives in `guide/examples/nn_shapes.rs`.

```rust
use ndarray::Array;
use rustyml::prelude::neural_network::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images = Array::<f32, _>::zeros((2, 8, 8, 1)).into_dyn();
    let cnn = SequentialBuilder::new()
        .add(Conv2D::new(4, (3, 3), (1, 1), Activation::ReLU)?.with_random_state(11))
        .add(MaxPooling2D::new((2, 2)))
        .add(Flatten::new())
        .add(Dense::new(3, Activation::Linear)?.with_random_state(12))
        .build(&Shape::known(images.shape()))?;
    println!("image predictions: {:?}", cnn.predict(&images)?.shape());

    let sequence = Array::<f32, _>::zeros((2, 5, 3)).into_dyn();
    let rnn = SequentialBuilder::new()
        .add(
            LSTM::new(4, Activation::Tanh)?
                .with_random_state(13)
                .with_return_sequences(true),
        )
        .add(Dense::new(2, Activation::Linear)?.with_random_state(14))
        .build(&Shape::known(sequence.shape()))?;
    println!(
        "sequence predictions: {:?}",
        rnn.predict(&sequence)?.shape()
    );
    Ok(())
}
```

Expected output:

```text
image predictions: [2, 3]
sequence predictions: [2, 5, 2]
```

## Fix shape errors

Compare every layer output with the next layer input requirements.
Use `model.summary()` to inspect the complete stack.
Build with the actual input tensor shape before training.
Check channel order when convolution rejects image input.
Check `with_return_sequences` when recurrent layers reject a preceding layer output.
Check target dimensions against the selected loss before fitting.
